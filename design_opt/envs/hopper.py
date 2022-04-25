import numpy as np
from gym import utils
from khrylib.rl.envs.common.mujoco_env_gym import MujocoEnv
from khrylib.robot.xml_robot import Robot
from khrylib.utils import get_single_body_qposaddr, get_graph_fc_edges
from copy import deepcopy
import mujoco_py
import time


class HopperEnv(MujocoEnv, utils.EzPickle):
    def __init__(self, cfg, agent):
        self.cur_t = 0
        self.cfg = cfg
        self.agent = agent
        self.model_xml_file = 'assets/mujoco_envs/hopper.xml'
        # robot xml
        self.robot = Robot(cfg.robot_cfg, xml=self.model_xml_file)
        self.init_xml_str = self.robot.export_xml_string()
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        # design options
        self.clip_qvel = cfg.obs_specs.get('clip_qvel', False)
        self.use_projected_params = cfg.obs_specs.get('use_projected_params', True)
        self.abs_design = cfg.obs_specs.get('abs_design', False)
        self.use_body_ind = cfg.obs_specs.get('use_body_ind', False)
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()
        self.design_param_names = self.robot.get_params(get_name=True)
        self.attr_design_dim = self.design_ref_params.shape[-1]
        self.index_base = self.cfg.add_body_condition.get('max_nchild', 3) + 1
        self.stage = 'skeleton_transform'    # transform or execute
        self.control_nsteps = 0
        self.sim_specs = set(cfg.obs_specs.get('sim', []))
        self.attr_specs = set(cfg.obs_specs.get('attr', []))
        MujocoEnv.__init__(self, self.model_xml_file, 4)
        utils.EzPickle.__init__(self)
        self.control_action_dim = 1
        self.skel_num_action = 3 if cfg.enable_remove else 2
        self.sim_obs_dim = self.get_sim_obs().shape[-1]
        self.attr_fixed_dim = self.get_attr_fixed().shape[-1]
        self.ground_geoms = np.where(self.model.geom_bodyid == 0)[0]

    def allow_add_body(self, body):
        add_body_condition = self.cfg.add_body_condition
        max_nchild = add_body_condition.get('max_nchild', 3)
        min_nchild = add_body_condition.get('min_nchild', 0)
        return body.depth >= self.cfg.min_body_depth and body.depth < self.cfg.max_body_depth - 1 and len(body.child) < max_nchild and len(body.child) >= min_nchild
    
    def allow_remove_body(self, body):
        if body.depth >= self.cfg.min_body_depth + 1 and len(body.child) == 0:
            if body.depth == 1:
                return body.parent.child.index(body) > 0
            else:
                return True
        return False

    def apply_skel_action(self, skel_action):
        bodies = list(self.robot.bodies)
        for body, a in zip(bodies, skel_action):
            if a == 1 and self.allow_add_body(body):
                self.robot.add_child_to_body(body)
            if a == 2 and self.allow_remove_body(body):
                self.robot.remove_body(body)

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            self.reload_sim_model(xml_str.decode('utf-8'))
        except:
            print(self.cur_xml_str)
            return False      
        self.design_cur_params = self.get_attr_design()
        return True

    def set_design_params(self, in_design_params):
        design_params = in_design_params
        for params, body in zip(design_params, self.robot.bodies):
            body.set_params(params, pad_zeros=True, map_params=True)
            # new_params = body.get_params([], pad_zeros=True, demap_params=True)
            body.sync_node()

        xml_str = self.robot.export_xml_string()
        self.cur_xml_str = xml_str.decode('utf-8')
        try:
            self.reload_sim_model(xml_str.decode('utf-8'))
        except:
            print(self.cur_xml_str)
            return False
        if self.use_projected_params:
            self.design_cur_params = self.get_attr_design()
        else:
            self.design_cur_params = in_design_params.copy()
        return True

    def action_to_control(self, a):
        ctrl = np.zeros_like(self.data.ctrl)
        assert a.shape[0] == len(self.robot.bodies)
        for body, body_a in zip(self.robot.bodies[1:], a[1:]):
            aname = body.get_actuator_name()
            aind = self.model.actuator_names.index(aname)
            ctrl[aind] = body_a
        return ctrl        

    def step(self, a):
        if not self.is_inited:
            return self._get_obs(), 0, False, {'use_transform_action': False, 'stage': 'execution'}

        self.cur_t += 1
        # skeleton transform stage
        if self.stage == 'skeleton_transform':
            skel_a = a[:, -1]
            succ = self.apply_skel_action(skel_a)
            if not succ:
                return self._get_obs(), 0.0, True, {'use_transform_action': True, 'stage': 'skeleton_transform'}

            if self.cur_t == self.cfg.skel_transform_nsteps:
                self.transit_attribute_transform()

            ob = self._get_obs()
            reward = 0.0
            done = False
            return ob, reward, done, {'use_transform_action': True, 'stage': 'skeleton_transform'}
        # attribute transform stage
        elif self.stage == 'attribute_transform':
            design_a = a[:, self.control_action_dim:-1] 
            if self.abs_design:
                design_params = design_a * self.cfg.robot_param_scale
            else:
                design_params = self.design_cur_params + design_a * self.cfg.robot_param_scale
            succ = self.set_design_params(design_params)
            if not succ:
                return self._get_obs(), 0.0, True, {'use_transform_action': True, 'stage': 'attribute_transform'}

            if self.cur_t == self.cfg.skel_transform_nsteps + 1:
                succ = self.transit_execution()
                if not succ:
                    return self._get_obs(), 0.0, True, {'use_transform_action': True, 'stage': 'attribute_transform'}

            ob = self._get_obs()
            reward = 0.0
            done = False
            return ob, reward, done, {'use_transform_action': True, 'stage': 'attribute_transform'}
        # execution stage
        else:
            self.control_nsteps += 1
            assert np.all(a[:, self.control_action_dim:] == 0)
            control_a = a[:, :self.control_action_dim]
            ctrl = self.action_to_control(control_a)
            posbefore = self.sim.data.qpos[0]

            try:
                self.do_simulation(ctrl, self.frame_skip)
            except:
                print(self.cur_xml_str)
                return self._get_obs(), 0, True, {'use_transform_action': False, 'stage': 'execution'}

            posafter, height, ang = self.sim.data.qpos[0:3]
            alive_bonus = self.cfg.reward_specs.get('alive_bonus', 1.0)
            reward = (posafter - posbefore) / self.dt
            reward += alive_bonus
            scale = self.cfg.reward_specs.get('exec_reward_scale', 1.0)
            reward *= scale

            s = self.state_vector()
            # misc
            done_condition = self.cfg.done_condition
            min_height = done_condition.get('min_height', 0.7)
            max_height = done_condition.get('max_height', 2.0)
            max_ang = done_condition.get('max_ang', 3600)
            max_nsteps = done_condition.get('max_nsteps', 1000)
            done = not (np.isfinite(s).all() and (height > min_height) and (height < max_height) and (abs(ang) < np.deg2rad(max_ang)) and (self.control_nsteps < max_nsteps))
            ob = self._get_obs()
            return ob, reward, done, {'use_transform_action': False, 'stage': 'execution'}
    
    def transit_attribute_transform(self):
        self.stage = 'attribute_transform'

    def transit_execution(self):
        self.stage = 'execution'
        self.control_nsteps = 0
        try:
            self.reset_state(True)
        except:
            print(self.cur_xml_str)
            return False
        return True
        

    def if_use_transform_action(self):
        return ['skeleton_transform', 'attribute_transform', 'execution'].index(self.stage)

    def get_sim_obs(self):
        obs = []
        if 'root_offset' in self.sim_specs:
            root_pos = self.data.body_xpos[self.model._body_name2id[self.robot.bodies[0].name]]
            
        for i, body in enumerate(self.robot.bodies):
            qvel = self.data.qvel.copy()
            if self.clip_qvel:
                qvel = np.clip(qvel, -10, 10)
            if i == 0:
                obs_i = [np.flip(self.data.qpos[1:3]), np.flip(qvel[:3])]
            else:
                qs, qe = get_single_body_qposaddr(self.model, body.name)
                assert qe - qs == 1
                obs_i = [self.data.qpos[qs:qe], np.zeros(1), qvel[qs:qe], np.zeros(2)]
                # print(qs)
            if 'root_offset' in self.sim_specs:
                offset = self.data.body_xpos[self.model._body_name2id[body.name]][[0, 2]] - root_pos[[0, 2]]
                obs_i.append(offset)
            obs_i = np.concatenate(obs_i)
            obs.append(obs_i)
        obs = np.stack(obs)
        if self.control_nsteps == 1:
            assert np.count_nonzero(obs[:, :5]) == self.model.nq + self.model.nv - 1
        return obs

    def get_attr_fixed(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = []
            if 'depth' in self.attr_specs:
                obs_depth = np.zeros(self.cfg.max_body_depth)
                obs_depth[body.depth] = 1.0
                obs_i.append(obs_depth)
            if 'jrange' in self.attr_specs:
                obs_jrange = body.get_joint_range()
                obs_i.append(obs_jrange)
            if 'skel' in self.attr_specs:
                obs_add = self.allow_add_body(body)
                obs_rm = self.allow_remove_body(body)
                obs_i.append(np.array([float(obs_add), float(obs_rm)]))
            if len(obs_i) > 0:
                obs_i = np.concatenate(obs_i)
                obs.append(obs_i)
        
        if len(obs) == 0:
            return None
        obs = np.stack(obs)
        return obs

    def get_attr_design(self):
        obs = []
        for i, body in enumerate(self.robot.bodies):
            obs_i = body.get_params([], pad_zeros=True, demap_params=True)
            obs.append(obs_i)
        obs = np.stack(obs)
        return obs

    def get_body_index(self):
        index = []
        for i, body in enumerate(self.robot.bodies):
            ind = int(body.name, base=self.index_base)
            index.append(ind)
        index = np.array(index)
        return index

    def _get_obs(self):
        obs = []
        attr_fixed_obs = self.get_attr_fixed()
        sim_obs = self.get_sim_obs()
        design_obs = self.design_cur_params
        obs = np.concatenate(list(filter(lambda x: x is not None, [attr_fixed_obs, sim_obs, design_obs])), axis=-1)
        if self.cfg.obs_specs.get('fc_graph', False):
            edges = get_graph_fc_edges(len(self.robot.bodies))
        else:
            edges = self.robot.get_gnn_edges()
        use_transform_action = np.array([self.if_use_transform_action()])
        num_nodes = np.array([sim_obs.shape[0]])
        all_obs = [obs, edges, use_transform_action, num_nodes]
        if self.use_body_ind:
            body_index = self.get_body_index()
            all_obs.append(body_index)
        return all_obs

    def reset_state(self, add_noise):
        if add_noise:
            qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
            qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        else:
            qpos = self.init_qpos
            qvel = self.init_qvel

        if self.stage == 'execution' and self.cfg.env_init_height:
            qpos[1] = 0.0
            while True:
                self.set_state(qpos, qvel)
                has_contact = False
                for contact in self.data.contact[:self.data.ncon]:
                    g1, g2 = contact.geom1, contact.geom2
                    # print(f'g1: {g1} g2: {g2}')
                    if g1 in self.ground_geoms or g2 in self.ground_geoms:
                        has_contact = True
                        break
                if has_contact:
                    qpos[1] += 0.05
                else:
                    break
        else:
            self.set_state(qpos, qvel)

    def reset_robot(self):
        del self.robot
        self.robot = Robot(self.cfg.robot_cfg, xml=self.init_xml_str, is_xml_str=True)
        self.cur_xml_str = self.init_xml_str.decode('utf-8')
        self.reload_sim_model(self.cur_xml_str)
        self.design_ref_params = self.get_attr_design()
        self.design_cur_params = self.design_ref_params.copy()

    def reset_model(self):
        self.reset_robot()
        self.control_nsteps = 0
        self.stage = 'skeleton_transform'
        self.cur_t = 0
        self.reset_state(False)
        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = 4
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.lookat[0] = self.data.qpos[0] 
        self.viewer.cam.elevation = -20