import yaml
import os
import glob
import numpy as np


class Config:

    def __init__(self, cfg_id, tmp, cfg_dict=None):
        self.id = cfg_id
        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cfg_path = 'design_opt/cfg/**/%s.yml' % cfg_id
            files = glob.glob(cfg_path, recursive=True)
            assert(len(files) == 1)
            cfg = yaml.safe_load(open(files[0], 'r'))
        # create dirs
        base_dir = '/tmp/design_opt' if tmp else 'results'
        self.base_dir = os.path.expanduser(base_dir)

        self.cfg_dir = '%s/%s' % (self.base_dir, cfg_id)
        self.model_dir = '%s/models' % self.cfg_dir
        self.log_dir = '%s/log' % self.cfg_dir
        self.tb_dir = '%s/tb' % self.cfg_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)

        # training config
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.95)
        self.agent_specs = cfg.get('agent_specs', dict())
        self.policy_specs = cfg.get('policy_specs', dict())
        self.policy_optimizer = cfg.get('policy_optimizer', 'Adam')
        self.policy_lr = cfg.get('policy_lr', 5e-5)
        self.policy_momentum = cfg.get('policy_momentum', 0.0)
        self.policy_weightdecay = cfg.get('policy_weightdecay', 0.0)
        self.value_specs = cfg.get('value_specs', dict())
        self.value_optimizer = cfg.get('value_optimizer', 'Adam')
        self.value_lr = cfg.get('value_lr', 3e-4)
        self.value_momentum = cfg.get('value_momentum', 0.0)
        self.value_weightdecay = cfg.get('value_weightdecay', 0.0)
        self.adv_clip = cfg.get('adv_clip', np.inf)
        self.clip_epsilon = cfg.get('clip_epsilon', 0.2)
        self.num_optim_epoch = cfg.get('num_optim_epoch', 10)
        self.min_batch_size = cfg.get('min_batch_size', 50000)
        self.mini_batch_size = cfg.get('mini_batch_size', self.min_batch_size)
        self.eval_batch_size = cfg.get('eval_batch_size', 10000)
        self.max_epoch_num = cfg.get('max_epoch_num', 1000)
        self.seed = cfg.get('seed', 1)
        self.seed_method = cfg.get('seed_method', 'deep')
        self.save_model_interval = cfg.get('save_model_interval', 100)

        # anneal parameters
        self.scheduled_params = cfg.get('scheduled_params', dict())

        # env
        self.env_name = cfg.get('env_name', 'hopper')
        self.done_condition = cfg.get('done_condition', dict())
        self.env_specs = cfg.get('env_specs', dict())
        self.reward_specs = cfg.get('reward_specs', dict())
        self.obs_specs = cfg.get('obs_specs', dict())
        self.add_body_condition = cfg.get('add_body_condition', dict())
        self.max_body_depth = cfg.get('max_body_depth', 4)
        self.min_body_depth = cfg.get('min_body_depth', 1)
        self.enable_remove = cfg.get('enable_remove', True)
        self.skel_transform_nsteps = cfg.get('skel_transform_nsteps', 5)
        self.env_init_height = cfg.get('env_init_height', False)

        # robot config
        self.robot_param_scale = cfg.get('robot_param_scale', 0.1)
        self.robot_cfg = cfg.get('robot', dict())


