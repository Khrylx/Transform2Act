import math
import pickle
import time
from khrylib.utils import *
from khrylib.utils.torch import *
from khrylib.rl.agents import AgentPPO
from khrylib.rl.core import estimate_advantages
from torch.utils.tensorboard import SummaryWriter
from design_opt.envs import env_dict
from design_opt.models.transform2act_policy import Transform2ActPolicy
from design_opt.models.transform2act_critic import Transform2ActValue
from design_opt.utils.logger import LoggerRLV1
from design_opt.utils.tools import TrajBatchDisc


def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) if i < 2 else x for i, x in enumerate(y)] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]


class Transform2ActAgent(AgentPPO):

    def __init__(self, cfg, dtype, device, seed, num_threads, training=True, checkpoint=0):
        self.cfg = cfg
        self.training = training
        self.device = device
        self.loss_iter = 0
        self.setup_env()
        self.seed(seed)
        self.setup_logger()
        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()
        self.setup_param_scheduler()
        if checkpoint != 0:
            self.load_checkpoint(checkpoint)
        super().__init__(env=self.env, dtype=dtype, device=device, running_state=self.running_state,
                         custom_reward=None, logger_cls=LoggerRLV1, traj_cls=TrajBatchDisc, num_threads=num_threads,
                         policy_net=self.policy_net, value_net=self.value_net,
                         optimizer_policy=self.optimizer_policy, optimizer_value=self.optimizer_value, opt_num_epochs=cfg.num_optim_epoch,
                         gamma=cfg.gamma, tau=cfg.tau, clip_epsilon=cfg.clip_epsilon,
                         policy_grad_clip=[(self.policy_net.parameters(), 40)],
                         use_mini_batch=cfg.mini_batch_size < cfg.min_batch_size, mini_batch_size=cfg.mini_batch_size)

    def sample_worker(self, pid, queue, min_batch_size, mean_action, render):
        self.seed_worker(pid)
        memory = Memory()
        logger = self.logger_cls(**self.logger_kwargs)

        while logger.num_steps < min_batch_size:
            state = self.env.reset()
            if self.running_state is not None:
                state = self.running_state(state)
            logger.start_episode(self.env)
            self.pre_episode()

            for t in range(10000):
                state_var = tensorfy([state])
                use_mean_action = mean_action or torch.bernoulli(torch.tensor([1 - self.noise_rate])).item()
                action = self.policy_net.select_action(state_var, use_mean_action).numpy().astype(np.float64)
                next_state, env_reward, done, info = self.env.step(action)
                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                # use custom or env reward
                if self.custom_reward is not None:
                    c_reward, c_info = self.custom_reward(self.env, state, action, env_reward, info)
                    reward = c_reward
                else:
                    c_reward, c_info = 0.0, np.array([0.0])
                    reward = env_reward
                # add end reward
                if self.end_reward and info.get('end', False):
                    reward += self.env.end_reward
                # logging
                logger.step(self.env, env_reward, c_reward, c_info, info)

                mask = 0 if done else 1
                exp = 1 - use_mean_action
                self.push_memory(memory, state, action, mask, next_state, reward, exp)

                if pid == 0 and render:
                    if t < 10:
                        self.env._get_viewer('human')._paused = True
                    self.env.render()
                if done:
                    break
                state = next_state

            logger.end_episode(self.env)
        logger.end_sampling()

        if queue is not None:
            queue.put([pid, memory, logger])
        else:
            return memory, logger
            
    def setup_env(self):
        env_class = env_dict[self.cfg.env_name]
        self.env = env = env_class(self.cfg, self)
        self.attr_fixed_dim = env.attr_fixed_dim
        self.attr_design_dim = env.attr_design_dim
        self.sim_obs_dim = env.sim_obs_dim
        self.state_dim = self.attr_fixed_dim + self.sim_obs_dim + self.attr_design_dim
        self.control_action_dim = env.control_action_dim
        self.skel_num_action = env.skel_num_action
        self.action_dim = self.control_action_dim + self.attr_design_dim
        self.running_state = None

    def seed(self, seed):
        self.env.seed(seed)

    def setup_policy(self):
        cfg = self.cfg
        self.policy_net = Transform2ActPolicy(cfg.policy_specs, self)
        to_device(self.device, self.policy_net)

    def setup_value(self):
        cfg = self.cfg
        self.value_net = Transform2ActValue(cfg.value_specs, self)
        to_device(self.device, self.value_net)

    def setup_optimizer(self):
        cfg = self.cfg
        # policy optimizer
        if cfg.policy_optimizer == 'Adam':
            self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.policy_lr, weight_decay=cfg.policy_weightdecay)
        else:
            self.optimizer_policy = torch.optim.SGD(self.policy_net.parameters(), lr=cfg.policy_lr, momentum=cfg.policy_momentum, weight_decay=cfg.policy_weightdecay)
        # value optimizer
        if cfg.value_optimizer == 'Adam':
            self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=cfg.value_lr, weight_decay=cfg.value_weightdecay)
        else:
            self.optimizer_value = torch.optim.SGD(self.value_net.parameters(), lr=cfg.value_lr, momentum=cfg.value_momentum, weight_decay=cfg.value_weightdecay)

    def setup_param_scheduler(self):
        self.scheduled_params = {}
        for name, specs in self.cfg.scheduled_params.items():
            if specs['type'] == 'step':
                self.scheduled_params[name] = StepParamScheduler(specs['start_val'], specs['step_size'], specs['gamma'], specs.get('smooth', False))
            elif specs['type'] == 'linear':
                self.scheduled_params[name] = LinearParamScheduler(specs['start_val'], specs['end_val'], specs['start_epoch'], specs['end_epoch'])

    def setup_logger(self):
        cfg = self.cfg
        self.tb_logger = SummaryWriter(cfg.tb_dir) if self.training else None
        self.logger = create_logger(os.path.join(cfg.log_dir, f'log_{"train" if self.training else "eval"}.txt'), file_handle=True)
        self.best_rewards = -1000.0
        self.save_best_flag = False

    def load_checkpoint(self, checkpoint):
        cfg = self.cfg
        if isinstance(checkpoint, int):
            cp_path = '%s/epoch_%04d.p' % (cfg.model_dir, checkpoint)
            epoch = checkpoint
        else:
            assert isinstance(checkpoint, str)
            cp_path = '%s/%s.p' % (cfg.model_dir, checkpoint)
        self.logger.info('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        self.policy_net.load_state_dict(model_cp['policy_dict'])
        self.value_net.load_state_dict(model_cp['value_dict'])
        self.running_state = model_cp['running_state']
        self.loss_iter = model_cp['loss_iter']
        self.best_rewards = model_cp.get('best_rewards', self.best_rewards)
        if 'epoch' in model_cp:
            epoch = model_cp['epoch']
        self.pre_epoch_update(epoch)
    
    def save_checkpoint(self, epoch):

        def save(cp_path):
            with to_cpu(self.policy_net, self.value_net):
                model_cp = {'policy_dict': self.policy_net.state_dict(),
                            'value_dict': self.value_net.state_dict(),
                            'running_state': self.running_state,
                            'loss_iter': self.loss_iter,
                            'best_rewards': self.best_rewards,
                            'epoch': epoch}
                pickle.dump(model_cp, open(cp_path, 'wb'))

        cfg = self.cfg
        additional_saves = self.cfg.agent_specs.get('additional_saves', None)
        if (cfg.save_model_interval > 0 and (epoch+1) % cfg.save_model_interval == 0) or \
           (additional_saves is not None and (epoch+1) % additional_saves[0] == 0 and epoch+1 <= additional_saves[1]):
            self.tb_logger.flush()
            save('%s/epoch_%04d.p' % (cfg.model_dir, epoch + 1))
        if self.save_best_flag:
            self.tb_logger.flush()
            self.logger.info(f'save best checkpoint with rewards {self.best_rewards:.2f}!')
            save('%s/best.p' % cfg.model_dir)
            
    def pre_epoch_update(self, epoch):
        for param in self.scheduled_params.values():
            param.set_epoch(epoch)

    def optimize(self, epoch):
        self.pre_epoch_update(epoch)
        info = self.optimize_policy(epoch)
        self.log_optimize_policy(epoch, info)

    def optimize_policy(self, epoch):
        """generate multiple trajectories that reach the minimum batch_size"""
        t0 = time.time()
        batch, log = self.sample(self.cfg.min_batch_size)

        """update networks"""
        t1 = time.time()
        self.update_params(batch)
        t2 = time.time()

        """evaluate policy"""
        _, log_eval = self.sample(self.cfg.eval_batch_size, mean_action=True)
        t3 = time.time() 

        info = {
            'log': log, 'log_eval': log_eval, 'T_sample': t1 - t0, 'T_update': t2 - t1, 'T_eval': t3 - t2, 'T_total': t3 - t0
        }
        return info

    def update_params(self, batch):
        t0 = time.time()
        to_train(*self.update_modules)
        states = tensorfy(batch.states, self.device)
        actions = tensorfy(batch.actions, self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        masks = torch.from_numpy(batch.masks).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = []
                chunk = 10000
                for i in range(0, len(states), chunk):
                    states_i = states[i:min(i + chunk, len(states))]
                    values_i = self.value_net(self.trans_value(states_i))
                    values.append(values_i)
                values = torch.cat(values)

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, masks, values, self.gamma, self.tau)

        if self.cfg.agent_specs.get('reinforce', False):
            advantages = returns.clone()

        self.update_policy(states, actions, returns, advantages, exps)

        return time.time() - t0

    def get_perm_batch_design(self, states):
        inds = [[], [], []]
        for i, x in enumerate(states):
            use_transform_action = x[2]
            inds[use_transform_action.item()].append(i)
        perm = np.array(inds[0] + inds[1] + inds[2])
        return perm, LongTensor(perm).to(self.device)

    def update_policy(self, states, actions, returns, advantages, exps):
        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = []
                chunk = 10000
                for i in range(0, len(states), chunk):
                    states_i = states[i:min(i + chunk, len(states))]
                    actions_i = actions[i:min(i + chunk, len(states))]
                    fixed_log_probs_i = self.policy_net.get_log_prob(self.trans_policy(states_i), actions_i)
                    fixed_log_probs.append(fixed_log_probs_i)
                fixed_log_probs = torch.cat(fixed_log_probs)
        num_state = len(states)

        for _ in range(self.opt_num_epochs):
            if self.use_mini_batch:
                perm_np = np.arange(num_state)
                np.random.shuffle(perm_np)
                perm = LongTensor(perm_np).to(self.device)

                states, actions, returns, advantages, fixed_log_probs, exps = \
                    index_select_list(states, perm_np), index_select_list(actions, perm_np), returns[perm].clone(), advantages[perm].clone(), \
                    fixed_log_probs[perm].clone(), exps[perm].clone()

                if self.cfg.agent_specs.get('batch_design', False):
                    perm_design_np, perm_design = self.get_perm_batch_design(states)
                    states, actions, returns, advantages, fixed_log_probs, exps = \
                        index_select_list(states, perm_design_np), index_select_list(actions, perm_design_np), returns[perm_design].clone(), advantages[perm_design].clone(), \
                        fixed_log_probs[perm_design].clone(), exps[perm_design].clone()

                optim_iter_num = int(math.floor(num_state / self.mini_batch_size))
                for i in range(optim_iter_num):
                    ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, num_state))
                    states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                        states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind], exps[ind]
                    self.update_value(states_b, returns_b)
                    surr_loss = self.ppo_loss(states_b, actions_b, advantages_b, fixed_log_probs_b)
                    self.optimizer_policy.zero_grad()
                    surr_loss.backward()
                    self.clip_policy_grad()
                    self.optimizer_policy.step()
            else:
                ind = exps.nonzero(as_tuple=False).squeeze(1)
                self.update_value(states, returns)
                surr_loss = self.ppo_loss(states, actions, advantages, fixed_log_probs)
                self.optimizer_policy.zero_grad()
                surr_loss.backward()
                self.clip_policy_grad()
                self.optimizer_policy.step()

    def ppo_loss(self, states, actions, advantages, fixed_log_probs):
        log_probs = self.policy_net.get_log_prob(self.trans_policy(states), actions)
        ratio = torch.exp(log_probs - fixed_log_probs)
        advantages = advantages
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        surr_loss = -torch.min(surr1, surr2).mean()
        return surr_loss

    def log_optimize_policy(self, epoch, info):
        cfg = self.cfg
        log, log_eval = info['log'], info['log_eval']
        logger, tb_logger = self.logger, self.tb_logger
        log_str = f'{epoch}\tT_sample {info["T_sample"]:.2f}\tT_update {info["T_update"]:.2f}\tT_eval {info["T_eval"]:.2f}\t'\
            f'ETA {get_eta_str(epoch, cfg.max_epoch_num, info["T_total"])}\ttrain_R {log.avg_reward:.2f}\ttrain_R_eps {log.avg_episode_reward:.2f}\t'\
            f'exec_R {log_eval.avg_exec_reward:.2f}\texec_R_eps {log_eval.avg_exec_episode_reward:.2f}\t{cfg.id}'
        logger.info(log_str)

        if log_eval.avg_exec_episode_reward > self.best_rewards:
            self.best_rewards = log_eval.avg_exec_episode_reward
            self.save_best_flag = True
        else:
            self.save_best_flag = False

        tb_logger.add_scalar('train_R_avg ', log.avg_reward, epoch)
        tb_logger.add_scalar('train_R_eps_avg', log.avg_episode_reward, epoch)
        tb_logger.add_scalar('eval_R_eps_avg', log_eval.avg_episode_reward, epoch)
        tb_logger.add_scalar('exec_R_avg', log_eval.avg_exec_reward, epoch)
        tb_logger.add_scalar('exec_R_eps_avg', log_eval.avg_exec_episode_reward, epoch)

    def visualize_agent(self, num_episode=1, mean_action=True, save_video=False, pause_design=False, max_num_frames=1000):
        fr = 0
        env = self.env
        paused = not save_video and pause_design
        for _ in range(num_episode):
            state = env.reset()
            if self.running_state is not None:
                state = self.running_state(state)

            env._get_viewer('human')._paused = paused
            env.render()
            for t in range(10000):
                state_var = tensorfy([state])
                with torch.no_grad():
                    action = self.policy_net.select_action(state_var, mean_action).numpy().astype(np.float64)
                next_state, env_reward, done, info = env.step(action)
                if self.running_state is not None:
                    next_state = self.running_state(next_state)
                
                if t < self.cfg.skel_transform_nsteps + 1:
                    env._get_viewer('human')._paused = paused
                    env._get_viewer('human')._hide_overlay = save_video
                for _ in range(15 if save_video else 1):
                    env.render()
                if save_video:
                    frame_dir = f'out/videos/{self.cfg.id}_frames'
                    os.makedirs(frame_dir, exist_ok=True)
                    save_screen_shots(env.viewer.window, f'{frame_dir}/%04d.png' % fr)
                    fr += 1
                    if fr >= max_num_frames:
                        break

                if done:
                    break
                state = next_state

            if save_video and fr >= max_num_frames:
                break

        if save_video:
            save_video_ffmpeg(f'{frame_dir}/%04d.png', f'out/videos/{self.cfg.id}.mp4', fps=30)
            shutil.rmtree(frame_dir)
