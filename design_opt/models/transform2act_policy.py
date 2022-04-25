from collections import defaultdict
from khrylib.utils.torch import LongTensor
import torch.nn as nn
from khrylib.rl.core.distributions import Categorical, DiagGaussian
from khrylib.rl.core.policy import Policy
from khrylib.rl.core.running_norm import RunningNorm
from khrylib.models.mlp import MLP
from khrylib.utils.math import *
from design_opt.utils.tools import *
from design_opt.models.gnn import GNNSimple
from design_opt.models.jsmlp import JSMLP


class Transform2ActPolicy(Policy):
    def __init__(self, cfg, agent):
        super().__init__()
        self.type = 'gaussian'
        self.cfg = cfg
        self.agent = agent
        self.attr_fixed_dim = agent.attr_fixed_dim
        self.sim_obs_dim = agent.sim_obs_dim
        self.attr_design_dim = agent.attr_design_dim
        self.control_state_dim = self.attr_fixed_dim + self.sim_obs_dim + self.attr_design_dim
        self.skel_state_dim = self.attr_fixed_dim + self.attr_design_dim
        self.attr_state_dim = self.attr_fixed_dim + self.attr_design_dim
        self.skel_action_dim = agent.skel_num_action
        self.control_action_dim = agent.control_action_dim
        self.attr_action_dim = agent.attr_design_dim
        self.action_dim = self.control_action_dim + self.attr_action_dim + 1
        self.skel_uniform_prob = cfg.get('skel_uniform_prob', 0.0)

        # skeleton transform
        self.skel_norm = RunningNorm(self.attr_state_dim)
        cur_dim = self.attr_state_dim
        if 'skel_pre_mlp' in cfg:
            self.skel_pre_mlp = MLP(cur_dim, cfg['skel_pre_mlp'], cfg['htype'])
            cur_dim = self.skel_pre_mlp.out_dim
        else:
            self.skel_pre_mlp = None
        if 'skel_gnn_specs' in cfg:
            self.skel_gnn = GNNSimple(cur_dim, cfg['skel_gnn_specs'])
            cur_dim = self.skel_gnn.out_dim
        else:
            self.skel_gnn = None
        if 'skel_mlp' in cfg:
            self.skel_mlp = MLP(cur_dim, cfg['skel_mlp'], cfg['htype'])
            cur_dim = self.skel_mlp.out_dim
        else:
            self.skel_mlp = None
        if 'skel_index_mlp' in cfg:
            imlp_cfg = cfg['skel_index_mlp']
            self.skel_ind_mlp = JSMLP(cur_dim, imlp_cfg['hdims'], self.skel_action_dim, imlp_cfg.get('max_index', 256), imlp_cfg.get('htype', 'tanh'), imlp_cfg['rescale_linear'], imlp_cfg.get('zero_init', False))
            cur_dim = self.skel_ind_mlp.out_dim
        else:
            self.skel_ind_mlp = None
            self.skel_action_logits = nn.Linear(cur_dim, self.skel_action_dim)

        # attribute transform
        self.attr_norm = RunningNorm(self.skel_state_dim) if cfg.get('attr_norm', True) else None
        cur_dim = self.skel_state_dim
        if 'attr_pre_mlp' in cfg:
            self.attr_pre_mlp = MLP(self.skel_state_dim, cfg['attr_pre_mlp'], cfg['htype'])
            cur_dim = self.attr_pre_mlp.out_dim
        else:
            self.attr_pre_mlp = None
        if 'attr_gnn_specs' in cfg:
            self.attr_gnn = GNNSimple(cur_dim, cfg['attr_gnn_specs'])
            cur_dim = self.attr_gnn.out_dim
        else:
            self.attr_gnn = None
        if 'attr_mlp' in cfg:
            self.attr_mlp = MLP(cur_dim, cfg['attr_mlp'], cfg['htype'])
            cur_dim = self.attr_mlp.out_dim
        else:
            self.attr_mlp = None
        if 'attr_index_mlp' in cfg:
            imlp_cfg = cfg['attr_index_mlp']
            self.attr_ind_mlp = JSMLP(cur_dim, imlp_cfg['hdims'], self.attr_action_dim, imlp_cfg.get('max_index', 256), imlp_cfg.get('htype', 'tanh'), imlp_cfg['rescale_linear'], imlp_cfg.get('zero_init', False))
            cur_dim = self.attr_ind_mlp.out_dim
        else:
            self.attr_ind_mlp = None
            self.attr_action_mean = nn.Linear(cur_dim, self.attr_action_dim)
            init_fc_weights(self.attr_action_mean)
        self.attr_action_log_std = nn.Parameter(torch.ones(1, self.attr_action_dim) * cfg['attr_log_std'], requires_grad=not cfg['fix_attr_std'])
        
        # execution
        self.control_norm = RunningNorm(self.control_state_dim)
        cur_dim = self.control_state_dim
        if 'control_pre_mlp' in cfg:
            self.control_pre_mlp = MLP(cur_dim, cfg['control_pre_mlp'], cfg['htype'])
            cur_dim = self.control_pre_mlp.out_dim
        else:
            self.control_pre_mlp = None
        if 'control_gnn_specs' in cfg:
            self.control_gnn = GNNSimple(cur_dim, cfg['control_gnn_specs'])
            cur_dim = self.control_gnn.out_dim
        else:
            self.control_gnn = None
        if 'control_mlp' in cfg:
            self.control_mlp = MLP(cur_dim, cfg['control_mlp'], cfg['htype'])
            cur_dim = self.control_mlp.out_dim
        else:
            self.control_mlp = None
        if 'control_index_mlp' in cfg:
            imlp_cfg = cfg['control_index_mlp']
            self.control_ind_mlp = JSMLP(cur_dim, imlp_cfg['hdims'], self.control_action_dim, imlp_cfg.get('max_index', 256), imlp_cfg.get('htype', 'tanh'), imlp_cfg['rescale_linear'], imlp_cfg.get('zero_init', False))
            cur_dim = self.control_ind_mlp.out_dim
        else:
            self.control_ind_mlp = None
            self.control_action_mean = nn.Linear(cur_dim, self.control_action_dim)
            init_fc_weights(self.control_action_mean)
        self.control_action_log_std = nn.Parameter(torch.ones(1, self.control_action_dim) * cfg['control_log_std'], requires_grad=not cfg['fix_control_std'])

    def batch_data(self, x):
        obs, edges, use_transform_action, num_nodes, body_ind = zip(*x)
        obs = torch.cat(obs)
        use_transform_action = np.concatenate(use_transform_action)
        num_nodes = np.concatenate(num_nodes)
        edges_new = torch.cat(edges, dim=1)
        num_nodes_cum = np.cumsum(num_nodes)
        body_ind = torch.from_numpy(np.concatenate(body_ind))
        if len(x) > 1:
            repeat_num = [x.shape[1] for x in edges[1:]]
            e_offset = np.repeat(num_nodes_cum[:-1], repeat_num)
            e_offset = torch.tensor(e_offset, device=obs.device)
            edges_new[:, -e_offset.shape[0]:] += e_offset
        return obs, edges_new, use_transform_action, num_nodes, num_nodes_cum, body_ind

    def forward(self, x):
        stages = ['skel_trans', 'attr_trans', 'execution']
        x_dict = defaultdict(list)
        node_design_mask = defaultdict(list)
        design_mask = defaultdict(list)
        total_num_nodes = 0
        for i, x_i in enumerate(x):
            num = x_i[-2].item()
            cur_stage = stages[int(x_i[-3].item())]
            x_dict[cur_stage].append(x_i)
            for stage in stages:
                node_design_mask[stage] += [cur_stage == stage] * num
                design_mask[stage].append(cur_stage == stage)
            total_num_nodes += num
        for stage in stages:
            node_design_mask[stage] = torch.BoolTensor(node_design_mask[stage])
            design_mask[stage] = torch.BoolTensor(design_mask[stage])
            
        # execution
        if len(x_dict['execution']) > 0:
            obs, edges, _, num_nodes, num_nodes_cum_control, body_ind = self.batch_data(x_dict['execution'])
            x = self.control_norm(obs)
            if self.control_pre_mlp is not None:
                x = self.control_pre_mlp(x)
            if self.control_gnn is not None:
                x = self.control_gnn(x, edges)
            if self.control_mlp is not None:
                x = self.control_mlp(x)
            if self.control_ind_mlp is not None:
                control_action_mean = self.control_ind_mlp(x, body_ind)
            else:
                control_action_mean = self.control_action_mean(x)
            control_action_std = self.control_action_log_std.expand_as(control_action_mean).exp()
            control_dist = DiagGaussian(control_action_mean, control_action_std)
        else:
            num_nodes_cum_control = None
            control_dist = None
            
        # attribute transform
        if len(x_dict['attr_trans']) > 0:
            obs, edges, _, num_nodes, num_nodes_cum_design, body_ind = self.batch_data(x_dict['attr_trans'])
            obs = torch.cat((obs[:, :self.attr_fixed_dim], obs[:, -self.attr_design_dim:]), dim=-1)
            if self.attr_norm is not None:
                x = self.attr_norm(obs)
            else:
                x = obs
            if self.attr_pre_mlp is not None:
                x = self.attr_pre_mlp(x)
            if self.attr_gnn is not None:
                x = self.attr_gnn(x, edges)
            if self.attr_mlp is not None:
                x = self.attr_mlp(x)
            if self.attr_ind_mlp is not None:
                attr_action_mean = self.attr_ind_mlp(x, body_ind)
            else:
                attr_action_mean = self.attr_action_mean(x)
            attr_action_std = self.attr_action_log_std.expand_as(attr_action_mean).exp()
            attr_dist = DiagGaussian(attr_action_mean, attr_action_std)
        else:
            num_nodes_cum_design = None
            attr_dist = None

        # skeleleton transform
        if len(x_dict['skel_trans']) > 0:
            obs, edges, _, num_nodes, num_nodes_cum_skel, body_ind = self.batch_data(x_dict['skel_trans'])
            obs = torch.cat((obs[:, :self.attr_fixed_dim], obs[:, -self.attr_design_dim:]), dim=-1)
            x = self.skel_norm(obs)
            if self.skel_pre_mlp is not None:
                x = self.skel_pre_mlp(x)
            if self.skel_gnn is not None:
                x = self.skel_gnn(x, edges)

            if self.skel_mlp is not None:
                x = self.skel_mlp(x)
            if self.skel_ind_mlp is not None:
                skel_logits = self.skel_ind_mlp(x, body_ind)
            else:
                skel_logits = self.skel_action_logits(x)
            skel_dist = Categorical(logits=skel_logits, uniform_prob=self.skel_uniform_prob)
        else:
            num_nodes_cum_skel = None
            skel_dist = None

        return control_dist, attr_dist, skel_dist, node_design_mask, design_mask, total_num_nodes, num_nodes_cum_control, num_nodes_cum_design, num_nodes_cum_skel, x[0][0].device

    def select_action(self, x, mean_action=False):
        
        control_dist, attr_dist, skel_dist, node_design_mask, _, total_num_nodes, _, _, _, device = self.forward(x)
        if control_dist is not None:
            control_action = control_dist.mean_sample() if mean_action else control_dist.sample()
        else:
            control_action = None

        if attr_dist is not None:
            attr_action = attr_dist.mean_sample() if mean_action else attr_dist.sample()
        else:
            attr_action = None

        if skel_dist is not None:
            skel_action = skel_dist.mean_sample() if mean_action else skel_dist.sample()
        else:
            skel_action = None

        action = torch.zeros(total_num_nodes, self.action_dim).to(device)
        if control_action is not None:
            action[node_design_mask['execution'], :self.control_action_dim] = control_action
        if attr_action is not None:
            action[node_design_mask['attr_trans'], self.control_action_dim:-1] = attr_action
        if skel_action is not None:
            action[node_design_mask['skel_trans'], [-1]] = skel_action.double()
        return action

    def get_log_prob(self, x, action):
        action = torch.cat(action)
        control_dist, attr_dist, skel_dist, node_design_mask, design_mask, total_num_nodes, num_nodes_cum_control, num_nodes_cum_design, num_nodes_cum_skel, device = self.forward(x)
        action_log_prob = torch.zeros(design_mask['execution'].shape[0], 1).to(device)
        # execution log prob
        if control_dist is not None:
            control_action = action[node_design_mask['execution'], :self.control_action_dim]
            control_action_log_prob_nodes = control_dist.log_prob(control_action)
            control_action_log_prob_cum = torch.cumsum(control_action_log_prob_nodes, dim=0)
            control_action_log_prob_cum = control_action_log_prob_cum[torch.LongTensor(num_nodes_cum_control) - 1]
            control_action_log_prob = torch.cat([control_action_log_prob_cum[[0]], control_action_log_prob_cum[1:] - control_action_log_prob_cum[:-1]])
            action_log_prob[design_mask['execution']] = control_action_log_prob
        # attribute transform log prob
        if attr_dist is not None:
            attr_action = action[node_design_mask['attr_trans'], self.control_action_dim:-1]
            attr_action_log_prob_nodes = attr_dist.log_prob(attr_action)
            attr_action_log_prob_cum = torch.cumsum(attr_action_log_prob_nodes, dim=0)
            attr_action_log_prob_cum = attr_action_log_prob_cum[torch.LongTensor(num_nodes_cum_design) - 1]
            attr_action_log_prob = torch.cat([attr_action_log_prob_cum[[0]], attr_action_log_prob_cum[1:] - attr_action_log_prob_cum[:-1]])
            action_log_prob[design_mask['attr_trans']] = attr_action_log_prob
        # skeleton transform log prob
        if skel_dist is not None:
            skel_action = action[node_design_mask['skel_trans'], [-1]]
            skel_action_log_prob_nodes = skel_dist.log_prob(skel_action)
            skel_action_log_prob_cum = torch.cumsum(skel_action_log_prob_nodes, dim=0)
            skel_action_log_prob_cum = skel_action_log_prob_cum[torch.LongTensor(num_nodes_cum_skel) - 1]
            skel_action_log_prob = torch.cat([skel_action_log_prob_cum[[0]], skel_action_log_prob_cum[1:] - skel_action_log_prob_cum[:-1]])
            action_log_prob[design_mask['skel_trans']] = skel_action_log_prob
        return action_log_prob


