import argparse
import os
import sys
sys.path.append(os.getcwd())

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.transform2act_agent import Transform2ActAgent


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument('--epoch', default='best')
parser.add_argument('--save_video', action='store_true', default=False)
parser.add_argument('--pause_design', action='store_true', default=False)
args = parser.parse_args()
cfg = Config(args.cfg, tmp=False, create_dirs=False)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cpu')
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

epoch = int(args.epoch) if args.epoch.isnumeric() else args.epoch

"""create agent"""
agent = Transform2ActAgent(cfg=cfg, dtype=dtype, device=device, seed=cfg.seed, num_threads=1, training=False, checkpoint=epoch)

agent.visualize_agent(num_episode=4, save_video=args.save_video, pause_design=args.pause_design)



