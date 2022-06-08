import argparse
import os
import sys
sys.path.append(os.getcwd())

from khrylib.utils import *
from design_opt.utils.config import Config
from design_opt.agents.transform2act_agent import Transform2ActAgent


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default=None)
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('--tmp', action='store_true', default=False)
parser.add_argument('--num_threads', type=int, default=20)
parser.add_argument('--gpu_index', type=int, default=0)
parser.add_argument('--epoch', default='0')
parser.add_argument('--show_noise', action='store_true', default=False)
args = parser.parse_args()
if args.render:
    args.num_threads = 1
cfg = Config(args.cfg, args.tmp)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)

start_epoch = int(args.epoch) if args.epoch.isnumeric() else args.epoch

"""create agent"""
agent = Transform2ActAgent(cfg=cfg, dtype=dtype, device=device, seed=cfg.seed, num_threads=args.num_threads, training=True, checkpoint=start_epoch)


def main_loop():

    if args.render:
        agent.pre_epoch_update(start_epoch)
        agent.sample(1e8, mean_action=not args.show_noise, render=True)
    else:
        for epoch in range(start_epoch, cfg.max_epoch_num):          
            agent.optimize(epoch)
            agent.save_checkpoint(epoch)

            """clean up gpu memory"""
            torch.cuda.empty_cache()

        agent.logger.info('training done!')


main_loop()
