from .hopper import HopperEnv
from .swimmer import SwimmerEnv
from .ant import AntEnv
from .gap import GapEnv


env_dict = {
    'hopper': HopperEnv,
    'swimmer': SwimmerEnv,
    'ant': AntEnv,
    'gap': GapEnv
}