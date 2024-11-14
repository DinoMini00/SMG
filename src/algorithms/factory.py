from algorithms.sac import SAC
from algorithms.rad import RAD
from algorithms.curl import CURL
from algorithms.pad import PAD
from algorithms.soda import SODA
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.sgsac import SGSAC
from algorithms.smg import SMG
from algorithms.srm import SRM

algorithm = {
    "sac": SAC,
    "rad": RAD,
    "curl": CURL,
    "pad": PAD,
    "soda": SODA,
    "drq": DrQ,
    "svea": SVEA,
    "sgsac": SGSAC,
    "srm":SRM,
    "smg": SMG
}


def make_agent(obs_shape, action_shape, args, writer):
    return algorithm[args.algorithm](obs_shape, action_shape, args, writer)
