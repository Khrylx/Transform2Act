from khrylib.utils.math import *


def get_single_body_qposaddr(model, body):
    i = model.body_names.index(body)
    start_joint = model.body_jntadr[i]
    # assert start_joint >= 0
    end_joint = start_joint + model.body_jntnum[i]
    start_qposaddr = model.jnt_qposadr[start_joint]
    if end_joint < len(model.jnt_qposadr):
        end_qposaddr = model.jnt_qposadr[end_joint]
    else:
        end_qposaddr = model.nq
    return start_qposaddr, end_qposaddr
