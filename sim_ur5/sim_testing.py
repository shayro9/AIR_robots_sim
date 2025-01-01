from sim_ur5.mujoco_env.sim_env import SimEnv
from sim_ur5.motion_planning.motion_executor import MotionExecutor
from sim_ur5.mujoco_env.common.ur5e_fk import forward

env = SimEnv()

"""
workspace_x_lims = [-0.9, -0.54]
workspace_y_lims = [-1.0, -0.55]
"""
block_position = [
    [-0.7, -0.6, 0.03],
    [-0.7, -0.7, 0.03],
    [-0.7, -0.8, 0.03],
    [-0.7, -0.9, 0.03]]

executor = MotionExecutor(env)

# Add blocks to the world
env.reset(randomize=False, block_positions=block_position)

"""
executor.plan_and_move_to_xyz_facing_down("ur5e_2", [-0.7, -0.6, 0.15])
"""

move_to = [1.305356658502026, -0.7908733209856437, 1.4010098471710881, 4.102251451313659, -1.5707962412281837, -0.26543967541515895]
executor.moveJ("ur5e_2", move_to)

executor.pick_up("ur5e_2", -0.7, -0.6, 0.15)

executor.plan_and_move_to_xyz_facing_down("ur5e_2", [-0.7, -0.7, 0.15])
executor.put_down("ur5e_2", -0.7, -0.7, 0.20)

executor.plan_and_move_to_xyz_facing_down("ur5e_2", [-0.7, -0.8, 0.15])
executor.pick_up("ur5e_2", -0.7, -0.8, 0.15)

executor.plan_and_move_to_xyz_facing_down("ur5e_2", [-0.7, -0.7, 0.15])
executor.put_down("ur5e_2", -0.7, -0.7, 0.20)
executor.plan_and_move_to_xyz_facing_down("ur5e_2", [-0.7, -0.7, 0.2])


executor.plan_and_move_to_xyz_facing_down("ur5e_2", [-0.7, -0.9, 0.15])
executor.pick_up("ur5e_2", -0.7, -0.9, 0.15)

executor.plan_and_move_to_xyz_facing_down("ur5e_2", [-0.7, -0.7, 0.20])
executor.put_down("ur5e_2", -0.7, -0.7, 0.20)

executor.wait(4)













