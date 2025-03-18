from sim_ur5.mujoco_env.sim_env import SimEnv
from sim_ur5.motion_planning.motion_executor import MotionExecutor
from sim_ur5.mujoco_env.common.ur5e_fk import forward
from numpy import pi

X = 0
Y = 1
Z = 2


def stack_stage1(block_positions, target):
    env = SimEnv()
    executor = MotionExecutor(env)

    env.reset(randomize=False, block_positions=block_positions)
    for i, block in enumerate(block_positions):
        executor.pick_up("ur5e_2", block[X], block[Y], block[Z] + 0.12)

        executor.put_down("ur5e_2", target[X], target[Y], target[Z] + i * 0.12 + 0.03)


def stack_stage2(cube_position):
    env = SimEnv()
    executor = MotionExecutor(env)
    env.reset(randomize=False, block_positions=[cube_position])

    executor.pick_up("ur5e_2", cube_position[X], cube_position[Y], cube_position[Z] + 0.12)
    left_robot_pos = env.get_ee_pos()

    left_robot_pos[Z] += 0.4  # go up
    executor.plan_and_move_to_xyz_facing_down("ur5e_2", left_robot_pos)

    # set e_1 position to be under e_2 end effector
    left_robot_pos = env.get_ee_pos()
    left_robot_pos[Z] -= 0.3  # go under
    executor.plan_and_move_to_xyz_facing_down("ur5e_1", left_robot_pos)

    # flip e_1 end effector to face to e_2 end effector
    flip_pos_1 = env.get_agent_joint("ur5e_1")
    flip_pos_1[3] -= 0.1
    executor.moveJ("ur5e_1", flip_pos_1)
    flip_pos_1[3] += 0.1
    flip_pos_1[4] += pi
    executor.moveJ("ur5e_1", flip_pos_1)

    #
    # # transfare the cube from e_2 end effector to e_1's
    target_pos_2 = env.get_ee_pos()
    target_pos_2[Z] -= 0.03
    executor.plan_and_move_to_xyz_facing_down("ur5e_2", target_pos_2)
    #
    executor.deactivate_grasp()


# TODO: check the limits
"""
workspace_x_lims = [-1.0, -0.45]
workspace_y_lims = [-1.0, -0.45]
"""

# Locate an area where both robots can reach
# You can choose any location in the area to place the blocks

block_position = [
    [-0.7, -0.6, 0.2],
    [-0.7, -0.7, 0.03],
    [-0.7, -0.8, 0.03],
    [-0.7, -0.9, 0.03]]

stack_stage1(block_position, [-0.26, 0, 0.01])

# Create the simulation environment and the executor
# env = SimEnv()
# executor = MotionExecutor(env)
#
# # Add blocks to the world by enabling the randomize argument and setting the block position in the reset function of the SimEnv class
# env.reset(randomize=False, block_positions=block_position)
#
# # moveJ is utilized when the robot's joints are clear to you but use carefully because there is no planning here
# move_to = [1.305356658502026, -0.7908733209856437, 1.4010098471710881, 4.102251451313659, -1.5707962412281837,
#            -0.26543967541515895]
# executor.moveJ("ur5e_2", move_to)
#
# executor.pick_up("ur5e_2", -0.7, -0.6, 0.15)
#
# executor.plan_and_move_to_xyz_facing_down("ur5e_2", [-0.7, -0.7, 0.15])
# executor.put_down("ur5e_2", -0.7, -0.7, 0.20)
#
# executor.wait(4)
