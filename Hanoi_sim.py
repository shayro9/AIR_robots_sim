from sim_ur5.mujoco_env.sim_env import SimEnv
from sim_ur5.motion_planning.motion_executor import MotionExecutor
from sim_ur5.mujoco_env.common.ur5e_fk import forward
from numpy import pi
from copy import deepcopy
from sim_ur5.utils.Hanoi_solve import generate_hanoi_moves, kiss

X = 0
Y = 1
Z = 2

block_position = [
    [-0.995, -0.615, 0.15],
    [-0.995, -0.615, 0.11],
    [-0.995, -0.615, 0.07],
    [-0.995, -0.615, 0.03]]

regions = {
    'red': (-0.995, -0.615, 0.005),
    'green': (-0.805, -0.615, 0.005),
    'blue': (-0.615, -0.615, 0.005)
}

env = SimEnv()
executor = MotionExecutor(env)

env.reset(randomize=False, block_positions=block_position)

    # for i, block in enumerate(block_positions):
    #     executor.pick_up("ur5e_2", block[X], block[Y], block[Z] + 0.12)
    #
    #     executor.put_down("ur5e_2", target[X], target[Y], target[Z] + i * 0.12 + 0.03)

# Initial tower position on table2 (from your XML)
start_pos = (-0.995, -0.615, 0.005)  # (x, y, z_base)

# Height offsets based on your disc spacing (0.08 per level)
disc_heights = [0.03 + i * 0.04 for i in range(4)]

# Generate moves for 5 discs
moves = generate_hanoi_moves(4, regions, start_pos, disc_heights)
print(moves)
disks = {}
for i, (d, _, _) in enumerate(moves):
    disks[d] = i

executor.plan_and_move_to_xyz_facing_down("ur5e_1", [-0.115, -0.615, 0.05])
for i, (d, start_pos, target_pos) in enumerate(moves):
    executor.zero_all_robots_vels_except("ur5e_2")
    x, y, z = start_pos
    executor.pick_up("ur5e_2", x, y, z + 0.12)

    left_robot_pos = env.get_ee_pos()
    left_robot_pos[Z] += 0.1  # go up
    executor.plan_and_move_to_xyz_facing_down("ur5e_2", left_robot_pos)



    if disks[d] == i:
        kiss(env, executor, [-0.315, -0.615, 0.4])

    x, y, z = target_pos
    executor.put_down("ur5e_2", x, y, z + 0.12)

    left_robot_pos = env.get_ee_pos()
    left_robot_pos[Z] += 0.1  # go up
    executor.plan_and_move_to_xyz_facing_down("ur5e_2", left_robot_pos)
