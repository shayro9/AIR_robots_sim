from numpy import pi
def generate_hanoi_moves(n, regions, start_pos, disc_heights):
    """
    Generates Hanoi tower moves with position calculations for MuJoCo

    Args:
        n (int): Number of discs
        regions (dict): Dictionary with peg positions {'red': (x,y,z), ...}
        start_pos (tuple): Initial tower position (x,y,z_base)
        disc_heights (list): Height offsets for each disc level

    Returns:
        list: Sequence of moves with positions [(disc, from_pos, to_pos), ...]
    """
    moves = []
    stacks = {
        'red': [],
        'green': [],
        'blue': []
    }

    # Initialize starting position stack
    for i in range(n):
        z = start_pos[2] + disc_heights[i]
        stacks['red'].append(('disc%d' % i, (start_pos[0], start_pos[1], z)))

    def hanoi(n, source, target, auxiliary):
        if n == 0:
            return

        hanoi(n - 1, source, auxiliary, target)

        # Move the nth disc
        disc, source_pos = stacks[source].pop()

        # Calculate target height
        height_idx = len(stacks[target])
        target_z = regions[target][2] + disc_heights[height_idx]
        target_pos = (regions[target][0], regions[target][1], target_z)

        # Record move
        moves.append((disc, source_pos, target_pos))
        stacks[target].append((disc, target_pos))

        hanoi(n - 1, auxiliary, target, source)

    hanoi(n, 'red', 'blue', 'green')
    return moves

def kiss(env, executor, kiss_pos):
    executor.plan_and_move_to_xyz_facing_down("ur5e_2", kiss_pos)

    kiss_pos[2] -= 0.25  # go under
    executor.plan_and_move_to_xyz_facing_down("ur5e_1", kiss_pos)

    # flip e_1 end effector to face to e_2 end effector
    flip_pos_1 = env.get_agent_joint("ur5e_1")
    flip_pos_1[3] -= 0.1
    executor.moveJ("ur5e_1", flip_pos_1)
    flip_pos_1[3] += 0.1
    flip_pos_1[4] -= pi
    executor.moveJ("ur5e_1", flip_pos_1)


    target_pos_2 = env.get_ee_pos()
    target_pos_2[2] -= 0.03
    executor.plan_and_move_to_xyz_facing_down("ur5e_2", target_pos_2)

    target_pos_2 = env.get_ee_pos()
    target_pos_2[2] += 0.03
    executor.plan_and_move_to_xyz_facing_down("ur5e_2", target_pos_2)

    executor.plan_and_move_to_xyz_facing_down("ur5e_1", [-0.115, -0.615, 0.05])
