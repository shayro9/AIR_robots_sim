from sim_ur5.mujoco_env.sim_env import SimEnv
from sim_ur5.motion_planning.motion_executor import MotionExecutor
from sim_ur5.utils.Hanoi_solve import generate_hanoi_moves, kiss
from Hanoi_A_star import a_star
import random

X = 0
Y = 1
Z = 2


class Pole:
    def __init__(self, count, position):
        self.count = count
        self.position = position

    def top_height(self):
        return 0.03 + (self.count - 1) * 0.04

    def add_block(self):
        self.count += 1

    def remove_block(self):
        self.count -= 1

    def get_top_position(self):
        top_position = self.position
        top_position[Z] = self.top_height()
        return top_position


def is_valid_hanoi_state(state):
    if not (isinstance(state, list) and len(state) == 3 and all(isinstance(peg, list) for peg in state)):
        return False

    all_rings = set()
    for peg in state:
        if peg != sorted(peg, reverse=True):
            return False
        all_rings.update(peg)

    return all_rings.issubset({1, 2, 3, 4}) and len(all_rings) == 4


class Hanoi_robot:
    block_position = [
        [-0.995, -0.615, 0.15],
        [-0.995, -0.615, 0.11],
        [-0.995, -0.615, 0.07],
        [-0.995, -0.615, 0.03]]

    regions = [
        [-0.995, -0.615, 0.005],
        [-0.805, -0.615, 0.005],
        [-0.615, -0.615, 0.005]
    ]

    def __init__(self, is_random=False, init_state=None, final_pole=2):
        self.final_pole = final_pole
        if is_random:
            self.init_state = [[], [], []]
            for i in range(4, 0, -1):
                pole_number = random.randint(0, 2)
                position = [x for x in self.regions[pole_number]]
                position[Z] += 0.03 + len(init_state[pole_number]) * 0.04
                self.block_position[i - 1] = position
                self.init_state[pole_number].append(i)
        else:
            if init_state is None:
                init_state = [[4, 3, 2, 1], [], []]
            elif not is_valid_hanoi_state(init_state):
                raise ValueError("BAD HANOI STATE")
            self.init_state = init_state
            for i, p in enumerate(init_state):
                position = [x for x in self.regions[i]]
                for j, b in enumerate(p):
                    position[Z] += 0.03 + j * 0.04
                    self.block_position[b - 1] = position

        self.poles = [Pole(len(init_state[0]), self.regions[0]),
                      Pole(len(init_state[1]), self.regions[1]),
                      Pole(len(init_state[2]), self.regions[2])]

        self.env = SimEnv()
        self.executor = MotionExecutor(self.env)

        self.env.reset(randomize=False, block_positions=self.block_position)

    def pick_up(self, start_pole):
        x, y, z = self.poles[start_pole].get_top_position()
        self.executor.pick_up("ur5e_2", x, y, z + 0.12)
        self.poles[start_pole].remove_block()

    def put_down(self, dest_pole):
        x, y, z = self.poles[dest_pole].get_top_position()
        self.executor.put_down("ur5e_2", x, y, z + 0.12)
        self.poles[dest_pole].add_block()

    def move_up(self, dist=0.3):
        left_robot_pos = self.env.get_ee_pos()
        left_robot_pos[Z] = dist  # go up
        self.executor.plan_and_move_to_xyz_facing_down("ur5e_2", left_robot_pos)
        left_robot_pos[X] = self.regions[1][X]
        left_robot_pos[Y] = self.regions[1][Y]
        self.executor.plan_and_move_to_xyz_facing_down("ur5e_2", left_robot_pos)

    def solve(self):
        actions = a_star(self.init_state, self.final_pole)
        if actions is None:
            raise ValueError("No solution found")
        for start_pole, dest_pole in actions:
            self.executor.zero_all_robots_vels_except("ur5e_2")
            self.pick_up(start_pole)
            self.move_up()

            self.put_down(dest_pole)
            self.move_up()


robot = Hanoi_robot(True, [[1], [3, 2], [4]], 0)
robot.solve()
