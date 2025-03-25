from lab_ur5.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from lab_ur5.motion_planning.motion_planner import MotionPlanner
from lab_ur5.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur5.manipulation.manipulation_controller import ManipulationController

X = 0
Y = 1
Z = 2


def stack_stage1(block_positions, target):
    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)
    r2_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller.release_grasp()
    r2_controller.move_home()
    for i, block in enumerate(block_positions):
        r2_controller.pick_up(block[X], block[Y], block[Z] + 0.10)
        r2_controller.put_down(target[X], target[Y], target[Z] + i * 0.12 + 0.03)

    r2_controller.move_home()


def stack_stage2(cube_position):
    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)
    r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)

    # reset environment
    r1_controller.move_home(speed=1)
    #r2_controller.move_home(speed=1)
    r1_controller.release_grasp()
    r2_controller.release_grasp()

    r2_controller.pick_up(cube_position[X], cube_position[Y], cube_position[Z])
    r2_controller.plan_and_moveJ(
        [4.802989482879639, -1.6509276829161585, -1.7187272310256958, -1.4068054866841813, -0.0038974920855920914,
         -1.4862588087665003], speed=1, acceleration=1)

    # move e_1 towards e_2 end effector
    r1_controller.plan_and_moveJ(
        [-1.421, -1.952660699883932, -1.5125269889831543, -0.47875674188647466, 0.03028770722448826,
         -0.7987430731402796], speed=1, acceleration=1)
    r1_controller.moveL_relative([-0.19, -0.01, -0.01], speed=0.05)

    r1_controller.grasp()
    r2_controller.release_grasp()

    r1_controller.moveL_relative([0.035, 0, 0], speed=0.05)

    target = [0.3, -0.3, 0.03]
    r1_controller.put_down(target[0], target[1], target[2])

# block_position = [
#     [-0.7, -0.6, 0.03],
#     [-0.7, -0.7, 0.03],
#     [-0.7, -0.8, 0.03],
#     [-0.7, -0.9, 0.03]]

block_position = [
    [0.4, 0, 0.03],
    [0.5, 0, 0.03],
    [0.6, 0, 0.03],
    [0.7, 0, 0.03]]

stack_stage2([-0.7, -0.6, 0.03])
