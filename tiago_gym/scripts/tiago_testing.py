import numpy as np
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Quaternion, Pose, PoseStamped
from control_msgs.msg import PointHeadActionGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def head_control_demo():
        def create_command(joint):
                message = JointTrajectory()
                message.header = Header()
                message.joint_names = ['head_1_joint', 'head_2_joint']
                point = JointTrajectoryPoint(positions=joint, time_from_start = rospy.Duration(0.2))
                message.points.append(point)
                
                return message

        joints = [[0.4, 0.0], [-0.4, 0.0], [0.0, 0.3], [0.0, -0.3]]

        rospy.init_node('tiago_oculus_control')

        # h = Header(stamp=rospy.Time.now(), frame_id='/base_footprint')
        # m = PointHeadActionGoal()
        # m.header.frame_id = '/base_footprint'
        # m.goal.target.header.frame_id = '/base_footprint' 
        # m.goal.target.point = Point(1.0, 0.0, 0.5)

        # publisher = rospy.Publisher('/head_controller/point_head_action/goal', PointHeadActionGoal, queue_size=5)
        publisher = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=5)
        r = rospy.Rate(10)
        while not rospy.is_shutdown():

                l = input()
                m = create_command(joints[int(l)])
                print(m)
                publisher.publish(m)
                r.sleep()

def gripper_control_demo():

        def create_gripper_command(dist):
                message = JointTrajectory()
                message.header = Header()
                message.joint_names = ['gripper_right_finger_joint']
                point = JointTrajectoryPoint(positions=[dist], time_from_start = rospy.Duration(0.5))
                message.points.append(point)
                return message
        
        joints = [0.05, 0.65, 0.35]
        rospy.init_node('tiago_oculus_control')
        publisher = rospy.Publisher('/gripper_right_controller/command', JointTrajectory, queue_size=5)
        r = rospy.Rate(10)

        while not rospy.is_shutdown():
                l = input()
                m = create_gripper_command(joints[int(l)])
                print(m)
                publisher.publish(m)
                r.sleep()

if __name__=='__main__':
        rospy.init_node('tiago_testing')
        # from real_tiago.tiago.tiago_gym import TiagoGym

        # env = TiagoGym(
        #         frequency=10,
        #         head_policy=None,
        #         base_enabled=False,
        #         right_arm_enabled=True,
        #         left_arm_enabled=False,
        #         right_gripper_type='robotiq2F-140',
        #         left_gripper_type='robotiq2F-85',)
        # env.reset()

        # while not rospy.is_shutdown():
        #         j = float(input())
                
        #         env.step({'right': {'shoulder_1': j}})


        from real_tiago.utils.ros_utils import Listener
        from sensor_msgs.msg import LaserScan
                
        def process_scan(message):
            min_val = message.range_min
            max_val = message.range_max

            ranges = np.array(message.ranges)
        #     print(np.sum(np.not_equal(ranges, np.inf)))
            ranges = np.clip(message.ranges, min_val, max_val)
        #     print(np.sum(np.not_equal(ranges, 25)))
            return ranges

        scan = Listener(
                                input_topic_name='/scan',
                                input_message_type=LaserScan,
                                post_process_func=process_scan       
                                )
        
        while not rospy.is_shutdown():
                print(scan.get_most_recent_msg().shape)