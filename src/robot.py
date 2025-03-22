import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RobotArm:
    def __init__(self, num_joints, joint_lengths, joint_axes, joint_limits):
        self.num_joints = num_joints
        self.joint_lengths = np.array(joint_lengths)
        self.joint_axes = joint_axes
        self.joint_limits = joint_limits

    def forward_kinematics(self, joint_angles):
        position = np.zeros(3)
        orientation = R.identity()
        positions = [position.copy()]

        for i in range(self.num_joints):
            axis = self.joint_axes[i]
            angle = joint_angles[i]

            # Calculate the rotation matrix for the current joint
            rotation = R.from_rotvec(angle * np.array(axis))
            orientation = orientation * rotation

            # Move the end effector along the current joint's length
            position += orientation.apply([self.joint_lengths[i], 0, 0])
            positions.append(position.copy())

        return positions, orientation.as_euler('xyz')

    def inverse_kinematics(self, target_position, target_orientation):
        joint_angles = np.zeros(self.num_joints)
        position = np.zeros(3)
        orientation = R.identity()

        for i in range(self.num_joints):
            # Solve for the angle of joint i
            axis = self.joint_axes[i]

            if i == self.num_joints - 1:
                # At the last joint, match the orientation
                relative_orientation = R.from_euler('xyz', target_orientation) * orientation.inv()
                angle = np.arccos(np.clip(np.dot(relative_orientation.as_rotvec(), axis) / np.linalg.norm(axis), -1, 1))
            else:
                # Solve for the position
                relative_position = target_position - position
                angle = np.arccos(np.clip(np.dot(relative_position, axis) / np.linalg.norm(relative_position), -1, 1))

            joint_angles[i] = np.clip(angle, self.joint_limits[i][0], self.joint_limits[i][1])

            # Apply the rotation and update the current position and orientation
            rotation = R.from_rotvec(joint_angles[i] * np.array(axis))
            orientation = orientation * rotation
            position += orientation.apply([self.joint_lengths[i], 0, 0])

        return joint_angles

    def plot_robot(self, joint_angles):
        positions, _ = self.forward_kinematics(joint_angles)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        xs = [pos[0] for pos in positions]
        ys = [pos[1] for pos in positions]
        zs = [pos[2] for pos in positions]

        # Plot the start joint in RED
        ax.plot([xs[0]], [ys[0]], [zs[0]], 'ro', markersize=10, label='Start Joint')

        # Plot the rest of the joints in BLUE
        for i in range(1, len(xs) - 1):
            ax.plot([xs[i]], [ys[i]], [zs[i]], 'bo', markersize=8)

        # Plot the end joint in GREEN
        ax.plot([xs[-1]], [ys[-1]], [zs[-1]], 'go', markersize=10, label='End Effector')

        # Plot the lines connecting the joints
        ax.plot(xs, ys, zs, '-', linewidth=2, color='b')

        ax.set_xlim([-np.sum(self.joint_lengths), np.sum(self.joint_lengths)])
        ax.set_ylim([-np.sum(self.joint_lengths), np.sum(self.joint_lengths)])
        ax.set_zlim([-np.sum(self.joint_lengths), np.sum(self.joint_lengths)])

        plt.legend()
        plt.show()

# Test Code
if __name__ == "__main__":
    num_joints = 3
    joint_lengths = [1, 1, 1]
    joint_axes = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    joint_limits = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]

    robot = RobotArm(num_joints, joint_lengths, joint_axes, joint_limits)

    joint_angles = [0.5, 0.3, 0.2]
    position, orientation = robot.forward_kinematics(joint_angles)
    print("Forward Kinematics:")
    print("Position:", position[-1])
    print("Orientation:", orientation)

    target_position = position[-1]
    target_orientation = orientation
    calculated_angles = robot.inverse_kinematics(target_position, target_orientation)
    print("\nInverse Kinematics:")
    print("Joint Angles:", calculated_angles)

    print("\nPlotting the robot:")
    robot.plot_robot(joint_angles)
