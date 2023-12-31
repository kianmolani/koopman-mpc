"""
Set of matrix-math related utility functions. Functions copied from 
https://github.com/uzh-rpg/data_driven_mpc/tree/main


Created by: Kian Molani
Last updated: Sept. 25, 2023

"""

import numpy as np
import pyquaternion

from casadi import *

def q_dot_q(q, r):
    """
    Applies the rotation of quaternion r to quaternion q. In order words, rotates quaternion q by r. Quaternion format: wxyz

    q      :: 4-length numpy array or CasADi MX. Initial rotation
    r      :: 4-length numpy array or CasADi MX. Applied rotation
    return :: the quaternion q rotated by r, with the same format as in the input

    """

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rw, rx, ry, rz = r[0], r[1], r[2], r[3]

    t0 = rw * qw - rx * qx - ry * qy - rz * qz
    t1 = rw * qx + rx * qw - ry * qz + rz * qy
    t2 = rw * qy + rx * qz + ry * qw - rz * qx
    t3 = rw * qz - rx * qy + ry * qx + rz * qw

    if isinstance(q, np.ndarray):
        return np.array([t0, t1, t2, t3])
    else:
        return vertcat(t0, t1, t2, t3)
    

def rotation_matrix_to_quat(rot):
    """
    Calculate a quaternion from a 3x3 rotation matrix

    rot    :: 3x3 numpy array, representing a valid rotation matrix
    return :: a quaternion corresponding to the 3D rotation described by the input matrix. Quaternion format: wxyz
    
    """

    q = pyquaternion.Quaternion(matrix=rot)

    return np.array([q.w, q.x, q.y, q.z])


def separate_variables(traj):
    """
    Reshapes a trajectory into expected format

    traj   :: Nx13 array representing the reference trajectory
    return :: a list with the components: Nx3 position trajectory array, Nx4 quaternion trajectory array, Nx3 velocity trajectory array, Nx3 body rate trajectory array
    
    """

    p_traj = traj[:, :3]
    a_traj = traj[:, 3:7]
    v_traj = traj[:, 7:10]
    r_traj = traj[:, 10:]
    
    return [p_traj, a_traj, v_traj, r_traj]


def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3D vector (PAMPC version)

    v      :: 3D numpy vector or CasADi MX
    return :: the corresponding skew-symmetric matrix of v with the same data type as v
    
    """

    if isinstance(v, np.ndarray):
        return np.array([[0, -v[0], -v[1], -v[2]],
                         [v[0], 0, v[2], -v[1]],
                         [v[1], -v[2], 0, v[0]],
                         [v[2], v[1], -v[0], 0]])

    return vertcat(
        horzcat(0, -v[0], -v[1], -v[2]),
        horzcat(v[0], 0, v[2], -v[1]),
        horzcat(v[1], -v[2], 0, v[0]),
        horzcat(v[2], v[1], -v[0], 0))


def undo_quaternion_flip(q_past, q_current):
    """
    Detects if q_current generated a quaternion jump and corrects it. Requires knowledge of the previous quaternion in the series (q_past)
    
    q_past    :: 4-dimensional vector representing a quaternion in wxyz form
    q_current :: 4-dimensional vector representing a quaternion in wxyz form. Will be corrected if it generates a flip wrt q_past
    return    :: q_current with the flip removed if necessary

    """

    if np.sqrt(np.sum((q_past - q_current) ** 2)) > np.sqrt(np.sum((q_past + q_current) ** 2)):
        return -q_current
    
    return q_current


def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus

    q      :: 4-dimensional numpy array or CasADi object
    return :: the unit quaternion in the same data format as the original one

    """

    if isinstance(q, np.ndarray):
        q_norm = np.sqrt(np.sum(q ** 2))
    else:
        q_norm = sqrt(sumsqr(q))

    return 1 / q_norm * q


def quaternion_inverse(q):
    """
    Unknown function

    q      :: unknown
    return :: unknown

    """
        
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return vertcat(w, -x, -y, -z)


def q_to_rot_mat(q):
    """
    Unknown function

    q      :: unknown
    return :: unknown

    """

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])
    else:
        rot_mat = vertcat(
            horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
            horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

    return rot_mat


def v_dot_q(v, q):
    """
    Unknown function

    q      :: unknown
    return :: unknown

    """

    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return mtimes(rot_mat, v)


def quaternion_to_euler(q):
    """
    Unknown function

    q      :: unknown
    return :: unknown

    """
        
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    
    return [roll, pitch, yaw]
