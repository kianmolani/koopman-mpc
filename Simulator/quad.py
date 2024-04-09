""" 
Implementation of 3D quadrotor model. Model dynamics burrowed from 
https://rpg.ifi.uzh.ch/docs/RAL21_Torrente.pdf. Assuming '+' configuration.
Implementation of this class is agnostic to whether it's being used as 
part of Koopman-MPC architectures or not, since methods `p_dynamics`, 
`q_dynamics`, `v_dynamics`, and `w_dynamics` represent our attempts at 
simulating actual quadrotor and world dynamics, rather than mere models of
those dynamics.


Created by: Kian Molani
Last updated: Feb. 13, 2024

"""

import numpy as np
from utils.math import quaternion_inverse, skew_symmetric, unit_quat, v_dot_q


class Quad:
    def __init__(self, m, J, l, c, T_max, g, drag, rotor_drag=None, aero_drag=None):
        """
        Initialization of quadrotor class

        m (float)             :: quadrotor mass in [kg]
        J (np.array)          :: quadrotor moment of inertia vector in [kg⋅m⋅m]
        l (float)             :: length between motor and quadrotor CoG in [m]
        c (float)             :: torque generated by each motor in direction of quadrotor z-axis in [N⋅m]
        T_max (float)         :: max thrust generated by each motor in [N]
        g (float)             :: gravitational acceleration of world in [m/s/s]
        drag (bool)           :: set to True if you want to include aerodynamic drag as an unmodeled effect / disturbance
        rotor_drag (np.array) :: rotor drag coefficients in [kg/m]
        aero_drag (float)     :: aerodynamic drag coefficient in [kg/m]

        """

        assert g > 0.0

        self.m = m
        self.J = J
        self.l = l
        self.c = c
        self.T_max = T_max
        self.g = g
        self.rotor_drag = rotor_drag
        self.aero_drag = aero_drag
        self.drag = drag

        # define state variables

        self.p = np.zeros(3,)  # position
        self.q = np.zeros(4,)  # angle quaternion
        self.v = np.zeros(3,)  # linear velocity
        self.w = np.zeros(3,)  # angular velocity

        # define thruster positions

        self.d_x = np.array([self.l, 0, -self.l, 0])
        self.d_y = np.array([0, self.l, 0, -self.l])

        # define actuation thrusts and collective thrust

        self.u = np.array([0.0, 0.0, 0.0, 0.0])  # thrust of each motor in [N]
        self.T_B = np.array([0, 0, np.sum(self.u)])

        # define gravity vector

        self.g = np.array([0, 0, -g])

        # other

        self.c_tau = np.array([-self.c, self.c, -self.c, self.c])

    def get_state(self):
        """
        Returns quadrotor state

        return (list) :: list enumerating all 13 quadrotor states

        """

        return [self.p[0], self.p[1], self.p[2], self.q[0], self.q[1], self.q[2], self.q[3],
            self.v[0], self.v[1], self.v[2], self.w[0], self.w[1], self.w[2]]

    def set_state(self, p, q, v, w):
        """
        Sets quadrotor state

        p (np.array) :: quadrotor position
        q (np.array) :: quadrotor angle quaternion
        v (np.array) :: quadrotor linear velocity
        w (np.array) :: quadrotor angular velocity

        """

        self.p = p
        self.q = q
        self.v = v
        self.w = w

    def update(self, u, h):
        """
        Peforms Runge-Kutta 4th-order dynamics integration

        u (np.array)  :: array whose elements represent the activation of each motor. Values must be between 0.0 and 1.0
        h (float)     :: step-size

        """

        if max(u) > 1.0:
            assert np.any(np.abs(max(u) - 1.0) < 1e-6)

        assert min(u) >= 0.0
        assert len(u) == 4
        assert h > 0.0

        self.u = u * self.T_max
        self.T_B = np.array([0, 0, np.sum(self.u)])

        x = [self.p, self.q, self.v, self.w]

        # evaluate RK4 estimate of state advanced one timestep into the future

        k1 = [self.p_dynamics(x[2]), self.q_dynamics(x[1], x[3]), self.v_dynamics(x[1], x[2]), self.w_dynamics(x[3])]
        x_aux = [x[i] + h / 2 * k1[i] for i in range(4)]
        k2 = [self.p_dynamics(x_aux[2]), self.q_dynamics(x_aux[1], x_aux[3]), self.v_dynamics(x_aux[1], x_aux[2]), self.w_dynamics(x_aux[3])]
        x_aux = [x[i] + h / 2 * k2[i] for i in range(4)]
        k3 = [self.p_dynamics(x_aux[2]), self.q_dynamics(x_aux[1], x_aux[3]), self.v_dynamics(x_aux[1], x_aux[2]), self.w_dynamics(x_aux[3])]
        x_aux = [x[i] + h * k3[i] for i in range(4)]
        k4 = [self.p_dynamics(x_aux[2]), self.q_dynamics(x_aux[1], x_aux[3]), self.v_dynamics(x_aux[1], x_aux[2]), self.w_dynamics(x_aux[3])]

        x = [x[i] + h * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in range(4)]

        # ensure unit quaternion (burrowed from https://github.com/uzh-rpg/data_driven_mpc)

        x[1] = unit_quat(x[1])

        # update state

        self.p, self.q, self.v, self.w = x

    def p_dynamics(self, v):
        """
        Evaluates the time derivative of the position vector

        v (np.array) :: quadrotor linear velocity

        """

        return v

    def q_dynamics(self, q, w):
        """
        Evaluates the time derivative of the attitude vector in quaternion form

        q (np.array) :: quadrotor angle quaternion
        w (np.array) :: quadrotor angular velocity

        """

        return 1 / 2 * skew_symmetric(w).dot(q)

    def v_dynamics(self, q, v):
        """
        Evaluates the time derivative of the linear velocity vector

        q (np.array) :: quadrotor angle quaternion
        v (np.array) :: quadrotor linear velocity

        """

        if self.drag:

            # transform velocity to body frame

            v_b = v_dot_q(v, quaternion_inverse(q))[:, np.newaxis]

            # compute aerodynamic drag acceleration in world frame

            a_drag = -self.aero_drag * v_b**2 * np.sign(v_b) / self.m

            # add rotor drag

            a_drag -= self.rotor_drag * v_b / self.m

            # transform drag acceleration to world frame

            a_drag = np.squeeze(v_dot_q(a_drag, q))

        else:
            
            a_drag = np.zeros((3,))

        return v_dot_q((self.T_B / self.m), q) + self.g + a_drag

    def w_dynamics(self, w):
        """
        Evaluates the time derivative of the angular velocity vector

        w (np.array) :: quadrotor angular velocity

        """

        return np.array([
            1 / self.J[0] * (self.u.dot(self.d_y) + (self.J[1] - self.J[2]) * w[1] * w[2]),
            1 / self.J[1] * (-self.u.dot(self.d_x) + (self.J[2] - self.J[0]) * w[2] * w[0]),
            1 / self.J[2] * (self.u.dot(self.c_tau) + (self.J[0] - self.J[1]) * w[0] * w[1]),
        ])
    