import numpy as np

# from casadi import *
from scipy.integrate import solve_ivp
# from math import skew_symmetric, v_dot_q

# class Quadrotor3D:
#     def __init__(self, p, q, v, w, u, mass, length, max_thrust, c, J):
#         self.p = p
#         self.q = q
#         self.v = v
#         self.w = w
#         self.u = u
#         self.mass = mass
#         self.length = length
#         self.max_thrust = max_thrust
#         self.c = c
#         self.J = J

#     def p_dynamics(self):
#         return self.v
    
#     def q_dynamics(self):
#         return 1/2 * mtimes(skew_symmetric(self.w), self.q)
    
#     def v_dynamics(self, g: float):
#         f_thrust = self.u * self.max_thrust
#         g = vertcat(0.0, 0.0, g)
#         a_thrust = vertcat(0.0, 0.0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3]) / self.mass
#         return v_dot_q(a_thrust, self.q) - g
    
#     def w_dynamics(self):
#         f_thrust = self.u * self.max_thrust
#         x_f = MX(np.array([self.length, 0, -self.length, 0]))
#         y_f = MX(np.array([0, self.length, 0, -self.length]))
#         z_l_tau = MX(np.array([-self.c, self.c, -self.c, self.c]))
#         c_f = MX(z_l_tau)
#         return vertcat(
#             (mtimes(f_thrust.T, y_f) + (self.J[1] - self.J[2]) * self.w[1] * self.w[2]) / self.J[0],
#             (-mtimes(f_thrust.T, x_f) + (self.J[2] - self.J[0]) * self.w[2] * self.w[0]) / self.J[1],
#             (mtimes(f_thrust.T, c_f) + (self.J[0] - self.J[1]) * self.w[0] * self.w[1]) / self.J[2])


class SlowManifold:
    def __init__(self, mu, lam):
        self.mu = mu
        self.lam = lam

    def model(self, t, variables):
        x, y = variables
        dxdt = self.mu * x
        dydt = self.lam * (y - x**2)
        return [dxdt, dydt]

    def solve(self, x0, y0, t_start, t_end, num_points):
        initial_conditions = [x0, y0]
        t_span = np.linspace(t_start, t_end, num_points)
        solution = solve_ivp(self.model, [t_start, t_end], initial_conditions, t_eval=t_span)
        return solution.t, solution.y[0], solution.y[1]