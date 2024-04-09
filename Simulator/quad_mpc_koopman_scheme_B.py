""" 
Implementation of Quadrotor MPC class for the following model
representations: x_k+1 = f(x_k, u_k) + g(x_k, u_k, w_k), where
f is the quadrotor dynamics model and g is the noise model. Here,
g is assumed to be a Koopman derived model that takes the following form:
the following scheme

YThe same as nominal but

Later abstractify to add to whatever noise elements






Koopman 
representations: 



. This class 
contains properties and functions inherent and required by MPC control schemes.


Created by: Kian Molani
Last updated: Mar. 18, 2024

"""

""" 
Implementation of neural network modules required to learn, from data, our 
lifting dictionary and linear ("Koopman") operators


Created by: Kian Molani
Last updated: Feb. 22, 2024

"""

import sys

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import *
from quad import Quad
from networks import Encoder


class QuadMPCKoopmanEDMD:
    def __init__(self, quad: Quad, A: np.array, B: np.array, C: np.array, lifted_state_dim: int,
                 state_space_dim: int, control_dim: int, t_horizon: float, n_nodes: int, Q, R,
                 model_name: str, Ψ_inv: np.array):
        """
        Initialization of quadrotor MPC class for Koopman representations

        quad (Quad)             :: 3D quadrotor model
        lifted_state_dim (int)  :: dimension of state vector in lifted space
        no_control_inputs (int) :: dimension of control vector
        A (np.array)            :: linear ("Koopman") operator acting on lifted state
        B (np.array)            :: linear ("Koopman") operator acting on control states
        t_horizon (float)       :: prediction horizon
        n_nodes (int)           :: number of control nodes within prediction horizon
        Q (np.array)            :: weighing matrix for quadratic cost function of dimension `lifted_state_dim`
        R (np.array)            :: weighing matrix for quadratic cost function of dimension `no_control_inputs`
        
        """

        self.quad = quad
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.lifted_state_dim = lifted_state_dim
        self.state_space_dim = state_space_dim
        self.control_dim = control_dim
        self.model_name = model_name
        self.t_horizon = t_horizon
        self.n_nodes = n_nodes
        self.N = n_nodes
        self.Q_diagonal = Q
        self.Ψ_inv = Ψ_inv

        # declare model states as CasADi symbolic variables

        self.z = MX.sym('z', lifted_state_dim)

        # declare model states as CasADi symbolic variables

        self.x = MX.sym('x', state_space_dim)

        # declare control inputs as CasADi symbolic variables

        self.u = MX.sym('u', control_dim)

        # declare time derivative of model states governing drone dynamics

        self.z_dot = self.z_dynamics()

        # create CasADi Function object for DAEs

        self.z_dot = Function('z_dot', [self.z, self.x, self.u], [self.z_dot], ['z', 'u'], ['z_dot'])

        # formulate acados dynamics model from CasADi model

        self.acados_model = self.setup_acados_model(self.z_dot(z=self.z, x=self.x, u=self.u)['z_dot'])

        # setup and compile acados OCP solver

        self.acados_ocp_solver = self.setup_acados_ocp(self.acados_model)

    def setup_acados_model(self, casadi_model):
        """
        Builds acados symbolic model using CasADi expressions. Needed when creating 
        acados OCP solver

        casadi_model (CasADi Function) :: CasADi Function object representing symbolic model of quadrotor dynamics

        """

        acados_model = AcadosModel()
        acados_model.name = self.model_name
        acados_model.f_expl_expr = casadi_model
        acados_model.f_impl_expr = MX.sym('z_dot', casadi_model.shape) - casadi_model
        acados_model.x = self.z
        acados_model.xdot = MX.sym('z_dot', casadi_model.shape)
        acados_model.u = self.u
        acados_model.p = []

        return acados_model

    def setup_acados_ocp(self, acados_model):
        """
        Builds acados OCP object and OCP solver using acados model. Please refer to 
        https://docs.acados.org/python_interface/ docs

        acados_model (AcadosModel) :: acados model of system dynamics

        """

        nx = acados_model.x.size()[0]
        nu = acados_model.u.size()[0]
        ny = nx + nu
        n_param = acados_model.p.size()[0] if isinstance(acados_model.p, MX) else 0

        # create OCP object and initialize parameters

        ocp = AcadosOcp()
        ocp.model = acados_model
        ocp.solver_options.tf = self.t_horizon
        ocp.dims.N = self.n_nodes
        ocp.dims.np = n_param
        ocp.parameter_values = np.zeros(n_param)
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = np.diag(np.concatenate((self.Q_diagonal, self.R)))
        ocp.cost.W_e = np.diag(self.Q_diagonal)
        terminal_cost = 1
        ocp.cost.W_e *= terminal_cost
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :nx] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[-4:, -4:] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # set initial reference trajectory and state (to be overwritten)

        x_ref = np.zeros(nx)
        ocp.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0, 0.0])))
        ocp.cost.yref_e = x_ref
        ocp.constraints.x0 = x_ref

        # set constraints

        ocp.constraints.lbu = np.array([0] * 4)
        ocp.constraints.ubu = np.array([1] * 4)
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # set solver options

        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        return AcadosOcpSolver(ocp)
    
    def p_dynamics(self):
        """
        Symbolic representation of the time derivative of the position vector
        
        """
                
        return self.v


    def q_dynamics(self):
        """
        Symbolic representation of the time derivative of the attitude vector
        
        """
                
        return 1/2 * mtimes(skew_symmetric(self.w), self.q)


    def v_dynamics(self):
        """
        Symbolic representation of the time derivative of the linear velocity vector
        
        """

        f_thrust = self.u * self.quad.T_max
        g = self.quad.g
        a_thrust = vertcat(0.0, 0.0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3]) / self.quad.m

        return (v_dot_q(a_thrust, self.q) + g) + Ψ_inv(mtimes(self.A, self.z) + mtimes(self.B, self.x) + mtimes(self.C, self.u))
    

    def w_dynamics(self):
        """
        Symbolic representation of the time derivative of the angular velocity vector
        
        """
                
        f_thrust = self.u * self.quad.T_max
        
        d_x = MX(self.quad.d_x)
        d_y = MX(self.quad.d_y)
        c_tau = MX(self.quad.c_tau)

        return vertcat(
            (mtimes(f_thrust.T, d_y) + (self.quad.J[1] - self.quad.J[2]) * self.w[1] * self.w[2]) / self.quad.J[0],
            (-mtimes(f_thrust.T, d_x) + (self.quad.J[2] - self.quad.J[0]) * self.w[2] * self.w[0]) / self.quad.J[1],
            (mtimes(f_thrust.T, c_tau) + (self.quad.J[0] - self.quad.J[1]) * self.w[0] * self.w[1]) / self.quad.J[2])


    def z_dynamics(self):
        """
        Symbolic representation of the time derivative of the lifted state vector
        
        """

        return mtimes(self.A, self.z) + mtimes(self.B, self.x) + mtimes(self.C, self.u)

    def set_reference(self, z_ref, u_ref):
        """
        Sets the reference trajectory for MPC optimizer

        z_ref (np.array) :: Nxlifted_space_dim lifted reference trajectory containing a sequence of N tracking points
        u_ref (np.array) :: Nx4 target control input vector

        """
        
        if z_ref is None or u_ref is None:
            print("State or control references are empty.")
            sys.exit(1)
        
        assert z_ref[0].shape[0] == (u_ref.shape[0] + 1) or z_ref[0].shape[0] == u_ref.shape[0]

        # if there aren't enough states in target sequence, append last state until required length is met

        while z_ref[0].shape[0] < self.N + 1:
            z_ref = [np.concatenate((z, np.expand_dims(z[-1, :], 0)), 0) for z in z_ref]
            u_ref = np.concatenate((u_ref, np.expand_dims(u_ref[-1, :], 0)), 0)

        # set references

        z_ref = np.concatenate([z for z in z_ref], 1)

        for i in range(self.N):
            ref = z_ref[i, :]
            ref = np.concatenate((ref, u_ref[i, :]))
            self.acados_ocp_solver.set(i, "yref", ref)

        # the last MPC node has only a state reference but no input reference

        self.acados_ocp_solver.set(self.N, "yref", z_ref[self.N, :])

    def optimize(self):
        """
        Solve optimal control problems with initial quadrotor state set as an equality constraint
        
        return :: two numpy arrays will be returned:
        
                - u_opt_acados: a (flattened) optimized control input sequence
                - x_opt_acados: a optimized sequence of states

        NOTES & TO-DO:

            (1) I'm a little confused as to how calling 'self.acados_ocp_solver.set(0, 'lbx', x_init)' sets an equality constraint
            (2) What's the point of all these data transformations (e.g., 'np.stack', 'np.reshape')

        """

        # retrieve quadrotor state

        x_init = np.stack(self.quad.get_state())

        assert x_init is not None

        # lift states

        z_init = self.encoder.forward(x_init)

        # set initial conditions / equality constraints

        self.acados_ocp_solver.set(0, 'lbx', z_init)
        self.acados_ocp_solver.set(0, 'ubx', z_init)

        # solve OCP

        self.acados_ocp_solver.solve()

        # retrieve optimized control and state sequences

        u_opt_acados = np.ndarray((self.N, 4))
        z_opt_acados = np.ndarray((self.N+1, len(z_init)))
        z_opt_acados[0, :] = self.acados_ocp_solver.get(0, "x")

        for i in range(self.N):
            u_opt_acados[i, :] = self.acados_ocp_solver.get(i, "u")
            z_opt_acados[i+1, :] = self.acados_ocp_solver.get(i+1, "x")

        u_opt_acados = np.reshape(u_opt_acados, (-1))
        
        return u_opt_acados, z_opt_acados
    