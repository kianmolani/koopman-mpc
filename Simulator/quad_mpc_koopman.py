""" 
Implementation of Quadrotor MPC class for Koopman representations. This class contains 
properties and functions inherent and required by MPC control schemes.


Created by: Kian Molani
Last updated: Oct. 31, 2023

"""

import sys

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from casadi import *
from quad import Quad

from utils.math import skew_symmetric


class QuadKoopmanMPC:
    def __init__(self, quad: Quad, lifted_state_dim: int, no_control_inputs: int, 
                 K: np.array, t_horizon: float, n_nodes: int):
        """
        Initialization of quadrotor MPC class for Koopman representations

        quad (Quad)             :: 3D quadrotor model
        lifted_state_dim (int)  :: dimension of state vector in lifted space
        no_control_inputs (int) :: dimension of control vector
        K (np.array)            :: Koopman operator
        t_horizon (float)       :: prediction horizon
        n_nodes (int)           :: number of control nodes within prediction horizon
        
        """

        self.quad = Quad
        self.lifted_state_dim = lifted_state_dim
        self.no_control_inputs = no_control_inputs
        self.K = K

        # declare model states as CasADi symbolic variables

        self.z = MX.sym('z', lifted_state_dim)

        # declare control inputs as CasADi symbolic variables

        u = MX.sym('u', no_control_inputs)

        # declare time derivative of model states governing drone dynamics

        self.z_dot = self.z_dynamics()

        # create CasADi Function object for DAEs

        self.z_dot = Function('z_dot', [self.z, self.u], [self.z_dot], ['z', 'u'], ['z_dot'])

        # formulate acados dynamics model from CasADi model

        self.acados_model = self.setup_acados_model(self.z_dot(z=self.z, u=self.u)['z_dot'])

        # setup and compile acados OCP solver

        self.acados_ocp_solver = self.setup_acados_ocp(self.acados_model)


    def setup_acados_model(self, casadi_model):
        """
        Builds acados symbolic model using CasADi expressions. Needed when creating acados OCP
        solver. Method inspired from https://github.com/uzh-rpg/data_driven_mpc/tree/main

        casadi_model (CasADi Function) :: CasADi Function object representing symbolic model of quadrotor dynamics

        """

        acados_model = AcadosModel()
        acados_model.name = "koopman"
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

        ocp = AcadosOcp()
        ocp.model = acados_model
        ocp.solver_options.tf = self.t_horizon
        ocp.dims.N = self.n_nodes









        ocp.dims.np = n_param
        ocp.parameter_values = np.zeros(n_param)
        
        nx = acados_model.x.size()[0]
        nu = acados_model.u.size()[0]
        ny = nx + nu
        n_param = acados_model.p.size()[0] if isinstance(acados_model.p, MX) else 0

        # create OCP object and initialize parameters




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
    

    def z_dynamics(self):
        """
        Symbolic representation of the time derivative of the lifted state vector
        
        """
                
        return mtimes(self.K, self.z)


    def set_reference(self, x_ref, u_ref):
        """
        Sets the reference trajectory for MPC optimizer

        x_ref (np.array) :: Nx13 reference trajectory containing a sequence of N tracking points
        u_ref (np.array) :: Nx4 target control input vector
        r  
        NOTES & TO-DO:

            (1) Try to eliminate use of 'separate_variables' (had to re-incoporate due to apparent need to append states)
            (2) I'm unclear as to why we're setting reference trajectories as Nx13 chunks, where 'N' is equal to 'n_mpc_nodes'. In general, I need greater understanding of the programmatic incorporation of reference states in OCPs in acados
            (3) I did not know that we could combine state and control references in this way. How does acados make the distinction between what's our state and control constraints?
            (3) Be careful of redundant assertions / checks here
            (4) I don't fully understand when and why condition 'x_ref.shape[0] < self.N + 1' would occur, and why we'd need to take corrective measures

        """
        
        if x_ref is None or u_ref is None:
            print("State or control references are empty.")
            sys.exit(1)
        
        assert x_ref[0].shape[0] == (u_ref.shape[0] + 1) or x_ref[0].shape[0] == u_ref.shape[0]

        # if there aren't enough states in target sequence, append last state until required length is met

        while x_ref[0].shape[0] < self.N + 1:
            x_ref = [np.concatenate((x, np.expand_dims(x[-1, :], 0)), 0) for x in x_ref]
            u_ref = np.concatenate((u_ref, np.expand_dims(u_ref[-1, :], 0)), 0)

        # set references

        x_ref = np.concatenate([x for x in x_ref], 1)

        for i in range(self.N):
            ref = x_ref[i, :]
            ref = np.concatenate((ref, u_ref[i, :]))
            self.acados_ocp_solver.set(i, "yref", ref)

        # the last MPC node has only a state reference but no input reference

        self.acados_ocp_solver.set(self.N, "yref", x_ref[self.N, :])


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

        # set initial conditions / equality constraints

        self.acados_ocp_solver.set(0, 'lbx', x_init)
        self.acados_ocp_solver.set(0, 'ubx', x_init)

        # solve OCP

        self.acados_ocp_solver.solve()

        # retrieve optimized control and state sequences

        u_opt_acados = np.ndarray((self.N, 4))
        x_opt_acados = np.ndarray((self.N+1, len(x_init)))
        x_opt_acados[0, :] = self.acados_ocp_solver.get(0, "x")

        for i in range(self.N):
            u_opt_acados[i, :] = self.acados_ocp_solver.get(i, "u")
            x_opt_acados[i+1, :] = self.acados_ocp_solver.get(i+1, "x")

        u_opt_acados = np.reshape(u_opt_acados, (-1))
        
        return u_opt_acados, x_opt_acados
    