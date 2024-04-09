""" 
Set of utility functions for running MPC optimization loop. Functions copied from 
https://github.com/uzh-rpg/data_driven_mpc/tree/main


Created by: Kian Molani
Last updated: Sept. 25, 2023

"""

import numpy as np


def get_reference_chunk(reference_traj, reference_u, current_idx, n_mpc_nodes, reference_over_sampling):
    """
    Extracts the reference states and controls for the current MPC optimization given the over-sampled counterparts

    reference_traj (np.array)     :: reference trajectory, which has been finely over-sampled by a factor of 'reference_over_sampling'. It should be a vector of shape (Nx13), where N is the length of the trajectory in samples
    reference_u (np.array)        :: reference controls, following the same requirements as reference_traj. Should be a vector of shape (Nx4)
    current_idx (int)             :: current index of the trajectory tracking. Should be an integer number between 0 and N-1
    n_mpc_nodes (int)             :: number of MPC nodes considered in the optimization
    reference_over_sampling (int) :: the over-sampling factor of the reference trajectories. Should be a positive integer
    return                        :: returns the chunks of reference selected for the current MPC iteration. Two numpy arrays will be returned:

            - An ((N+1)x13) array, corresponding to the reference trajectory. The first row is the state of current_idx
            - An (Nx4) array, corresponding to the reference controls

    NOTES & TO-DO:

        (1) I do not fully understand what this method is doing, nor why it is needed for MPC. It seems to be related with 'n_mpc_nodes' and how references are set in acados when dealing with sequences of tracking points. See 'set_reference()' method in 'quad_mpc.py' for clarification
    
    """

    # dense references

    ref_traj_chunk = reference_traj[current_idx:current_idx + (n_mpc_nodes + 1) * reference_over_sampling, :]
    ref_u_chunk = reference_u[current_idx:current_idx + n_mpc_nodes * reference_over_sampling, :]

    # indices for down-sampling the reference to number of MPC nodes

    downsample_ref_ind = np.arange(0, min(reference_over_sampling * (n_mpc_nodes + 1), ref_traj_chunk.shape[0]),
                                   reference_over_sampling, dtype=int)

    # sparser references (same dt as node separation)

    ref_traj_chunk = ref_traj_chunk[downsample_ref_ind, :]
    ref_u_chunk = ref_u_chunk[downsample_ref_ind[:max(len(downsample_ref_ind) - 1, 1)], :]

    return ref_traj_chunk, ref_u_chunk
