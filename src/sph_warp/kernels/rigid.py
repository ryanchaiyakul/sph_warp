"""Kernels for rigid-fluid coupling."""

import warp as wp


@wp.kernel
def interpolate_rigid_states_kernel(
    body_q0: wp.array(dtype=wp.transform),
    body_q1: wp.array(dtype=wp.transform),
    body_qd0: wp.array(dtype=wp.spatial_vector),
    body_qd1: wp.array(dtype=wp.spatial_vector),
    alpha: float,
    body_q_interp: wp.array(dtype=wp.transform),
    body_qd_interp: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    # Interpolate position (linear)
    pos0 = wp.transform_get_translation(body_q0[tid])
    pos1 = wp.transform_get_translation(body_q1[tid])
    pos_interp = (1.0 - alpha) * pos0 + alpha * pos1

    # Interpolate rotation (SLERP)
    quat0 = wp.transform_get_rotation(body_q0[tid])
    quat1 = wp.transform_get_rotation(body_q1[tid])
    quat_interp = wp.quat_slerp(quat0, quat1, alpha)

    # Combine into interpolated transform
    body_q_interp[tid] = wp.transform(pos_interp, quat_interp)

    # Linear interpolation for velocities
    vel0 = body_qd0[tid]
    vel1 = body_qd1[tid]
    body_qd_interp[tid] = (1.0 - alpha) * vel0 + alpha * vel1


@wp.kernel
def divide_force_kernel(body_f: wp.array(dtype=wp.spatial_vector), count: float):
    tid = wp.tid()
    body_f[tid] = body_f[tid] / count
