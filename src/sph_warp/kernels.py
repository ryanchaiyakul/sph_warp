import warp as wp


@wp.func
def square(q: float):
    return q * q


@wp.func
def cube(q: float):
    return q * q * q


@wp.func
def cubic(r: wp.vec3, h: float) -> float:
    k = 8.0 / wp.pi / cube(h)
    q = wp.length(r) / h
    if q <= 1.0:
        if q <= 0.5:
            return k * (6.0 * cube(q) - 6.0 * square(q) + 1.0)
        return k * 2.0 * cube(1.0 - q)
    return 0.0


@wp.func
def diff_cubic(r: wp.vec3, h: float) -> wp.vec3:
    k = 48.0 / wp.pi / cube(h)
    r_norm = wp.length(r)
    q = r_norm / h
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            return k * q * (3.0 * q - 2.0) * grad_q
        return -k * square(1.0 - q) * grad_q
    return wp.vec3(0.0)


@wp.func
def get_viscosity(
    r: wp.vec3,  # q[i] - q[j]
    v: wp.vec3,  # qd[i] - qd[j]
    rho_i: float,
    rho_j: float,
    h: float,
    c_s: float,
    alpha: float,  # between 0.08 and 0.5
    epsilon: float = 0.01,
) -> float:
    dot_v_p = wp.dot(v, r)
    if dot_v_p < 0.0:
        mu_ij = dot_v_p / (wp.dot(r, r) + epsilon * square(h))
        viscous_term = (2.0 * alpha * h * c_s) / (rho_i + rho_j)
        return -viscous_term * mu_ij
    return 0.0


@wp.kernel
def update_rigid_particles(
    particle_q0_local: wp.array(dtype=wp.vec3),
    particle_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    body_id = particle_body[tid]

    if body_id >= 0:
        X_wb = body_q[body_id]
        particle_q[tid] = wp.transform_point(X_wb, particle_q0_local[tid])
        # TODO: update particle_qd


@wp.kernel
def compute_particle_q0_local(
    particle_q0: wp.array(dtype=wp.vec3),
    particle_body: wp.array(dtype=int),
    body_q0: wp.array(dtype=wp.transform),
    particle_q0_local: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    body_id = particle_body[tid]

    if body_id >= 0:
        X_bw = wp.transform_inverse(body_q0[body_id])
        particle_q0_local[tid] = wp.transform_point(X_bw, particle_q0[tid])


@wp.kernel
def compute_body_f(
    particle_q: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()
    body_id = particle_body[tid]

    if body_id >= 0:
        f = particle_f[tid]
        pos = particle_q[tid]
        r = pos - wp.transform_point(body_q[body_id], body_com[body_id])
        wp.atomic_add(body_f, body_id, wp.spatial_vector(f, wp.cross(r, f)))


@wp.kernel
def compute_m_V(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_body: wp.array(dtype=int),
    h: float,
    rho0: float,
    m_V: wp.array(dtype=float),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if particle_body[i] < 0:
        return

    q_i = particle_q[i]
    delta = float(0.0)
    for j in wp.hash_grid_query(grid, q_i, h):
        if particle_body[j] >= 0:
            delta += cubic(q_i - particle_q[j], h)

    m_V[i] = rho0 * 1.0 / delta if delta > 0.0 else 0.0


@wp.kernel
def compute_rho(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_body: wp.array(dtype=int),
    m_V: wp.array(dtype=float),
    h: float,
    rho: wp.array(dtype=float),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if particle_body[i] >= 0:
        return

    q_i = particle_q[i]
    rho_sum = float(0.0)
    for j in wp.hash_grid_query(grid, q_i, h):
        rho_sum += m_V[j] * cubic(q_i - particle_q[j], h)
    rho[i] = rho_sum


@wp.kernel
def compute_p(
    rho: wp.array(dtype=float),
    particle_body: wp.array(dtype=int),
    rho0: float,
    stiffness: float,
    p: wp.array(dtype=float),
):
    tid = wp.tid()
    if particle_body[tid] >= 0:
        return

    p[tid] = stiffness * (wp.pow(wp.max(rho[tid] / rho0, 1.0), 7.0) - 1.0)


@wp.kernel
def compute_particle_f(
    grid: wp.uint64,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    particle_body: wp.array(dtype=int),
    m_V: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    p: wp.array(dtype=float),
    h: float,
    rho0: float,
    c_s: float,
    alpha: float,
    gravity: wp.vec3,
    particle_f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    if particle_body[i] >= 0:
        return

    q_i, qd_i, rho_i = particle_q[i], particle_qd[i], rho[i]
    dp_i = p[i] / square(rho_i)
    a_sum = wp.vec3(0.0)
    for j in wp.hash_grid_query(grid, q_i, h):
        if i == j:
            continue

        dist = q_i - particle_q[j]
        del_v = qd_i - particle_qd[j]
        grad_w = diff_cubic(dist, h)

        is_rigid = particle_body[j] >= 0
        rho_j = rho0 if is_rigid else rho[j]
        dp_j = (p[i] if is_rigid else p[j]) / square(rho_j)
        visc_ij = (
            0.0 if is_rigid else get_viscosity(dist, del_v, rho_i, rho_j, h, c_s, alpha)
        )
        a = -m_V[j] * (dp_i + dp_j + visc_ij) * grad_w
        a_sum += a

        if is_rigid:
            wp.atomic_add(particle_f, j, -m_V[i] * a)

    particle_f[i] = m_V[i] * (a_sum + gravity)


@wp.kernel
def advect(
    particle_q_in: wp.array(dtype=wp.vec3),
    particle_qd_in: wp.array(dtype=wp.vec3),
    particle_f_in: wp.array(dtype=wp.vec3),
    particle_body: wp.array(dtype=int),
    m_V: wp.array(dtype=float),
    dt: float,
    particle_q_out: wp.array(dtype=wp.vec3),
    particle_qd_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if particle_body[tid] >= 0:
        return

    v = particle_qd_in[tid] + (particle_f_in[tid] / m_V[tid]) * dt
    x = particle_q_in[tid] + v * dt

    # Simple bounds w/ collison
    # TODO: parametrize bounds
    length = 3.5
    width = 1.0
    restitution = 0.5

    normal = wp.vec3(0.0)
    if x[0] < 0.0:
        x = wp.vec3(0.0, x[1], x[2])
        normal += wp.vec3(1.0, 0.0, 0.0)
    elif x[0] > length:
        x = wp.vec3(length, x[1], x[2])
        normal += wp.vec3(-1.0, 0.0, 0.0)
    if x[1] < 0.0:
        x = wp.vec3(x[0], 0.0, x[2])
        normal += wp.vec3(0.0, 1.0, 0.0)
    elif x[1] > width:
        x = wp.vec3(x[0], width, x[2])
        normal += wp.vec3(0.0, -1.0, 0.0)
    if x[2] < 0.0:
        x = wp.vec3(x[0], x[1], 0.0)
        normal += wp.vec3(0.0, 0.0, 1.0)

    mag = wp.length(normal)
    if mag > 0.0:
        n = wp.normalize(normal)
        v -= (1.0 + restitution) * wp.dot(v, n) * n

    particle_q_out[tid] = x
    particle_qd_out[tid] = v


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
def average_forces_kernel(body_f: wp.array(dtype=wp.spatial_vector), count: float):
    tid = wp.tid()
    body_f[tid] = body_f[tid] / count
