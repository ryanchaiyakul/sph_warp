import warp as wp


@wp.func
def square(q: float):
    return q * q


@wp.func
def cube(q: float):
    return q * q * q


# W(x_i-x_j, h)
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
def compute_rho(
    grid: wp.uint64,
    q: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    h: float,
    rho: wp.array(dtype=float),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    q_i = q[i]
    rho_sum = float(0.0)
    neighbors = wp.hash_grid_query(grid, q_i, h)
    for index in neighbors:
        rho_sum += mass[index] * cubic(q_i - q[index], h)
    rho[i] = rho_sum


@wp.kernel
def compute_p(
    rho: wp.array(dtype=float),
    rho0: float,
    stiffness: float,
    p: wp.array(dtype=float),
):
    tid = wp.tid()
    p[tid] = stiffness * (wp.pow(wp.max(rho[tid] / rho0, 1.0), 7.0) - 1.0)


@wp.kernel
def compute_particle_f(
    grid: wp.uint64,
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    rho: wp.array(dtype=float),
    p: wp.array(dtype=float),
    h: float,
    c_s: float,
    alpha: float,
    gravity: wp.vec3,
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    q_i = q[i]
    qd_i = qd[i]
    rho_i = rho[i]
    f_sum = wp.vec3(0.0)
    for j in wp.hash_grid_query(grid, q_i, h):
        if i == j:
            continue
        distance = q_i - q[j]
        del_v = qd_i - qd[j]
        dpi = p[i] / square(rho_i)
        dpj = p[j] / square(rho[j])
        visc_ij = get_viscosity(distance, del_v, rho_i, rho[j], h, c_s, alpha)
        f_sum += -mass[j] * (dpi + dpj + visc_ij) * diff_cubic(q_i - q[j], h)
    f[i] = mass[i] * (f_sum + gravity)


@wp.kernel
def advect(
    q_in: wp.array(dtype=wp.vec3),
    qd_in: wp.array(dtype=wp.vec3),
    f_in: wp.array(dtype=wp.vec3),
    m: wp.array(dtype=float),
    dt: float,
    q_out: wp.array(dtype=wp.vec3),
    qd_out: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    v = qd_in[tid] + (f_in[tid] / m[tid]) * dt
    x = q_in[tid] + v * dt

    # Simple bounds
    length = 2.0
    width = 2.0
    damping = -0.5

    if x[0] < -length:
        x = wp.vec3(-length, x[1], x[2])
        v = wp.vec3(v[0] * damping, v[1], v[2])
    if x[0] > length:
        x = wp.vec3(length, x[1], x[2])
        v = wp.vec3(v[0] * damping, v[1], v[2])
    if x[1] < -width:
        x = wp.vec3(x[0], -width, x[2])
        v = wp.vec3(v[0], v[1] * damping, v[2])
    if x[1] > width:
        x = wp.vec3(x[0], width, x[2])
        v = wp.vec3(v[0], v[1] * damping, v[2])
    if x[2] < 0.0:
        x = wp.vec3(x[0], x[1], 0.0)
        v = wp.vec3(v[0], v[1], v[2] * damping)

    q_out[tid] = x
    qd_out[tid] = v
