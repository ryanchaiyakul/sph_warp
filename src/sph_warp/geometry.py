import numpy as np


def generate_box_particles(hx, hy, hz, dx):
    """Generate particles for a solid box (already solid in your original code)."""
    x = np.arange(-hx + dx / 2, hx, dx)
    y = np.arange(-hy + dx / 2, hy, dx)
    z = np.arange(-hz + dx / 2, hz, dx)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T


def generate_sphere_particles(radius, dx):
    """Generate volume particles for a solid sphere."""
    n_r = int(radius / dx)
    r_vals = np.linspace(dx / 2, radius, n_r)

    all_pts = []
    for r in r_vals:
        n_theta = max(1, int(2 * np.pi * r / dx))
        n_phi = max(1, int(np.pi * r / dx))
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

        x = r * np.sin(PHI) * np.cos(THETA)
        y = r * np.sin(PHI) * np.sin(THETA)
        z = r * np.cos(PHI)
        all_pts.append(np.vstack([x.ravel(), y.ravel(), z.ravel()]).T)

    return np.concatenate(all_pts) if all_pts else np.empty((0, 3))


def generate_cylinder_particles(radius, half_height, dx):
    """Generate volume particles for a solid cylinder."""
    n_r = int(radius / dx)
    r_vals = np.linspace(dx / 2, radius, n_r)
    n_z = int(2 * half_height / dx)
    z = np.linspace(-half_height, half_height, n_z)

    all_pts = []
    for r in r_vals:
        n_theta = max(1, int(2 * np.pi * r / dx))
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        THETA, Z = np.meshgrid(theta, z, indexing="ij")
        x = r * np.cos(THETA)
        y = r * np.sin(THETA)
        all_pts.append(np.vstack([x.ravel(), y.ravel(), Z.ravel()]).T)

    return np.concatenate(all_pts)


def generate_capsule_particles(radius, half_height, dx):
    """Generate volume particles for a solid capsule."""
    cylinder = generate_cylinder_particles(radius, half_height, dx)

    # Hemisphere caps
    n_r = int(radius / dx)
    r_vals = np.linspace(dx / 2, radius, n_r)
    cap_pts = []

    for r in r_vals:
        n_theta = max(1, int(2 * np.pi * r / dx))
        n_phi_cap = max(1, int((np.pi / 2) * r / dx))
        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        phi_cap = np.linspace(0, np.pi / 2, n_phi_cap)

        PHI_CAP, THETA_CAP = np.meshgrid(phi_cap, theta, indexing="ij")
        x = r * np.sin(PHI_CAP) * np.cos(THETA_CAP)
        y = r * np.sin(PHI_CAP) * np.sin(THETA_CAP)
        z_off = r * np.cos(PHI_CAP)

        cap_pts.append(
            np.vstack([x.ravel(), y.ravel(), (z_off + half_height).ravel()]).T
        )
        cap_pts.append(
            np.vstack([x.ravel(), y.ravel(), (-z_off - half_height).ravel()]).T
        )

    return np.vstack([cylinder] + cap_pts)


# TODO: not the best but okay
def generate_cone_particles(radius, half_height, dx):
    """Generate volume particles for a solid cone using grid-masking."""
    x = np.arange(-radius, radius + dx, dx)
    y = np.arange(-radius, radius + dx, dx)
    z = np.arange(-half_height, half_height + dx, dx)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    height_from_apex = half_height - Z
    total_height = 2 * half_height
    r_at_z = radius * (height_from_apex / total_height)
    dist_sq = X**2 + Y**2
    mask = (dist_sq <= r_at_z**2) & (height_from_apex >= 0)

    return np.vstack([X[mask], Y[mask], Z[mask]]).T


def generate_ellipsoid_particles(rx, ry, rz, dx):
    """Generate volume particles for a solid ellipsoid."""
    n_shells = int(max(rx, ry, rz) / dx)
    scales = np.linspace(dx / max(rx, ry, rz), 1.0, n_shells)

    all_pts = []
    for s in scales:
        curr_rx, curr_ry, curr_rz = rx * s, ry * s, rz * s
        n_theta = max(1, int(2 * np.pi * max(curr_rx, curr_ry) / dx))
        n_phi = max(1, int(np.pi * max(curr_rx, curr_ry, curr_rz) / dx))

        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
        phi = np.linspace(0, np.pi, n_phi)
        THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

        x = curr_rx * np.sin(PHI) * np.cos(THETA)
        y = curr_ry * np.sin(PHI) * np.sin(THETA)
        z = curr_rz * np.cos(PHI)
        all_pts.append(np.vstack([x.ravel(), y.ravel(), z.ravel()]).T)

    return np.concatenate(all_pts)
