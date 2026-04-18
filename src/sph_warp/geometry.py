import numpy as np


def generate_box_particles(hx, hy, hz, dx):
    """Generate particles for a box with given half-extents."""
    x = np.arange(-hx + dx / 2, hx, dx)
    y = np.arange(-hy + dx / 2, hy, dx)
    z = np.arange(-hz + dx / 2, hz, dx)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T


def generate_sphere_particles(radius, dx):
    """Generate surface particles for a sphere."""
    n_theta = int(2 * np.pi * radius / dx)
    n_phi = int(np.pi * radius / dx)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    x = radius * np.sin(PHI) * np.cos(THETA)
    y = radius * np.sin(PHI) * np.sin(THETA)
    z = radius * np.cos(PHI)
    return np.vstack([x.ravel(), y.ravel(), z.ravel()]).T


def generate_capsule_particles(radius, half_height, dx):
    """Generate surface particles for a capsule."""
    # Cylinder part
    n_theta = int(2 * np.pi * radius / dx)
    n_height = int(2 * half_height / dx)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(-half_height, half_height, n_height)
    THETA, Z = np.meshgrid(theta, z, indexing="ij")
    x = radius * np.cos(THETA)
    y = radius * np.sin(THETA)
    cylinder_pts = np.vstack([x.ravel(), y.ravel(), Z.ravel()]).T

    # Hemispherical caps
    n_cap = int(np.pi * radius / dx)
    phi_cap = np.linspace(0, np.pi / 2, n_cap)
    PHI_CAP, THETA_CAP = np.meshgrid(phi_cap, theta, indexing="ij")
    x_cap = radius * np.sin(PHI_CAP) * np.cos(THETA_CAP)
    y_cap = radius * np.sin(PHI_CAP) * np.sin(THETA_CAP)
    z_top = radius * np.cos(PHI_CAP) + half_height
    z_bottom = -radius * np.cos(PHI_CAP) - half_height
    top_pts = np.vstack([x_cap.ravel(), y_cap.ravel(), z_top.ravel()]).T
    bottom_pts = np.vstack([x_cap.ravel(), y_cap.ravel(), z_bottom.ravel()]).T

    return np.vstack([cylinder_pts, top_pts, bottom_pts])


def generate_cylinder_particles(radius, half_height, dx):
    """Generate surface particles for a cylinder."""
    n_theta = int(2 * np.pi * radius / dx)
    n_height = int(2 * half_height / dx)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(-half_height, half_height, n_height)
    THETA, Z = np.meshgrid(theta, z, indexing="ij")
    x = radius * np.cos(THETA)
    y = radius * np.sin(THETA)
    return np.vstack([x.ravel(), y.ravel(), Z.ravel()]).T


def generate_cone_particles(radius, half_height, dx):
    """Generate surface particles for a cone."""
    n_theta = int(2 * np.pi * radius / dx)
    n_height = int(half_height / dx)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    z = np.linspace(-half_height, half_height, n_height)
    THETA, Z = np.meshgrid(theta, z, indexing="ij")
    # Linear interpolation of radius from base to apex
    r_at_z = radius * (0.5 - Z / (2 * half_height))
    x = r_at_z * np.cos(THETA)
    y = r_at_z * np.sin(THETA)
    return np.vstack([x.ravel(), y.ravel(), Z.ravel()]).T


def generate_ellipsoid_particles(rx, ry, rz, dx):
    """Generate surface particles for an ellipsoid."""
    n_theta = int(2 * np.pi * max(rx, ry) / dx)
    n_phi = int(np.pi * max(rx, ry, rz) / dx)
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    x = rx * np.sin(PHI) * np.cos(THETA)
    y = ry * np.sin(PHI) * np.sin(THETA)
    z = rz * np.cos(PHI)
    return np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
