import warp as wp

from newton import ModelBuilder
from newton.viewer import ViewerGL

from sph_warp import SolverWCSPH

if __name__ == "__main__":
    wp.init()
    builder = ModelBuilder()
    SolverWCSPH.register_custom_attributes(builder)

    rho0 = 1000.0
    h = 0.1
    dx = 0.05

    # Water Column Dimensions (H = 4.0m as per your text)
    # Reducing width/length for fewer particles
    column_width = 1.0
    column_height = 4.0
    column_length = 1.0

    # Calculate particles per dimension
    dim_x = int(column_width / dx)
    dim_y = int(column_length / dx)
    dim_z = int(column_height / dx)

    # Mass of one particle: Volume per particle * Density
    particle_vol = dx**3
    particle_mass = particle_vol * rho0

    print(f"Simulating {dim_x * dim_y * dim_z} particles...")

    # Build the water column
    builder.add_particle_grid(
        pos=wp.vec3(0.0, 0.0, 0.05),  # Slightly above floor
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=dim_x,
        dim_y=dim_y,
        dim_z=dim_z,
        cell_x=dx,
        cell_y=dx,
        cell_z=dx,
        mass=particle_mass,
        radius_mean=dx * 0.5,
        jitter=0.1 * h,
    )

    model = builder.finalize()
    solver = SolverWCSPH(model)

    # Override defaults with paper's specific WCSPH values
    solver.rho0 = 1000.0
    solver.h = 0.1
    solver.stiffness = 1119000.0
    solver.c_s = 88.5
    solver.alpha = 0.1  # Start low for stability

    # The exact time step from the paper
    dt = 4.52e-4

    state0 = model.state()
    state1 = model.state()

    # Tank bounds (Mirroring the paper's dam break setup)
    # You may need to update your 'advect' kernel to use these limits
    tank_size = 5.0

    viewer = ViewerGL()
    viewer.set_model(model)
    viewer.show_particles = True
    fps = 60
    sim_substeps = int((1.0 / fps) / dt)

    for f in range(200):  # Total frames
        for _ in range(sim_substeps):
            solver.step(state0, state1, None, None, dt)
            state0, state1 = state1, state0

        viewer.begin_frame(f)
        viewer.log_state(state1)
        viewer.end_frame()
