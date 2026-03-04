from newton import ModelBuilder
from sph_warp import SolverWCSPH

import numpy as np
import warp as wp
import warp.render

if __name__ == "__main__":
    builder = ModelBuilder()
    SolverWCSPH.register_custom_attributes(builder)

    # 1. SETUP PARAMETERS
    h = 0.1
    m = 1.0
    # Place two particles heading for a head-on collision
    # Initially outside h-range, they will enter it around Frame 10
    builder.add_particle(
        pos=wp.vec3(-0.06, 0.0, 0.1), vel=wp.vec3(0.5, 0.0, 0.0), mass=m, radius=0.02
    )
    builder.add_particle(
        pos=wp.vec3(0.06, 0.0, 0.1), vel=wp.vec3(-0.5, 0.0, 0.0), mass=m, radius=0.02
    )

    model = builder.finalize()
    solver = SolverWCSPH(model)

    # 2. CALIBRATION FOR TEST
    solver.stiffness = 50000.0  # Medium stiffness to see clear compression
    solver.gravity = wp.vec3(0.0, 0.0, 0.0)  # Turn off gravity to isolate SPH forces

    state0 = model.state()
    state1 = model.state()
    dt = 1e-4

    print(f"{'Frame':<8} | {'Dist':<10} | {'RelVel':<10} | {'P0_Acc_X':<10}")
    print("-" * 50)

    for f in range(1000):
        solver.step(state0, state1, None, None, dt)

        # Access data for verification
        q = state1.particle_q.numpy()
        v = state1.particle_qd.numpy()
        f_vec = state0.particle_f.numpy()  # Force from the previous step calculation

        dist = np.linalg.norm(q[0] - q[1])
        rel_vel = v[0][0] - v[1][0]
        acc_x = f_vec[0][0] / m

        # Every 5 frames, print the status
        if f % 10 == 0:
            status = "Approaching" if dist > h else "INTERACTING"
            print(f"{f:<8} | {dist:.6f} | {rel_vel:.6f} | {acc_x:.6f} ({status})")

        state0, state1 = state1, state0

    print("-" * 50)
    print(
        "TEST COMPLETE: If acc_x became negative after interaction, pressure is REPULSIVE."
    )


def main():
    sand_builder = ModelBuilder()
    SolverWCSPH.register_custom_attributes(sand_builder)
    particles_per_cell = 3.0
    density = 2500.0
    voxel_size = 0.05

    bed_lo = np.array([-1.0, -1.0, 0.0])
    bed_hi = np.array([1.0, 1.0, 0.5])
    bed_res = np.array(
        np.ceil(particles_per_cell * (bed_hi - bed_lo) / voxel_size), dtype=int
    )

    cell_size = (bed_hi - bed_lo) / bed_res
    cell_volume = np.prod(cell_size)
    radius = float(np.max(cell_size) * 0.5)
    mass = float(np.prod(cell_volume) * density)

    sand_builder.add_particle_grid(
        pos=wp.vec3(bed_lo),
        rot=wp.quat_identity(),  # type: ignore
        vel=wp.vec3(0.0),
        dim_x=bed_res[0] + 1,
        dim_y=bed_res[1] + 1,
        dim_z=bed_res[2] + 1,
        cell_x=cell_size[0],
        cell_y=cell_size[1],
        cell_z=cell_size[2],
        mass=mass,
        jitter=0.0 * radius,
        radius_mean=radius,
        # custom_attributes={"mpm:friction": 0.75},
    )
    sand_model = sand_builder.finalize()
    solver = SolverWCSPH(sand_model)

    state0 = sand_model.state()
    state1 = sand_model.state()
    renderer = wp.render.OpenGLRenderer(up_axis="Z")
    dt = 1e-3
    frames = 500

    for f in range(frames):
        solver.step(state0, state1, None, None, dt)
        renderer.begin_frame(f * dt)
        renderer.render_points(
            name="fluid",
            points=state1.particle_q.numpy(),
            radius=radius,
            colors=(0.2, 0.6, 0.9),
        )
        renderer.end_frame()
        state0, state1 = state1, state0
    renderer.save()
