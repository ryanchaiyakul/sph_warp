import warp as wp

from newton import ModelBuilder, CollisionPipeline
from newton.solvers import SolverSemiImplicit

from sph_warp import SolverWCSPH, FluidRigidCoupler, CoupledRender


if __name__ == "__main__":
    # Initialize rigid solver
    rigid_builder = ModelBuilder()

    # Drop height for all shapes
    drop_height = 1.5
    spacing = 0.5
    x_gap = 0.5
    y_gap = 0.5

    # BOX
    body_id = rigid_builder.add_body(label="box")
    rigid_builder.add_shape_box(
        body=body_id,
        xform=[spacing, y_gap, drop_height, 0.0, 0.0, 0.0, 1.0],
        hx=0.125,
        hy=0.125,
        hz=0.125,
        cfg=ModelBuilder.ShapeConfig(density=1000.0),
        label="box_shape",
    )

    # SPHERE
    body_id = rigid_builder.add_body(label="sphere")
    rigid_builder.add_shape_sphere(
        body=body_id,
        xform=[spacing * 2, y_gap, drop_height, 0.0, 0.0, 0.0, 1.0],
        radius=0.125,
        cfg=ModelBuilder.ShapeConfig(density=1000.0),
        label="sphere_shape",
    )

    # CAPSULE
    body_id = rigid_builder.add_body(label="capsule")
    rigid_builder.add_shape_capsule(
        body=body_id,
        xform=[spacing * 3, y_gap, drop_height, 0.0, 0.0, 0.0, 1.0],
        radius=0.125,
        half_height=0.25,
        cfg=ModelBuilder.ShapeConfig(density=1000.0),
        label="capsule_shape",
    )

    # CYLINDER
    body_id = rigid_builder.add_body(label="cylinder")
    rigid_builder.add_shape_cylinder(
        body=body_id,
        xform=[spacing * 4, y_gap, drop_height, 0.0, 0.0, 0.0, 1.0],
        radius=0.125,
        half_height=0.25,
        cfg=ModelBuilder.ShapeConfig(density=1000.0),
        label="cylinder_shape",
    )

    # CONE
    body_id = rigid_builder.add_body(label="cone")
    rigid_builder.add_shape_cone(
        body=body_id,
        xform=[spacing * 5, y_gap, drop_height, 0.1, 0.2, 0.3, 0.9],
        radius=0.125,
        half_height=0.25,
        cfg=ModelBuilder.ShapeConfig(density=1000.0),
        label="cone_shape",
    )

    # ELLIPSOID
    body_id = rigid_builder.add_body(label="ellipsoid")
    rigid_builder.add_shape_ellipsoid(
        body=body_id,
        xform=[spacing * 6, y_gap, drop_height, 0.0, 0.0, 0.0, 1.0],
        a=0.15,
        b=0.10,
        c=0.125,
        cfg=ModelBuilder.ShapeConfig(density=1000.0),
        label="ellipsoid_shape",
    )

    rigid_model = rigid_builder.finalize()
    rigid_solver = SolverSemiImplicit(rigid_model)
    collision_pipeline = CollisionPipeline(rigid_model, broad_phase="sap")
    rigid_contacts = collision_pipeline.contacts()

    # Initialize fluid solver
    fluid_builder = ModelBuilder()
    SolverWCSPH.register_custom_attributes(fluid_builder)
    SolverWCSPH.add_fluid_block(fluid_builder, [3.5, 1.0, 1.0])
    FluidRigidCoupler.add_rigid_bodies(fluid_builder, rigid_model)
    fluid_model = fluid_builder.finalize()
    fluid_solver = SolverWCSPH(fluid_model)
    coupler = FluidRigidCoupler(fluid_model, rigid_model)

    # Initialize states for GPU swapping
    rigid_state0 = rigid_model.state()
    rigid_state1 = rigid_model.state()
    rigid_state2 = rigid_model.state()  # for interp

    fluid_state0 = fluid_model.state()
    fluid_state1 = fluid_model.state()

    renderer = CoupledRender(rigid_builder, fluid_builder, fluid_model)

    fps = 60
    frames = 360
    dt_fluid = 4.52e-4
    rigid_ratio = 8

    dt_rigid = dt_fluid * rigid_ratio
    macro_steps = int((1.0 / fps) / dt_rigid)

    for f in range(frames):
        with wp.ScopedTimer("step"):
            for _ in range(macro_steps):
                # rigid step
                collision_pipeline.collide(rigid_state0, rigid_contacts)
                rigid_solver.step(
                    rigid_state0, rigid_state1, None, rigid_contacts, dt_rigid
                )
                # eval_fk(rigid_model, rigid_state1.joint_q, rigid_state1.joint_qd, rigid_state1)
                rigid_state1.clear_forces()

                for i in range(rigid_ratio):
                    fluid_state0.clear_forces()

                    # r_s2 = (1 - α)r_s0 + αr_s1
                    alpha = (i + 1.0) / rigid_ratio
                    coupler.interpolate_rigid_states(
                        rigid_state0, rigid_state1, alpha, rigid_state2
                    )

                    # fluid step
                    coupler.update_fluid_boundaries(rigid_state2, fluid_state0)
                    fluid_solver.step(fluid_state0, fluid_state1, None, None, dt_fluid)
                    coupler.apply_fluid_forces(fluid_state1, rigid_state1)
                    fluid_state0, fluid_state1 = fluid_state1, fluid_state0

                # divide by # of steps
                coupler.average_rigid_forces(rigid_state1, rigid_ratio)
                rigid_state0, rigid_state1 = rigid_state1, rigid_state0

        renderer.render_frame(f * (1 / fps), rigid_state0, fluid_state0)

    renderer.close()
