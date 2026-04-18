import warp as wp
import numpy as np
from newton import ModelBuilder, Model, State, GeoType
from .constants import WCSPH
from .sph import SPHConfig
from .kernels import (
    compute_particle_q0_local,
    update_rigid_particles,
    compute_body_f,
    interpolate_rigid_states_kernel,
    divide_force_kernel,
)
from .geometry import (
    generate_box_particles,
    generate_capsule_particles,
    generate_cone_particles,
    generate_cylinder_particles,
    generate_ellipsoid_particles,
    generate_sphere_particles,
)


class FluidRigidCoupler:
    def __init__(self, fluid_model: Model, rigid_model: Model):
        self.n = fluid_model.particle_count
        self.particle_body = getattr(fluid_model, WCSPH).particle_body
        self.body_com = getattr(fluid_model, WCSPH).body_com

        # Local offset of particles relative to rigid bodies
        self.particle_q0_local = wp.zeros(
            self.n, dtype=wp.vec3, device=fluid_model.device
        )
        wp.launch(
            compute_particle_q0_local,
            self.n,
            inputs=[fluid_model.particle_q, self.particle_body, rigid_model.body_q],
            outputs=[self.particle_q0_local],
        )

    @staticmethod
    def add_rigid_bodies(
        fluid_builder: ModelBuilder,
        rigid_model: Model,
        config: SPHConfig = SPHConfig(),
        next_id: int = 0,
    ) -> int:
        # Alias shape arrays
        shape_types = rigid_model.shape_type.numpy()  # type: ignore
        shape_bodies = rigid_model.shape_body.numpy()  # type: ignore
        shape_transforms = rigid_model.shape_transform.numpy()  # type: ignore
        shape_scales = rigid_model.shape_scale.numpy()  # type: ignore

        dx = config.dx

        # For each body
        for body_idx, (com, body_q) in enumerate(
            zip(rigid_model.body_com.numpy(), rigid_model.body_q.numpy())  # type: ignore
        ):
            fluid_builder.add_custom_values(**{f"{WCSPH}:body_com": com})

            # Collect all particles for this body
            all_local_pts = []
            all_shape_transforms = []

            # We can have multiple contact primitives for each body
            for shape_idx in np.where(shape_bodies == body_idx)[0]:
                shape_type = shape_types[shape_idx]
                shape_xform = shape_transforms[shape_idx]
                scale = shape_scales[shape_idx]

                if shape_type == GeoType.BOX:
                    # Box: scale = (hx, hy, hz) half-extents [1](#15-0)
                    hx, hy, hz = scale[:3]
                    local_pts = generate_box_particles(hx, hy, hz, dx)
                elif shape_type == GeoType.SPHERE:
                    # Sphere: scale.x = radius [1](#15-0)
                    radius = scale[0]
                    local_pts = generate_sphere_particles(radius, dx)
                elif shape_type == GeoType.CAPSULE:
                    # Capsule: scale.x = radius, scale.y = half_height [1](#15-0)
                    radius = scale[0]
                    half_height = scale[1]
                    local_pts = generate_capsule_particles(radius, half_height, dx)
                elif shape_type == GeoType.CYLINDER:
                    # Cylinder: scale.x = radius, scale.y = half_height [1](#15-0)
                    radius = scale[0]
                    half_height = scale[1]
                    local_pts = generate_cylinder_particles(radius, half_height, dx)
                elif shape_type == GeoType.CONE:
                    # Cone: scale.x = radius, scale.y = half_height [1](#15-0)
                    radius = scale[0]
                    half_height = scale[1]
                    local_pts = generate_cone_particles(radius, half_height, dx)
                elif shape_type == GeoType.ELLIPSOID:
                    # Ellipsoid: scale = (semi_axis_x, semi_axis_y, semi_axis_z) [1](#15-0)
                    rx, ry, rz = scale[:3]
                    local_pts = generate_ellipsoid_particles(rx, ry, rz, dx)
                else:
                    raise ValueError(
                        f"add_rigid_bodies: encountered unsupported shape_type: {shape_type}"
                    )

                if len(local_pts) > 0:
                    all_local_pts.append(local_pts)
                    all_shape_transforms.append(shape_xform)

            # Transform particles to world coordinates
            if all_local_pts:
                body_pos = body_q[:3]
                body_quat = wp.quat(body_q[3], body_q[4], body_q[5], body_q[6])

                world_pts = []
                for pts, shape_xform in zip(all_local_pts, all_shape_transforms):
                    pos_local = shape_xform[:3]
                    quat_local = shape_xform[3:]
                    shape_quat = wp.quat(
                        quat_local[0], quat_local[1], quat_local[2], quat_local[3]
                    )
                    shape_world_pos = body_pos + wp.quat_rotate(
                        body_quat, wp.vec3(pos_local)
                    )
                    world_quat = body_quat * shape_quat
                    rot_matrix = np.array(wp.quat_to_matrix(world_quat)).reshape((3, 3))
                    transformed_pts = shape_world_pos + (pts @ rot_matrix.T)
                    world_pts.extend(transformed_pts)

                # Add particles
                num_pts = len(world_pts)
                fluid_builder.add_particles(
                    pos=[wp.vec3(*pt) for pt in world_pts],
                    vel=[wp.vec3(0.0, 0.0, 0.0)] * num_pts,
                    mass=[0.0] * num_pts,  # kinematic
                    radius=[dx * 0.5] * num_pts,
                    custom_attributes={f"{WCSPH}:particle_body": [next_id] * num_pts},
                )
                next_id += 1

        return next_id

    def update_fluid_boundaries(self, rigid_state: State, fluid_state: State) -> None:
        wp.launch(
            update_rigid_particles,
            self.n,
            inputs=[
                self.particle_q0_local,
                self.particle_body,
                rigid_state.body_q,
                rigid_state.body_qd,
            ],
            outputs=[fluid_state.particle_q, fluid_state.particle_qd],
        )

    def apply_fluid_forces(self, fluid_state: State, rigid_state: State) -> None:
        wp.launch(
            compute_body_f,
            self.n,
            inputs=[
                fluid_state.particle_q,
                fluid_state.particle_f,
                self.particle_body,
                rigid_state.body_q,
                self.body_com,
            ],
            outputs=[rigid_state.body_f],
        )

    def interpolate_rigid_states(
        self, state0: State, state1: State, alpha: float, state2: State
    ) -> None:
        if state0.body_q:
            wp.launch(
                interpolate_rigid_states_kernel,
                state0.body_q.shape[0],
                inputs=[
                    state0.body_q,
                    state1.body_q,
                    state0.body_qd,
                    state1.body_qd,
                    alpha,
                ],
                outputs=[state2.body_q, state2.body_qd],
            )

    def average_rigid_forces(self, state: State, count: float) -> None:
        if state.body_f:
            wp.launch(
                divide_force_kernel, len(state.body_f), inputs=[state.body_f, count]
            )
