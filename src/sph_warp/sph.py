from dataclasses import dataclass
from typing import Sequence
import numpy as np
import warp as wp
from newton.solvers import SolverBase
from newton import Model, State, Control, Contacts, ModelBuilder, GeoType
from .kernels import (
    compute_m_V,
    compute_rho,
    compute_p,
    compute_particle_f,
    advect,
    compute_particle_q0_local,
    update_rigid_particles,
    compute_body_f,
    interpolate_rigid_states_kernel,
    average_forces_kernel,
)
from .geometry import (
    generate_box_particles,
    generate_capsule_particles,
    generate_cone_particles,
    generate_cylinder_particles,
    generate_ellipsoid_particles,
    generate_sphere_particles,
)
from .constants import WCSPH


@dataclass
class SPHConfig:
    # Analytical fluid parameters
    rho0: float = 1000.0
    c_s: float = 88.5
    alpha: float = 0.05
    # Fluid block parameters
    dx: float = 0.03
    jitter: float = 0.01


class SolverWCSPH(SolverBase):
    def __init__(self, model: Model, rigid_model: Model | None = None):
        if not hasattr(model, WCSPH):
            raise AttributeError(
                "WCSPH custom attributes are missing from the model. "
                "Call SolverWCSPH.register_custom_attributes() before building the model. "
                "If you called the function above, your model does not contain particles. "
            )
        self.sph = getattr(model, WCSPH)

        self.h = self.sph.h.numpy()[0]
        self.rho0 = self.sph.rho0.numpy()[0]
        self.c_s = self.sph.c_s.numpy()[0]
        self.stiffness = self.rho0 * self.c_s**2.0 / 7.0  # from WCSPH

        self.alpha = self.sph.alpha.numpy()[0]
        self.gravity = model.gravity.numpy()[0]  # type: ignore

        # HashGrid params
        self.grid_res = self.sph.grid_res.numpy()[0]
        self.cell_size = self.h * 2.0
        self.grid = wp.HashGrid(
            self.grid_res,
            self.grid_res,
            self.grid_res,
            device=model.device,
        )

        # Construct internal arrays
        self.n = model.particle_count
        self.particle_rho = wp.zeros(shape=self.n, dtype=float, device=model.device)
        self.particle_p = wp.zeros(shape=self.n, dtype=float, device=model.device)
        self.particle_q0_local = wp.zeros(self.n, dtype=wp.vec3, device=model.device)
        self.particle_body = self.sph.particle_body
        self.m_V = wp.clone(model.particle_mass)  # type: ignore
        self.body_com = self.sph.body_com

        if rigid_model:
            wp.launch(
                compute_particle_q0_local,
                self.n,
                inputs=[
                    model.particle_q,
                    self.particle_body,
                    rigid_model.body_q,
                ],
                outputs=[self.particle_q0_local],
            )

    @classmethod
    def register_custom_attributes(
        cls, builder: ModelBuilder, config: SPHConfig = SPHConfig()
    ) -> None:
        # -1 = fluid, 0+ = body index
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="particle_body",
                frequency=Model.AttributeFrequency.PARTICLE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=int,
                namespace=WCSPH,
                default=-1,
            )
        )
        # CoM of rigid bodies
        builder.add_custom_frequency(ModelBuilder.CustomFrequency("body"))
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_com",
                frequency="body",
                assignment=Model.AttributeAssignment.MODEL,
                dtype=wp.vec3,
                namespace=WCSPH,
                default=wp.vec3(0.0),
            )
        )
        # Defaults fluid parameters for water
        # Particle radius
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="h",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=float,
                namespace=WCSPH,
                default=0.1,
            )
        )
        # Initial density
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="rho0",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=float,
                namespace=WCSPH,
                default=config.rho0,
            )
        )
        # Speed of sound
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="c_s",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=float,
                namespace=WCSPH,
                default=config.c_s,
            )
        )
        # Viscosity constant [0.08, 0.5]
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="alpha",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=float,
                namespace=WCSPH,
                default=config.alpha,
            )
        )
        # Grid resolution
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="grid_res",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=int,
                namespace=WCSPH,
                default=128,
            )
        )

    @staticmethod
    def add_fluid_block(
        fluid_builder: ModelBuilder,
        dim: Sequence[float | int],
        origin: wp.vec3 = wp.vec3(0.0, 0.0, 0.0),
        config: SPHConfig = SPHConfig(),
    ) -> None:
        """
        Add a rectangular block of fluid particles to fluid_builder.
        """
        fluid_builder.add_particle_grid(
            pos=origin,
            rot=wp.quat_identity(),  # type: ignore
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(dim[0] / config.dx),
            dim_y=int(dim[1] / config.dx),
            dim_z=int(dim[2] / config.dx),
            cell_x=config.dx,
            cell_y=config.dx,
            cell_z=config.dx,
            mass=config.rho0 * config.dx**3,  # m = ρV
            radius_mean=config.dx * 0.5,
            jitter=config.dx * config.jitter,
        )

    @staticmethod
    def add_rigid_bodies(
        fluid_builder: ModelBuilder,
        rigid_model: Model,
        config: SPHConfig = SPHConfig(),
        next_id: int = 0,
    ) -> int:
        # Alias shape arrays
        shape_types = rigid_model.shape_type.numpy()
        shape_bodies = rigid_model.shape_body.numpy()
        shape_transforms = rigid_model.shape_transform.numpy()
        shape_scales = rigid_model.shape_scale.numpy()

        dx = config.dx

        # For each body
        for body_idx, (com, body_q) in enumerate(
            zip(rigid_model.body_com.numpy(), rigid_model.body_q.numpy())
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

            # Transform and add all particles for this body
            if all_local_pts:
                # Get body world transform
                body_pos = body_q[:3]
                body_quat = wp.quat(body_q[3], body_q[4], body_q[5], body_q[6])

                world_pts = []
                for pts, shape_xform in zip(all_local_pts, all_shape_transforms):
                    pos_local = shape_xform[:3]
                    quat_local = shape_xform[3:]

                    # Compose transforms: world_transform = body_transform * shape_transform
                    shape_quat = wp.quat(
                        quat_local[0], quat_local[1], quat_local[2], quat_local[3]
                    )

                    # Transform shape position to world space
                    shape_world_pos = body_pos + wp.quat_rotate(
                        body_quat, wp.vec3(pos_local)
                    )

                    # Compose rotations
                    world_quat = body_quat * shape_quat

                    # Apply world transform to points
                    rot_matrix = np.array(wp.quat_to_matrix(world_quat)).reshape((3, 3))
                    transformed_pts = shape_world_pos + (pts @ rot_matrix.T)
                    world_pts.extend(transformed_pts)

                # Add particles in bulk
                num_pts = len(world_pts)
                fluid_builder.add_particles(
                    pos=[wp.vec3(*pt) for pt in world_pts],
                    vel=[wp.vec3(0.0, 0.0, 0.0)] * num_pts,
                    mass=[0.0] * num_pts,  # 0.0 for kinematic bodies
                    radius=[dx * 0.5] * num_pts,
                    custom_attributes={f"{WCSPH}:particle_body": [next_id] * num_pts},
                )
                next_id += 1

        return next_id

    def update_rigid_particles(self, rigid_state: State, fluid_state: State) -> None:
        """Updates rigid `particle_q` and `particle_qd` in `fluid_state`.

        Args:
            rigid_state (State): RigidSolver State object.
            fluid_state (State): WCSPH State object.
        """
        wp.launch(
            update_rigid_particles,
            self.n,
            inputs=[
                self.particle_q0_local,
                self.particle_body,
                rigid_state.body_q,
                rigid_state.body_qd,
            ],
            outputs=[
                fluid_state.particle_q,
                fluid_state.particle_qd,
            ],
        )

    def accumulate_rigid_forces(self, rigid_state: State, fluid_state: State) -> None:
        """Accumulates rigid forces and torques in `body_qdd` in `rigid_state`.

        Args:
            rigid_state (State): RigidSolver State object.
            fluid_state (State): WCSPH State object.
        """
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
            outputs=[
                rigid_state.body_f,
            ],
        )

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        self.grid.build(state_in.particle_q, self.cell_size)
        wp.launch(
            kernel=compute_m_V,
            dim=self.n,
            inputs=[
                self.grid.id,
                state_in.particle_q,
                self.particle_body,
                self.h,
                self.rho0,
            ],
            outputs=[self.m_V],
        )
        wp.launch(
            kernel=compute_rho,
            dim=self.n,
            inputs=[
                self.grid.id,
                state_in.particle_q,
                self.particle_body,
                self.m_V,
                self.h,
            ],
            outputs=[self.particle_rho],
        )

        wp.launch(
            kernel=compute_p,
            dim=self.n,
            inputs=[
                self.particle_rho,
                self.particle_body,
                self.rho0,
                self.stiffness,
            ],
            outputs=[
                self.particle_p,
            ],
        )

        wp.launch(
            kernel=compute_particle_f,
            dim=self.n,
            inputs=[
                self.grid.id,
                state_in.particle_q,
                state_in.particle_qd,
                self.particle_body,
                self.m_V,
                self.particle_rho,
                self.particle_p,
                self.h,
                self.rho0,
                self.c_s,
                self.alpha,
                self.gravity,
            ],
            outputs=[state_in.particle_f],
        )

        wp.launch(
            kernel=advect,
            dim=self.n,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                self.particle_body,
                self.m_V,
                dt,
            ],
            outputs=[
                state_out.particle_q,
                state_out.particle_qd,
            ],
        )

    # Helpers for rigid coupling
    @staticmethod
    def interpolate_rigid_states(
        state0: State, state1: State, alpha: float, state2: State
    ) -> None:
        if state0.body_q:
            wp.launch(
                kernel=interpolate_rigid_states_kernel,
                dim=state0.body_q.shape[0],
                inputs=[
                    state0.body_q,
                    state1.body_q,
                    state0.body_qd,
                    state1.body_qd,
                    alpha,
                ],
                outputs=[
                    state2.body_q,
                    state2.body_qd,
                ],
            )

    @staticmethod
    def average_rigid_forces(state: State, count: int | float) -> None:
        if state.body_f:
            wp.launch(
                kernel=average_forces_kernel,
                dim=len(state.body_f),
                inputs=[state.body_f, float(count)],
            )
