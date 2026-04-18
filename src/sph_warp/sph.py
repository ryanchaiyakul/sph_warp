from dataclasses import dataclass
from typing import Sequence
import warp as wp
from newton.solvers import SolverBase
from newton import Model, State, Control, Contacts, ModelBuilder
from .constants import WCSPH
from .kernels import (
    compute_m_V,
    compute_rho,
    compute_p,
    compute_particle_f,
    advect,
)


@dataclass
class SPHConfig:
    # Analytical fluid parameters
    rho0: float = 1000.0
    c_s: float = 88.5
    alpha: float = 0.05
    # Block parameters
    dx: float = 0.03
    jitter: float = 0.01  # for fluid only


class SolverWCSPH(SolverBase):
    def __init__(self, model: Model):
        if not hasattr(model, WCSPH):
            raise AttributeError(
                "WCSPH custom attributes are missing from the model. "
                "Call SolverWCSPH.register_custom_attributes() before building the model. "
                "If you called the function above, your model does not contain particles. "
            )
        self.sph = getattr(model, WCSPH)

        # SPH values
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
        self.particle_body = self.sph.particle_body
        self.m_V = wp.clone(model.particle_mass)  # type: ignore

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
