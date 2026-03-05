from newton.solvers import SolverBase
from newton import Model, State, Control, Contacts, ModelBuilder
import warp as wp

from .kernels import compute_rho, compute_p, compute_particle_f, advect

WCSPH = "wcsph"


class SolverWCSPH(SolverBase):
    def __init__(self, model: Model):
        self.model = model
        if not hasattr(model, WCSPH):
            raise AttributeError(
                "WCSPH custom attributes are missing from the model. "
                "Call SolverWCSPH.register_custom_attributes() before building the model. "
                "If you called the function above, your model does not contain particles. "
            )
        self.sph = getattr(model, WCSPH)
        self.grid_res = int(self.sph.grid_resolution.numpy()[0])
        self.h = float(self.sph.h.numpy()[0])
        self.rho0 = float(self.sph.rho0.numpy()[0])

        self.cell_size = self.h * 2.0
        self.grid = wp.HashGrid(
            self.grid_res,
            self.grid_res,
            self.grid_res,
            device=model.device,
        )

        self.n = model.particle_count
        self.particle_rho = wp.zeros(shape=self.n, dtype=float, device=model.device)
        self.particle_p = wp.zeros(shape=self.n, dtype=float, device=model.device)
        self.gravity = wp.vec3(0.0, 0.0, -9.81)
        self.rho0 = 1000.0
        self.stiffness = 1119e3
        self.c_s = 88.5
        self.alpha = 0.5

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="h",  # Smoothing length
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=float,
                namespace=WCSPH,
                default=0.1,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="rho0",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=float,
                namespace=WCSPH,
                default=2500.0,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="grid_resolution",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=int,
                namespace=WCSPH,
                default=128,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="stiffness",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=int,
                namespace=WCSPH,
                default=128,
            )
        )

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> State | None:
        self.grid.build(state_in.particle_q, self.cell_size)
        wp.launch(
            kernel=compute_rho,
            dim=self.n,
            inputs=[
                self.grid.id,
                state_in.particle_q,
                self.model.particle_mass,
                self.h,
            ],
            outputs=[self.particle_rho],
        )

        wp.launch(
            kernel=compute_p,
            dim=self.n,
            inputs=[
                self.particle_rho,
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
                self.model.particle_mass,
                self.particle_rho,
                self.particle_p,
                self.h,
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
                self.model.particle_mass,
                dt,
            ],
            outputs=[
                state_out.particle_q,
                state_out.particle_qd,
            ],
        )
        return state_out
