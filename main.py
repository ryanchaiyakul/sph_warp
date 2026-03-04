from newton.solvers import SolverBase
from newton import Model, State, Control, Contacts, ModelBuilder
import warp as wp

WCSPH = "wcsph"


@wp.func
def square(q: float):
    return q * q


@wp.func
def cube(q: float):
    return q * q * q


@wp.func
def cubic(r: wp.vec3, h: float):
    k = 8.0 / wp.pi / (h * h * h)
    q = wp.length(r) / h
    if q <= 1.0:
        if q <= 0.5:
            return k * (6.0 * cube(q) - 6.0 * square(q) + 1.0)
        return k * 2.0 * cube(1.0 - q)
    return 0.0


@wp.func
def diff_cubic(r: wp.vec3, h: float):
    k = 48.0 / wp.pi / (h * h * h)
    r_norm = wp.length(r)
    q = r_norm / h
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            return k * q * (3.0 * q - 2.0) * grad_q
        return -k * square(1.0 - q) * grad_q
    return wp.vec3(0.0)


@wp.kernel
def compute_density(
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
        distance = q_i - q[index]
        rho_sum += mass[index] * cubic(distance, h)
    rho[i] = rho_sum


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
    # symplectic euler
    tid = wp.tid()
    qd_out[tid] = qd_in[tid] + (f_in[tid] / m[tid]) * dt
    q_out[tid] = q_in[tid] + qd_out[tid] * dt


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
        self.density0 = float(self.sph.density0.numpy()[0])

        self.cell_size = self.h * 2.0
        self.grid = wp.HashGrid(
            self.grid_res,
            self.grid_res,
            self.grid_res,
            device=model.device,
        )

        self.n = model.particle_count
        self.particle_mass = model.particle_mass
        self.particle_rho = wp.zeros(shape=self.n, dtype=float, device=model.device)

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        # Smoothing length
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="h",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=float,
                namespace=WCSPH,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="density0",
                frequency=Model.AttributeFrequency.ONCE,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=float,
                namespace=WCSPH,
                default=1000.0,
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

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> State | None:
        self.grid.build(state_in.particle_q, self.cell_size)
        # compute density
        wp.launch(
            kernel=compute_density,
            dim=self.n,
            inputs=[
                self.grid.id,
                state_in.particle_q,
                self.particle_mass,
                self.h,
            ],
            outputs=[self.particle_rho],
        )
        print(self.particle_rho)
        # compute non-pressure forces
        # compute pressure forces

        # --- VALIDATION LOGIC ---
        # Bring data back to CPU for a quick check
        rho_host = self.particle_rho.numpy()
        avg_rho = np.mean(rho_host)
        max_rho = np.max(rho_host)
        min_rho = np.min(rho_host)

        print(f"--- Density Test ---")
        print(f"Target Rho0: {self.density0}")
        print(f"Average Rho: {avg_rho:.2f} ({(avg_rho / self.density0) * 100:.1f}%)")
        print(f"Max Rho:     {max_rho:.2f}")
        print(f"Min Rho:     {min_rho:.2f} (Surface particles will be lower)")

        # advect
        wp.launch(
            kernel=advect,
            dim=self.n,
            inputs=[
                state_in.particle_q,
                state_in.particle_qd,
                state_in.particle_f,
                self.particle_mass,
                dt,
            ],
            outputs=[
                state_out.particle_q,
                state_out.particle_qd,
            ],
        )
        return state_out


import numpy as np
import warp as wp

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
    jitter=2.0 * radius,
    radius_mean=radius,
    # custom_attributes={"mpm:friction": 0.75},
)
sand_model = sand_builder.finalize()
solver = SolverWCSPH(sand_model)

state0 = sand_model.state()
state1 = sand_model.state()
solver.step(state0, state1, None, None, 1e-3)
