import warp as wp
from newton import ModelBuilder, State, Model
from newton.viewer import ViewerGL
from .constants import WCSPH


class CoupledRender:
    def __init__(
        self,
        rigid_builder: ModelBuilder,
        fluid_builder: ModelBuilder,
        fluid_model: Model,
    ):
        # Create combined model for viewer
        combined_builder = ModelBuilder()
        combined_builder.add_world(rigid_builder)
        combined_builder.add_world(fluid_builder)

        self.combined_model = combined_builder.finalize()
        self.combined_state = self.combined_model.state()

        self.viewer = ViewerGL()
        self.viewer.set_model(self.combined_model)
        self.viewer.set_world_offsets((0.0, 0.0, 0.0))
        self.viewer.show_particles = True

        # Pre-compute fluid particle mask (excluding rigid boundary particles)
        particle_body = getattr(fluid_model, WCSPH).particle_body.numpy()
        self.fluid_mask = particle_body == -1

        # Pre-filter static visual properties
        self.combined_model.particle_count = int(sum(self.fluid_mask))
        self.combined_model.particle_radius = wp.array(
            fluid_model.particle_radius.numpy()[self.fluid_mask],
            dtype=fluid_model.particle_radius.dtype,
            device=fluid_model.particle_radius.device,
        )

    def render_frame(self, time: float, rigid_state: State, fluid_state: State):
        # Update World 0 (Rigid bodies)
        self.combined_state.body_q = rigid_state.body_q
        self.combined_state.body_qd = rigid_state.body_qd

        # Update World 1 (Fluid particles, masked)
        self.combined_state.particle_q = wp.array(
            fluid_state.particle_q.numpy()[self.fluid_mask],
            dtype=fluid_state.particle_q.dtype,
            device=fluid_state.particle_q.device,
        )
        self.combined_state.particle_qd = wp.array(
            fluid_state.particle_qd.numpy()[self.fluid_mask],
            dtype=fluid_state.particle_qd.dtype,
            device=fluid_state.particle_qd.device,
        )

        self.viewer.begin_frame(time)
        self.viewer.log_state(self.combined_state)
        self.viewer.end_frame()

    def close(self):
        self.viewer.close()
