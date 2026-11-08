"""A flock of birds (or a school of fish) from three local rules.

Each animal only reacts to the animals it can see near it, and follows three
simple rules, in three zones around it (Couzin et al. 2002, "Collective memory
and spatial sorting in animal groups", J. Theor. Biol. 218):
  1. alignment:  in the middle zone, match the direction of the neighbours
  2. separation: in the inner zone, turn away from them; this overrides 1 and 3
  3. cohesion:   in the outer zone, turn toward them
Like real animals they also have a blind zone behind them, turn at a limited
rate, keep a nearly constant speed and move with some random wobble. With the
"birds" preset the flock flies in parallel and keeps changing shape; with the
"fish" preset the school starts milling in a rotating ring, as real schools do.

They start at random positions inside a circle with random directions. The
number of animals is a parameter (BirdSimulation.n_birds) and the animation
lasts 10 seconds.

Run:      python bird_sim.py   (see the __main__ block at the bottom)
Preview:  python C:/Users/amohamma/.claude/skills/manim_skill/scripts/preview.py bird_sim.py BirdSimulation
Final:    manim -qh bird_sim.py BirdSimulation

Other settings without editing the class:
    class FishSchool(BirdSimulation):
        species = "fish"
        n_birds = 300
"""
from manim import *
import numpy as np

Text.set_default(font="Cambria")

# Behaviour presets. Zone radii and turning radius are in body lengths (BL), so they
# hold for any flock size; speed is in screen units per second. Tuned by simulating
# 500 animals from the circle start and measuring the group (see the notes below).
SPECIES = {
    # Flock: a wide, strongly weighted alignment zone. One group flying in parallel
    # (polarization ~0.8) that turns and changes shape; neighbours ~1.2 BL apart.
    "birds": dict(repulsion=1, orientation=8, attraction=20.0, alignment=5.0, blind_angle=60.0,
                  turn_radius=6.0, noise=0.3, speed=1.8),
    # School: a narrow alignment zone, wide attraction and a large blind zone behind.
    # The school mills in a rotating ring (milling ~0.7); neighbours ~0.8 BL apart.
    "fish": dict(repulsion=1.0, orientation=2.0, attraction=25.0, blind_angle=120.0,
                 turn_radius=4.0, noise=0.2, speed=1.5),
}


def unit_rows(v):
    """Rows of v scaled to length 1; zero rows stay zero."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 1e-12)


class Flock(VGroup):
    """Animals that move by the three rules on their own, in a dt-updater.

    Positions and headings live in numpy arrays (one row per animal) and the
    shapes are redrawn from them. The physics advances in fixed steps of STEP
    seconds with a seeded random generator, so a -ql preview and a -qh render
    show the same motion.
    """
    STEP = 1 / 120
    SHAPE = np.array([[1, 0], [-0.7, 0.6], [-0.35, 0], [-0.7, -0.6], [1, 0]])  # a dart pointing along +x

    def __init__(
        self,
        n_birds=40,
        seed=0,
        box=(-6, 6, -4, 4),           # airspace: x_min, x_max, y_min, y_max
        start_radius=2.0,             # animals start inside a circle of this radius at the frame centre
        repulsion=1.0,                # rule 2 zone: closer than this many BL, turn away
        orientation=5.0,              # rule 1 zone: up to this many BL, match their direction
        attraction=15.0,              # rule 3 zone: up to this many BL, turn toward them
        alignment=1.0,                # weight of rule 1 against rule 3
        cohesion=1.0,                 # weight of rule 3 against rule 1
        blind_angle=60.0,             # degrees behind the animal that it cannot see
        turn_radius=6.0,              # tightest turn, in BL
        noise=0.6,                    # random wobble of the heading, rad per sqrt(second)
        speed=2.0,                    # screen units per second, the same for every animal
        wall_margin=2,              # animals start turning this far inside the box edge
        size=0.17,
        tuned_for=40.0,               # flocks bigger than this get smaller animals
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Bigger flocks get smaller animals: the typical gap between them goes like
        # 1/sqrt(n_birds), so shrink the body by that factor. All distances are in
        # body lengths, so they shrink with it and the picture stays as tidy.
        size *= min(1.0, np.sqrt(tuned_for / n_birds))
        self.body_length = (self.SHAPE[:, 0].max() - self.SHAPE[:, 0].min()) * size
        self.zones = np.array([repulsion, orientation, attraction]) * self.body_length
        self.w_align, self.w_coh = alignment, cohesion
        self.cos_blind = -np.cos(np.radians(blind_angle) / 2)  # a neighbour is visible if cos(angle) > this
        self.speed, self.noise = speed, noise
        self.max_turn = speed / (turn_radius * self.body_length)  # rad/s
        self.box = np.array(box, dtype=float)
        self.wall_margin = max(wall_margin, 1.5 * turn_radius * self.body_length)  # room to turn away in time

        # Random start: positions spread evenly over the start circle, headings in any direction
        self.rng = np.random.default_rng(seed)
        r = start_radius * np.sqrt(self.rng.uniform(0, 1, n_birds))  # sqrt keeps the density even out to the rim
        theta = self.rng.uniform(0, TAU, n_birds)
        self.pos = np.c_[r * np.cos(theta), r * np.sin(theta)]
        heading = self.rng.uniform(0, TAU, n_birds)
        self.heading = np.c_[np.cos(heading), np.sin(heading)]  # unit vectors

        self.sizes = np.full(n_birds, size)
        self.birds = VGroup(*(VMobject(stroke_width=0, fill_color=BLUE, fill_opacity=1) for _ in range(n_birds)))
        self.add(self.birds)
        self.running = False
        self.backlog = 0.0
        self.redraw()
        self.add_updater(lambda m, dt: m.advance(dt))

    @property
    def vel(self):
        return self.speed * self.heading

    # Group measures of Couzin et al. (2002). Their v_i is the unit direction of
    # animal i, which is self.heading here.
    def centre(self):
        """c_group(t) = (1/N) sum_i c_i(t), eqn (6)."""
        return self.pos.mean(axis=0)

    def direction(self):
        """d_group(t) = (1/N) sum_i v_i(t), eqn (7). Its length is the group polarization p_group, eqn (4)."""
        return self.heading.mean(axis=0)

    def pairs(self):
        diff = self.pos[None, :, :] - self.pos[:, None, :]  # diff[i, j] = pos[j] - pos[i]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)  # an animal is not its own neighbour
        return diff, dist

    def step(self, h):
        pos, u = self.pos, self.heading
        diff, dist = self.pairs()
        towards = diff / dist[..., None]  # unit vector from i to j
        visible = np.einsum("ik,ijk->ij", u, towards) > self.cos_blind
        r_rep, r_ori, r_att = self.zones
        repel = visible & (dist < r_rep)
        orient = visible & (dist >= r_rep) & (dist < r_ori)
        attract = visible & (dist >= r_ori) & (dist < r_att)

        # 1. alignment: the average direction of the neighbours in the middle zone
        # 3. cohesion: the direction toward the neighbours in the outer zone
        desired = self.w_align * unit_rows(orient @ u) + self.w_coh * unit_rows(np.einsum("ij,ijk->ik", attract, towards))
        # 2. separation: away from anyone too close, and nothing else matters then
        too_close = repel.any(axis=1)
        desired[too_close] = -np.einsum("ij,ijk->ik", repel, towards)[too_close]

        # Turn away from the edge of the box, more urgently the closer it is
        lo, hi = self.box[[0, 2]] + self.wall_margin, self.box[[1, 3]] - self.wall_margin
        wall = (np.clip(lo - pos, 0, None) - np.clip(pos - hi, 0, None)) / self.wall_margin
        desired = unit_rows(desired) + 3 * wall
        lost = np.linalg.norm(desired, axis=1) < 1e-9  # nobody in sight and no wall: keep going
        desired[lost] = u[lost]

        # Turn toward the desired direction at a limited rate, plus a random wobble
        turn = np.arctan2(u[:, 0] * desired[:, 1] - u[:, 1] * desired[:, 0], (u * desired).sum(axis=1))
        turn = np.clip(turn, -self.max_turn * h, self.max_turn * h)
        turn += self.noise * np.sqrt(h) * self.rng.standard_normal(len(u))
        c, s = np.cos(turn), np.sin(turn)
        u = np.c_[c * u[:, 0] - s * u[:, 1], s * u[:, 0] + c * u[:, 1]]

        pos = pos + self.speed * u * h
        # Hard edge, a last resort: bounce off it
        edge_lo, edge_hi = self.box[[0, 2]], self.box[[1, 3]]
        out = (pos < edge_lo) | (pos > edge_hi)
        u[out] *= -1
        self.pos, self.heading = np.clip(pos, edge_lo, edge_hi), u

    def advance(self, dt):
        if not self.running:
            return
        self.backlog += dt
        while self.backlog >= self.STEP - 1e-9:
            self.step(self.STEP)
            self.backlog -= self.STEP
        self.redraw()

    def redraw(self):
        c, s = self.heading[:, 0], self.heading[:, 1]
        rot = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)  # one 2x2 rotation per animal
        corners = self.pos[:, None, :] + self.sizes[:, None, None] * np.einsum("nij,kj->nki", rot, self.SHAPE)
        for bird, pts in zip(self.birds, corners):
            bird.set_points_as_corners(np.c_[pts, np.zeros(len(pts))])


class BirdSimulation(Scene):
    species = "birds"  # "birds" or "fish": see SPECIES at the top
    n_birds = 1000   # the simulation parameter: number of animals
    seed = 2       # another seed gives another random start
    start_radius = 5.0  # animals start at random inside a circle of this radius at the frame centre
                        # (keep it inside the box of Flock, which reaches 3 above and below the centre)
    total_time = 10  # seconds, the whole animation (not "duration": Scene uses that name)
    intro = 1.0      # of which the animals wait this long at their random start
    arrow_scale = 3.0  # screen length of the d_group arrow when the group is fully aligned (|d_group| = 1)
    d_color = YELLOW   # colour of the d_group arrow and its label

    def construct(self):
        flock = Flock(self.n_birds, seed=self.seed, start_radius=self.start_radius, **SPECIES[self.species])
        font_size = 20
        # What the viewer is looking at
        title = Text(f"{self.n_birds} {self.species}", font_size=font_size).to_edge(UP, buff=0.25)

        def d_vector():
            return self.arrow_scale * np.r_[flock.direction(), 0]

        def d_arrow():
            start = np.r_[flock.centre(), 0]
            if np.linalg.norm(d_vector()) < 0.1:  # too short to draw an arrow
                return VectorizedPoint(start)
            return Arrow(start, start + d_vector(), buff=0, color=self.d_color, stroke_width=6,
                         max_tip_length_to_length_ratio=0.25).set_z_index(1)

        def away_from_centre(sign):  # a unit offset along +d_group (sign=1) or -d_group (sign=-1)
            d = d_vector()
            return sign * d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-6 else DOWN

        arrow = always_redraw(d_arrow)
        centre = always_redraw(lambda: Dot(np.r_[flock.centre(), 0], radius=0.08, color=WHITE).set_z_index(1))
        d_label = MathTex(r"\mathbf{d}_{group}", color=self.d_color, font_size=20)
        d_label.add_updater(lambda m: m.move_to(arrow.get_end() + 0.45 * away_from_centre(1)))
        c_label = MathTex(r"\mathbf{c}_{group}", font_size=20)
        c_label.add_updater(lambda m: m.move_to(centre.get_center() + 0.45 * away_from_centre(-1)))
        overlay = VGroup(title, arrow, centre, d_label, c_label)
        for mob in (title, d_label, c_label):  # dark outline so text reads when animals pass behind it
            mob.set_stroke(BLACK, 5, background=True)
        for mob in overlay:
            mob.set_z_index(1)

        # Random start: random positions, random directions
        self.play(FadeIn(flock), *map(FadeIn, overlay), run_time=self.intro)

        # Let them move
        flock.running = True
        self.wait(self.total_time - self.intro)


if __name__ == "__main__":
    with tempconfig({"quality": "medium_quality", "preview": False}):
        BirdSimulation().render()
