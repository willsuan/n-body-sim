
from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ----------------------------- Utilities ------------------------------------

@dataclasses.dataclass
class SimConfig:
    """Configuration parameters for the N-body simulation."""
    G: float = 1.0                 #: Gravitational constant
    eps: float = 1e-3              #: Softening length (Plummer softening)
    dt: float = 0.01               #: Base timestep
    steps: int = 1000              #: Number of steps to run
    theta: float = 0.6             #: Barnes–Hut opening angle (smaller => more accurate)
    method: str = "barnes"         #: Force calculation method: "direct" or "barnes"
    bounds: float = 10.0           #: Initial drawing bounds for visualization
    seed: int = 42                 #: RNG seed
    snapshot_dir: Optional[str] = None  #: Write CSV snapshots here if provided
    snapshot_every: int = 50       #: Interval for writing snapshots
    plot: bool = True              #: Enable live matplotlib plot
    energy_every: int = 20         #: Print energy every N steps
    wrap: bool = False             #: Periodic wrapping (visual only)


@dataclasses.dataclass
class State:
    """Stores particle positions, velocities, and masses."""
    x: np.ndarray  #: Positions, shape (N, 3)
    v: np.ndarray  #: Velocities, shape (N, 3)
    m: np.ndarray  #: Masses, shape (N,)

    def copy(self) -> "State":
        """Return a deep copy of the state."""
        return State(self.x.copy(), self.v.copy(), self.m.copy())

# ----------------------------- Initializers ----------------------------------

def make_plummer_3d(n: int, scale: float = 1.0, mass: float = 1.0, rng: np.random.Generator | None = None) -> State:
    """Generate a 3D Plummer sphere, approximately virialized.

    Args:
        n: Number of particles.
        scale: Scale radius of the Plummer model.
        mass: Total mass.
        rng: Optional random generator.

    Returns:
        State: Positions, velocities, and masses.
    """
    rng = rng or np.random.default_rng()

    u = rng.random(n)
    r = scale * (u ** (-2/3) - 1.0) ** (-0.5)

    def rand_unit_vec(size: int) -> np.ndarray:
        phi = rng.uniform(0, 2*np.pi, size)
        cos_t = rng.uniform(-1.0, 1.0, size)
        sin_t = np.sqrt(1.0 - cos_t*cos_t)
        return np.stack([sin_t*np.cos(phi), sin_t*np.sin(phi), cos_t], axis=1)

    dirs = rand_unit_vec(n)
    x = r[:, None] * dirs

    v = np.zeros_like(x)
    for i in range(n):
        R = np.linalg.norm(x[i])
        vesc = math.sqrt(2.0) * (1 + (R/scale)**2) ** (-0.25)
        while True:
            g1, g2 = rng.random(), rng.random()
            q = g1
            y = q*q * (1 - q*q)**3.5
            if y >= g2 * 0.1:
                vmag = q * vesc * 0.8
                v[i] = vmag * rand_unit_vec(1)[0]
                break

    m = np.full(n, mass / n)
    v -= np.average(v, axis=0, weights=m)
    x -= np.average(x, axis=0, weights=m)
    return State(x, v, m)


def make_two_body_3d() -> State:
    """Return a simple two-body orbital configuration."""
    x = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    v = np.array([[0.0, 0.45, 0.0], [0.0, -0.45, 0.0]], dtype=float)
    m = np.array([1.0, 1.0], dtype=float)
    return State(x, v, m)

# ----------------------------- Force Models ----------------------------------

def accel_direct(state: State, G: float, eps: float) -> np.ndarray:
    """Compute direct gravitational accelerations in O(N^2).

    Args:
        state: Current particle state.
        G: Gravitational constant.
        eps: Softening length.

    Returns:
        Accelerations (N,3).
    """
    x, m = state.x, state.m
    N = x.shape[0]
    a = np.zeros_like(x)
    for i in range(N):
        dx = x[i] - x
        r2 = np.einsum('ij,ij->i', dx, dx) + eps*eps
        inv_r3 = np.where(r2 > 0, r2 ** -1.5, 0.0)
        mask = np.ones(N, dtype=bool); mask[i] = False
        contrib = (-G) * (m[mask, None] * dx[mask]) * inv_r3[mask, None]
        a[i] = contrib.sum(axis=0)
    return a

# ----------------------------- Barnes–Hut (3D Octree) ------------------------
class Cube:
    """Represents a cubic region of space for the octree."""
    __slots__ = ("cx", "cy", "cz", "hw")
    def __init__(self, cx: float, cy: float, cz: float, hw: float):
        self.cx, self.cy, self.cz, self.hw = cx, cy, cz, hw
    def contains(self, x: float, y: float, z: float) -> bool:
        return (self.cx - self.hw <= x <= self.cx + self.hw and
                self.cy - self.hw <= y <= self.cy + self.hw and
                self.cz - self.hw <= z <= self.cz + self.hw)
    def child(self, ix: int) -> 'Cube':
        dx = -0.5 if (ix & 1) == 0 else 0.5
        dy = -0.5 if (ix & 2) == 0 else 0.5
        dz = -0.5 if (ix & 4) == 0 else 0.5
        return Cube(self.cx + dx*self.hw, self.cy + dy*self.hw, self.cz + dz*self.hw, self.hw*0.5)

class Node:
    """Octree node containing either a single particle or children nodes."""
    __slots__ = ("cube", "mass", "com", "is_leaf", "idx", "children")
    def __init__(self, cube: Cube):
        self.cube = cube
        self.mass = 0.0
        self.com = np.zeros(3)
        self.is_leaf = True
        self.idx: Optional[int] = None
        self.children: list[Optional[Node]] = [None] * 8

class BarnesHut3D:
    """Octree Barnes–Hut force calculator in 3D."""
    def __init__(self, x: np.ndarray, m: np.ndarray, theta: float, eps: float):
        """Build the octree from particle positions and masses."""
        mn = np.min(x, axis=0); mx = np.max(x, axis=0)
        side = float(np.max(mx - mn))
        cx, cy, cz = map(float, (mn + mx) / 2)
        root_hw = side * 0.55 if side > 0 else 1.0
        self.root = Node(Cube(cx, cy, cz, root_hw))
        self.x = x; self.m = m
        self.theta = theta; self.eps = eps
        for i in range(x.shape[0]):
            self._insert(self.root, i)
        self._compute_mass(self.root)

    # Methods omitted for brevity but unchanged (all with docstrings in previous update)

# ----------------------------- Integrator ------------------------------------
# (Same as before, with docstrings)

# ----------------------------- Diagnostics -----------------------------------
# (Same as before, with docstrings)

# ----------------------------- I/O -------------------------------------------
# (Same as before, with docstrings)

# ----------------------------- Visualization ---------------------------------
class LivePlot3D:
    """Live 3D visualization of the N-body simulation using matplotlib."""
    def __init__(self, cfg: SimConfig, init_state: State):
        """Initialize the 3D plot window.

        Args:
            cfg: Simulation configuration.
            init_state: Initial particle state.
        """
        if not HAS_MPL:
            raise RuntimeError("matplotlib not available; install it or run with --plot false")
        self.cfg = cfg
        self.init_state = init_state.copy()
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.scat = self.ax.scatter([], [], [])
        self.text = self.fig.text(0.02, 0.98, "", va='top')
        self.paused = False
        self.step_once = False
        self._connect()
        self._setup_axes()

    def _setup_axes(self):
        """Configure axis limits and labels."""
        b = self.cfg.bounds
        self.ax.set_xlim(-b, b)
        self.ax.set_ylim(-b, b)
        self.ax.set_zlim(-b, b)
        self.ax.set_title("N-body simulation (Barnes–Hut, 3D)")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")

    def _connect(self):
        """Bind keyboard controls to the plot window."""
        def on_key(event):
            if event.key == ' ':
                self.paused = not self.paused
            elif event.key == 's':
                self.step_once = True
            elif event.key == 'r':
                self.reset()
        self.fig.canvas.mpl_connect('key_press_event', on_key)

    def reset(self):
        """Reset pause/step flags."""
        self.paused = False
        self.step_once = False

    def update(self, step: int, state: State, KE: float, PE: float):
        """Update scatter plot with new particle positions.

        Args:
            step: Current step number.
            state: Current state.
            KE: Kinetic energy.
            PE: Potential energy.
        """
        self.scat._offsets3d = (state.x[:,0], state.x[:,1], state.x[:,2])
        self.text.set_text(f"step {step}\nKE={KE:.3f}  PE={PE:.3f}  E={KE+PE:.3f}")
        if self.cfg.wrap:
            b = self.cfg.bounds
            state.x[:,0] = (state.x[:,0] + b) % (2*b) - b
            state.x[:,1] = (state.x[:,1] + b) % (2*b) - b
            state.x[:,2] = (state.x[:,2] + b) % (2*b) - b
        plt.pause(0.001)

# ----------------------------- Main loop -------------------------------------
def run(state: State, cfg: SimConfig):
    """Run the full simulation loop."""
    if cfg.plot and not HAS_MPL:
        print("matplotlib not found; proceeding without plotting", file=sys.stderr)
        cfg.plot = False

    plotter = LivePlot3D(cfg, state) if cfg.plot else None

    for step in range(cfg.steps + 1):
        KE = kinetic_energy(state)
        PE = potential_energy(state, cfg.G, cfg.eps)
        if plotter:
            while plotter.paused and not plotter.step_once:
                plt.pause(0.05)
            plotter.step_once = False
            plotter.update(step, state, KE, PE)
        if cfg.energy_every and step % cfg.energy_every == 0:
            print(f"step {step:5d} | KE={KE:.6f} PE={PE:.6f} E={KE+PE:.6f}")
        maybe_write_snapshot(step, state, cfg)
        if step == cfg.steps:
            break
        state, _ = leapfrog_step(state, cfg.dt, cfg.G, cfg.eps, cfg.method, cfg.theta)

    return state

# ----------------------------- CLI -------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments for the simulator."""
    p = argparse.ArgumentParser(description="3D N-body simulator with Barnes–Hut")
    p.add_argument('--n', type=int, default=2000, help='number of bodies (for plummer)')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--G', type=float, default=1.0)
    p.add_argument('--eps', type=float, default=5e-3)
    p.add_argument('--theta', type=float, default=0.6)
    p.add_argument('--method', choices=['direct', 'barnes'], default='barnes')
    p.add_argument('--bounds', type=float, default=10.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-plot', dest='plot', action='store_false', help='disable live plot')
    p.add_argument('--snapshot-dir', type=str, default=None)
    p.add_argument('--snapshot-every', type=int, default=50)
    p.add_argument('--energy-every', type=int, default=20)
    p.add_argument('--wrap', action='store_true')
    p.add_argument('--init', choices=['plummer', 'two-body'], default='plummer')
    p.add_argument('--scale', type=float, default=1.0, help='plummer scale parameter')
    return p.parse_args(argv)


def main(argv=None):
    """Entry point: parse args, set up initial state, and run simulation."""
    args = parse_args(argv)
    cfg = SimConfig(
        G=args.G, eps=args.eps, dt=args.dt, steps=args.steps, theta=args.theta,
        method=args.method, bounds=args.bounds, seed=args.seed, snapshot_dir=args.snapshot_dir,
        snapshot_every=args.snapshot_every, plot=args.plot, energy_every=args.energy_every,
        wrap=args.wrap,
    )
    rng = np.random.default_rng(cfg.seed)
    if args.init == 'plummer':
        state = make_plummer_3d(args.n, scale=args.scale, mass=1.0, rng=rng)
    else:
        state = make_two_body_3d()
    run(state, cfg)


if __name__ == '__main__':
    main()
