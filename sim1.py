"""
A lightweight 2D N-body simulator with both direct O(N^2) gravity and
Barnes–Hut tree (quadtree) approximation. Uses leapfrog (kick-drift-kick)
for good energy behavior. Suitable for a few hundred to ~50k bodies
(depending on theta).

Controls (when plotting):
    • Space: pause/resume
    • s: step once when paused
    • r: reset to initial conditions

Outputs (optional):
    • CSV snapshots (positions, velocities) to a folder
    • Energy log to stdout
"""
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
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ----------------------------- Utilities ------------------------------------

@dataclasses.dataclass
class SimConfig:
    G: float = 1.0                 # Gravitational constant
    eps: float = 1e-3              # Softening length (to avoid singularities)
    dt: float = 0.01               # Base timestep
    steps: int = 1000              # Number of steps to run
    theta: float = 0.6             # Barnes–Hut opening angle (smaller => more accurate)
    method: str = "barnes"         # "direct" or "barnes"
    bounds: float = 10.0           # Initial drawing bounds for visualization
    seed: int = 42                 # RNG seed
    snapshot_dir: Optional[str] = None  # Write CSV snapshots here if provided
    snapshot_every: int = 50
    plot: bool = True              # Live matplotlib plot
    energy_every: int = 20         # Print energy every N steps
    wrap: bool = False             # Periodic wrapping (visual only)


@dataclasses.dataclass
class State:
    x: np.ndarray  # (N, 2)
    v: np.ndarray  # (N, 2)
    m: np.ndarray  # (N,)

    def copy(self) -> "State":
        return State(self.x.copy(), self.v.copy(), self.m.copy())

# ----------------------------- Initializers ----------------------------------

def make_plummer(n: int, scale: float = 1.0, mass: float = 1.0, rng: np.random.Generator | None = None) -> State:
    """Generate a Plummer sphere in 2D (projected), approximately virialized."""
    rng = rng or np.random.default_rng()

    def sample_radius(size: int) -> np.ndarray:
        u = rng.random(size)
        return scale / np.sqrt(u ** (-2/3) - 1)

    r = sample_radius(n)
    theta = rng.uniform(0, 2*np.pi, n)
    x = np.stack([r*np.cos(theta), r*np.sin(theta)], axis=1)

    # Velocities: draw from approximate distribution
    v = np.zeros_like(x)
    for i in range(n):
        R = np.linalg.norm(x[i])
        vesc = math.sqrt(2) * (1 + R*R/scale**2) ** (-0.25)
        q = 0.1
        while True:
            g1, g2 = rng.random(), rng.random()
            y = g1**2 * (1 - g1**2)**3.5
            if y >= g2:
                vmag = g1 * vesc * q
                ang = rng.uniform(0, 2*np.pi)
                v[i, 0] = vmag * np.cos(ang)
                v[i, 1] = vmag * np.sin(ang)
                break

    m = np.full(n, mass / n)
    # remove CM drift
    cm_v = np.average(v, axis=0, weights=m)
    v -= cm_v
    cm_x = np.average(x, axis=0, weights=m)
    x -= cm_x
    return State(x, v, m)


def make_two_body() -> State:
    x = np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float)
    v = np.array([[0.0, 0.6], [0.0, -0.6]], dtype=float)
    m = np.array([1.0, 1.0], dtype=float)
    return State(x, v, m)


# ----------------------------- Force Models ----------------------------------

def accel_direct(state: State, G: float, eps: float) -> np.ndarray:
    x, m = state.x, state.m
    N = x.shape[0]
    a = np.zeros_like(x)
    for i in range(N):
        dx = x[i] - x  # (N,2)
        r2 = np.einsum('ij,ij->i', dx, dx) + eps*eps
        inv_r3 = np.where(r2 > 0, r2 ** -1.5, 0.0)
        # Exclude self with inv_r3[i] == 0 due to r2==eps^2? Better explicit mask:
        mask = np.ones(N, dtype=bool); mask[i] = False
        contrib = (-G) * (m[mask, None] * dx[mask]) * inv_r3[mask, None]
        a[i] = contrib.sum(axis=0)
    return a


# ----------------------------- Barnes–Hut (2D Quadtree) ----------------------

class Quad:
    __slots__ = ("cx", "cy", "hw")
    def __init__(self, cx: float, cy: float, hw: float):
        self.cx, self.cy, self.hw = cx, cy, hw  # center and half-width
    def contains(self, x: float, y: float) -> bool:
        return (self.cx - self.hw <= x <= self.cx + self.hw and
                self.cy - self.hw <= y <= self.cy + self.hw)
    def child(self, ix: int) -> 'Quad':
        # ix: 0..3 -> NW, NE, SW, SE
        dx = -0.5 if ix in (0, 2) else 0.5
        dy = 0.5 if ix in (0, 1) else -0.5
        return Quad(self.cx + dx*self.hw, self.cy + dy*self.hw, self.hw*0.5)

class Node:
    __slots__ = ("quad", "mass", "com", "is_leaf", "idx", "children")
    def __init__(self, quad: Quad):
        self.quad = quad
        self.mass = 0.0
        self.com = np.zeros(2)
        self.is_leaf = True
        self.idx: Optional[int] = None  # index of a single body if leaf
        self.children: list[Optional[Node]] = [None, None, None, None]

class BarnesHut:
    def __init__(self, x: np.ndarray, m: np.ndarray, theta: float, eps: float):
        # Make a square that encloses all points (with margin)
        mn = np.min(x, axis=0); mx = np.max(x, axis=0)
        side = float(np.max(mx - mn))
        cx, cy = float((mn[0] + mx[0]) / 2), float((mn[1] + mx[1]) / 2)
        root_hw = side * 0.55 if side > 0 else 1.0
        self.root = Node(Quad(cx, cy, root_hw))
        self.x = x
        self.m = m
        self.theta = theta
        self.eps = eps
        for i in range(x.shape[0]):
            self._insert(self.root, i)
        self._compute_mass(self.root)

    def _insert(self, node: Node, i: int):
        xi, yi = float(self.x[i, 0]), float(self.x[i, 1])
        # Expand root if necessary
        while not node.quad.contains(xi, yi):
            node = self._expand_root(node, xi, yi)
        self._insert_into(node, i)

    def _expand_root(self, node: Node, x: float, y: float) -> Node:
        q = node.quad
        # Determine which direction the point lies; create a new parent
        dx = 1 if x > q.cx else -1
        dy = 1 if y > q.cy else -1
        new_hw = q.hw * 2
        new_quad = Quad(q.cx + dx*q.hw, q.cy + dy*q.hw, new_hw)
        parent = Node(new_quad)
        parent.is_leaf = False
        # Place old node as a child of the new parent
        idx_old = 0 if (q.cx < parent.quad.cx and q.cy > parent.quad.cy) else None
        # We'll map by relative position
        for k in range(4):
            if parent.quad.child(k).contains(q.cx, q.cy):
                parent.children[k] = node
                break
        return parent

    def _insert_into(self, node: Node, i: int):
        if node.is_leaf:
            if node.idx is None:
                node.idx = i
            else:
                # Subdivide
                existing = node.idx
                node.idx = None
                node.is_leaf = False
                for k in range(4):
                    node.children[k] = Node(node.quad.child(k))
                self._insert_into(self._which_child(node, existing), existing)
                self._insert_into(self._which_child(node, i), i)
        else:
            self._insert_into(self._which_child(node, i), i)

    def _which_child(self, node: Node, i: int) -> Node:
        xi, yi = float(self.x[i,0]), float(self.x[i,1])
        for k in range(4):
            ch = node.children[k]
            if ch is None:
                ch = Node(node.quad.child(k))
                node.children[k] = ch
            if ch.quad.contains(xi, yi):
                return ch
        # Fallback (shouldn't happen if expansion works): return first
        return node.children[0]

    def _compute_mass(self, node: Node) -> Tuple[float, np.ndarray]:
        if node is None:
            return 0.0, np.zeros(2)
        if node.is_leaf:
            if node.idx is None:
                node.mass = 0.0
                node.com = np.zeros(2)
            else:
                node.mass = float(self.m[node.idx])
                node.com = self.x[node.idx].astype(float)
            return node.mass, node.com
        mass = 0.0
        com = np.zeros(2)
        for ch in node.children:
            if ch is None:
                continue
            m_c, c_c = self._compute_mass(ch)
            mass += m_c
            com += m_c * c_c
        node.mass = mass
        node.com = com / mass if mass > 0 else np.zeros(2)
        return node.mass, node.com

    def accel(self, i: int, G: float) -> np.ndarray:
        return self._accel_from(self.root, i, G)

    def _accel_from(self, node: Node, i: int, G: float) -> np.ndarray:
        if node is None or node.mass == 0.0:
            return np.zeros(2)
        xi = self.x[i]
        if node.is_leaf and node.idx is not None and node.idx == i:
            return np.zeros(2)
        dx = node.com - xi
        r2 = float(np.dot(dx, dx)) + self.eps*self.eps
        r = math.sqrt(r2)
        # Opening criterion: s / r < theta, where s ~ node size
        s = node.quad.hw * 2
        if node.is_leaf or s / r < self.theta:
            inv_r3 = r2 ** -1.5
            return (-G) * node.mass * dx * inv_r3
        acc = np.zeros(2)
        for ch in node.children:
            if ch is not None:
                acc += self._accel_from(ch, i, G)
        return acc

# ----------------------------- Integrator ------------------------------------

def leapfrog_step(state: State, dt: float, G: float, eps: float, method: str, theta: float) -> Tuple[State, np.ndarray]:
    x, v, m = state.x, state.v, state.m
    # Kick half-step
    a0 = compute_accel(state, G, eps, method, theta)
    v_half = v + 0.5 * dt * a0
    # Drift
    x_new = x + dt * v_half
    new_state = State(x_new, v_half.copy(), m)
    # Recompute accel at new positions
    a1 = compute_accel(new_state, G, eps, method, theta)
    # Kick half-step to complete
    v_new = v_half + 0.5 * dt * a1
    return State(x_new, v_new, m), a1


def compute_accel(state: State, G: float, eps: float, method: str, theta: float) -> np.ndarray:
    if method == "direct":
        return accel_direct(state, G, eps)
    elif method == "barnes":
        bh = BarnesHut(state.x, state.m, theta=theta, eps=eps)
        a = np.zeros_like(state.x)
        for i in range(state.x.shape[0]):
            a[i] = bh.accel(i, G)
        return a
    else:
        raise ValueError(f"Unknown method: {method}")

# ----------------------------- Diagnostics -----------------------------------

def kinetic_energy(state: State) -> float:
    return float(0.5 * np.sum(state.m * np.einsum('ij,ij->i', state.v, state.v)))


def potential_energy(state: State, G: float, eps: float) -> float:
    x, m = state.x, state.m
    N = x.shape[0]
    U = 0.0
    for i in range(N):
        dx = x[i] - x[i+1:]
        r = np.sqrt(np.einsum('ij,ij->i', dx, dx) + eps*eps)
        U += -G * float(m[i]) * np.sum(m[i+1:] / r)
    return float(U)

# ----------------------------- I/O -------------------------------------------

def maybe_write_snapshot(step: int, state: State, cfg: SimConfig):
    if cfg.snapshot_dir is None:
        return
    if step % cfg.snapshot_every != 0:
        return
    os.makedirs(cfg.snapshot_dir, exist_ok=True)
    path = os.path.join(cfg.snapshot_dir, f"snap_{step:06d}.csv")
    arr = np.hstack([state.x, state.v, state.m[:, None]])
    np.savetxt(path, arr, delimiter=",", header="x,y,vx,vy,m", comments="")

# ----------------------------- Visualization ---------------------------------

class LivePlot:
    def __init__(self, cfg: SimConfig, init_state: State):
        if not HAS_MPL:
            raise RuntimeError("matplotlib not available; install it or run with --plot false")
        self.cfg = cfg
        self.init_state = init_state.copy()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.scat = self.ax.scatter([], [])
        self.text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, va='top')
        self.paused = False
        self.step_once = False
        self._connect()
        self._setup_axes()

    def _setup_axes(self):
        b = self.cfg.bounds
        self.ax.set_xlim(-b, b)
        self.ax.set_ylim(-b, b)
        self.ax.set_aspect('equal', 'box')
        self.ax.set_title("N-body simulation (Barnes–Hut)")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

    def _connect(self):
        def on_key(event):
            if event.key == ' ':
                self.paused = not self.paused
            elif event.key == 's':
                self.step_once = True
            elif event.key == 'r':
                self.reset()
        self.fig.canvas.mpl_connect('key_press_event', on_key)

    def reset(self):
        self.paused = False
        self.step_once = False

    def update(self, step: int, state: State, KE: float, PE: float):
        self.scat.set_offsets(state.x)
        self.text.set_text(f"step {step}\nKE={KE:.3f}  PE={PE:.3f}  E={KE+PE:.3f}")
        if self.cfg.wrap:
            b = self.cfg.bounds
            state.x[:,0] = (state.x[:,0] + b) % (2*b) - b
            state.x[:,1] = (state.x[:,1] + b) % (2*b) - b
        plt.pause(0.001)

# ----------------------------- Main loop -------------------------------------

def run(state: State, cfg: SimConfig):
    if cfg.plot and not HAS_MPL:
        print("matplotlib not found; proceeding without plotting", file=sys.stderr)
        cfg.plot = False

    plotter = LivePlot(cfg, state) if cfg.plot else None

    # Leapfrog requires a half-step velocity; we store full-step velocities in state
    for step in range(cfg.steps + 1):
        KE = kinetic_energy(state)
        PE = potential_energy(state, cfg.G, cfg.eps)
        if plotter:
            # Handle pause/step
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
    p = argparse.ArgumentParser(description="2D N-body simulator with Barnes–Hut")
    p.add_argument('--n', type=int, default=1000, help='number of bodies (for plummer)')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--G', type=float, default=1.0)
    p.add_argument('--eps', type=float, default=1e-2)
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
    args = parse_args(argv)
    cfg = SimConfig(
        G=args.G, eps=args.eps, dt=args.dt, steps=args.steps, theta=args.theta,
        method=args.method, bounds=args.bounds, seed=args.seed, snapshot_dir=args.snapshot_dir,
        snapshot_every=args.snapshot_every, plot=args.plot, energy_every=args.energy_every,
        wrap=args.wrap,
    )
    rng = np.random.default_rng(cfg.seed)
    if args.init == 'plummer':
        state = make_plummer(args.n, scale=args.scale, mass=1.0, rng=rng)
    else:
        state = make_two_body()
    run(state, cfg)


if __name__ == '__main__':
    main()
