"""Dzugutov cluster global minima: icosahedral subunit graph + genetic topology + hinge relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient, relax

_START = time.time()
TIME_LIMIT = 320.0  # safety margin under the 330 s kill


def _remaining():
    return TIME_LIMIT - (time.time() - _START)


# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16
_MIN_SEP = 0.5


def _energy(pos):
    """Vectorised Dzugutov energy (flat array, n*3)."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    diff = p[:, None, :] - p[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    ri = r[iu]
    e = np.zeros(ri.shape, dtype=float)
    m1 = ri < _a
    m2 = (ri >= _a) & (ri < _b)
    e[m1] = _A * (ri[m1] ** (-_m) - _B) * np.exp(_c / (ri[m1] - _a))
    e[m2] = _B * np.exp(_d / (ri[m2] - _b))
    return float(e.sum())


def _energy_fn(x):
    return _energy(x)


def _grad_fn(x):
    p = np.asarray(x, dtype=float).reshape(-1, 3)
    try:
        g = dzugutov_energy_gradient(p)
        g = np.asarray(g, dtype=float).reshape(-1)
        if np.all(np.isfinite(g)):
            return g
    except Exception:
        pass
    # finite-difference fallback
    n = p.size
    h = 1e-6
    e0 = _energy(x)
    g = np.empty(n, dtype=float)
    for i in range(n):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (_energy(xp) - _energy(xm)) / (2 * h)
    return g


def _min_sep_ok(pos):
    if pos.shape[0] < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(pos.shape[0], k=1)
    return r[iu].min() >= _MIN_SEP - 1e-9


def _make_icosahedron(scale):
    """12 vertices of a regular icosahedron scaled to nearest-neighbour distance ~scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    # edge length of this icosahedron is 2
    v = v * (scale / 2.0)
    return v


def _make_dodecahedron(scale):
    """20 vertices of a regular dodecahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    v = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, inv_phi, phi], [0, inv_phi, -phi], [0, -inv_phi, phi], [0, -inv_phi, -phi],
        [inv_phi, phi, 0], [inv_phi, -phi, 0], [-inv_phi, phi, 0], [-inv_phi, -phi, 0],
        [phi, 0, inv_phi], [phi, 0, -inv_phi], [-phi, 0, inv_phi], [-phi, 0, -inv_phi],
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms
    edge_to_circum = 2 * np.sin(np.pi / 5)
    v = v * (scale / edge_to_circum)
    return v


def _make_cuboctahedron(scale):
    """12 vertices of a cuboctahedron."""
    v = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms
    v = v * (scale / np.sqrt(2.0))
    return v


def _get_icosahedral_directions():
    """Return unit direction vectors for icosahedral vertices."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _get_dodecahedral_directions():
    """Return unit direction vectors for dodecahedral vertices."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi
    v = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, inv_phi, phi], [0, inv_phi, -phi], [0, -inv_phi, phi], [0, -inv_phi, -phi],
        [inv_phi, phi, 0], [inv_phi, -phi, 0], [-inv_phi, phi, 0], [-inv_phi, -phi, 0],
        [phi, 0, inv_phi], [phi, 0, -inv_phi], [-phi, 0, inv_phi], [-phi, 0, -inv_phi],
    ], dtype=float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _generate_icosahedral_subunits(n, base_radius=1.1):
    """
    Generate a set of icosahedral subunit configurations for cluster size n.
    Each subunit is a local icosahedral environment. We model the cluster as
    a collection of overlapping icosahedra.
    """
    ico_dirs = _get_icosahedral_directions()
    dod_dirs = _get_dodecahedral_directions()
    
    # For each possible center, we can place 12 neighbors in icosahedral directions
    # The key is to determine which centers are used and how they overlap
    
    subunits = []
    rng = np.random.default_rng(n)
    
    # Strategy: place a central atom, then add shells
    # Shell 0: 1 atom at center
    # Shell 1: 12 atoms at icosahedral positions
    # Shell 2: 20 atoms at dodecahedral positions (or 30 at edge midpoints)
    # Shell 3: 12 atoms at further out
    
    # Generate multiple candidate topologies
    for trial in range(8):
        pts = [np.zeros(3)]
        placed = 1
        
        # Shell 1: 12 icosahedral positions
        if placed < n:
            for d in ico_dirs:
                if placed >= n:
                    break
                r = base_radius * (1.0 + 0.05 * trial)
                pts.append(d * r)
                placed += 1
        
        # Shell 2: 20 dodecahedral positions
        if placed < n:
            for d in dod_dirs:
                if placed >= n:
                    break
                r = base_radius * (1.65 + 0.05 * trial)
                pts.append(d * r)
                placed += 1
        
        # Shell 3: additional 12 icosahedral positions (rotated)
        if placed < n:
            # Rotate icosahedral directions
            angle = 0.3 * trial
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            for d in ico_dirs:
                if placed >= n:
                    break
                d_rot = np.array([
                    d[0] * cos_a + d[2] * sin_a,
                    d[1],
                    -d[0] * sin_a + d[2] * cos_a
                ])
                d_rot = d_rot / (np.linalg.norm(d_rot) + 1e-12)
                r = base_radius * (2.3 + 0.05 * trial)
                pts.append(d_rot * r)
                placed += 1
        
        # Fill remaining with random positions on outer shells
        if placed < n:
            all_dirs = np.vstack([ico_dirs, dod_dirs])
            for i in range(n - placed):
                d = all_dirs[i % len(all_dirs)]
                # Add random perturbation
                d = d + rng.normal(0, 0.1, 3)
                d = d / (np.linalg.norm(d) + 1e-12)
                r = base_radius * (2.5 + 0.1 * (i % 5))
                pts.append(d * r)
        
        if len(pts) == n:
            cfg = np.array(pts[:n])
            if _min_sep_ok(cfg):
                subunits.append(cfg)
    
    return subunits


def _relax_lbfgsb(pos, maxiter=300):
    """L-BFGS-B relaxation; returns (energy, positions)."""
    x0 = pos.reshape(-1)
    try:
        res = minimize(
            _energy_fn, x0, jac=_grad_fn,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "gtol": 1e-6, "disp": False},
        )
        e = _energy(res.x)
        p = res.x.reshape(-1, 3)
        if _min_sep_ok(p):
            return e, p
    except Exception:
        pass
    return _energy(x0), pos


def _kick_and_relax(pos, kick_scale, n_kicks=3, rng=None):
    """Apply random kicks and relax; returns best (energy, positions)."""
    if rng is None:
        rng = np.random.default_rng(abs(hash(tuple(np.round(pos, 4)))) % (2**31))
    
    best_e = _energy(pos)
    best_p = pos.copy()
    
    for _ in range(n_kicks):
        if _remaining() < 5.0:
            break
        kicked = pos + rng.normal(0.0, kick_scale, size=pos.shape)
        e, p = _relax_lbfgsb(kicked, maxiter=150)
        if e < best_e and _min_sep_ok(p):
            best_e, best_p = e, p
    
    return best_e, best_p


def _genetic_topology_search(n, initial_seeds, pop_size=6, n_generations=3, rng=None):
    """
    Genetic algorithm over icosahedral subunit topologies.
    Each individual is a cluster configuration.
    Fitness is negative energy (lower is better).
    """
    if rng is None:
        rng = np.random.default_rng(n)
    
    # Initialize population
    population = []
    for s in initial_seeds[:pop_size]:
        e, p = _relax_lbfgsb(s, maxiter=200)
        if _min_sep_ok(p):
            population.append((e, p))
    
    # Fill population with random perturbations of best
    while len(population) < pop_size:
        if not population:
            break
        best_e, best_p = population[0]
        kicked = best_p + rng.normal(0.0, 0.3, size=best_p.shape)
        e, p = _relax_lbfgsb(kicked, maxiter=150)
        if _min_sep_ok(p):
            population.append((e, p))
    
    # Sort by energy
    population.sort(key=lambda x: x[0])
    
    best_e, best_p = population[0]
    
    for gen in range(n_generations):
        if _remaining() < 20.0:
            break
        
        # Select parents (tournament selection)
        next_gen = [population[0], population[1]]  # Elitism
        
        while len(next_gen) < pop_size:
            # Tournament selection
            idx1 = rng.integers(0, len(population))
            idx2 = rng.integers(0, len(population))
            if population[idx1][0] < population[idx2][0]:
                parent1 = population[idx1][1]
            else:
                parent1 = population[idx2][1]
            
            idx1 = rng.integers(0, len(population))
            idx2 = rng.integers(0, len(population))
            if population[idx1][0] < population[idx2][0]:
                parent2 = population[idx1][1]
            else:
                parent2 = population[idx2][1]
            
            # Crossover: mix atoms from two parents
            if rng.random() < 0.7:
                # Uniform crossover
                mask = rng.random(n) < 0.5
                child = np.where(mask[:, None], parent1, parent2)
            else:
                # Just take one parent
                child = parent1.copy()
            
            # Mutation: random kick
            kick_scale = 0.1 + 0.2 * gen / max(1, n_generations - 1)
            child = child + rng.normal(0.0, kick_scale, size=child.shape)
            
            # Relax
            e, p = _relax_lbfgsb(child, maxiter=150)
            if _min_sep_ok(p):
                next_gen.append((e, p))
        
        # Update population
        population = next_gen
        population.sort(key=lambda x: x[0])
        
        if population[0][0] < best_e:
            best_e, best_p = population[0]
    
    return best_e, best_p


def construct(instance):
    n = instance["n"]
    potential = instance.get("potential", "dzugutov")
    
    # Generate initial icosahedral subunit seeds
    seeds = _generate_icosahedral_subunits(n)
    
    if not seeds:
        # Last-resort compact random
        rng = np.random.default_rng(n)
        rad = 1.1 * n ** (1.0 / 3.0)
        pts = []
        while len(pts) < n:
            p = rng.uniform(-rad, rad, 3)
            if np.linalg.norm(p) <= rad and all(np.linalg.norm(p - q) >= 0.8 for q in pts):
                pts.append(p)
        seeds = [np.array(pts)]
    
    best_e = np.inf
    best_pos = seeds[0]
    t0 = time.time()
    
    # Phase 1: Relax each seed with L-BFGS-B
    for s in seeds:
        if _remaining() < 15.0:
            break
        try:
            e, p = _relax_lbfgsb(s)
            if e < best_e and _min_sep_ok(p):
                best_e, best_pos = e, p
        except Exception:
            pass
    
    # Phase 2: Genetic topology search
    if _remaining() > 30.0:
        try:
            rng = np.random.default_rng(n + 1000)
            e, p = _genetic_topology_search(n, seeds, pop_size=5, n_generations=3, rng=rng)
            if e < best_e and _min_sep_ok(p):
                best_e, best_pos = e, p
        except Exception:
            pass
    
    # Phase 3: Kick-and-relax from best structure
    kick_scales = [0.08, 0.15, 0.25, 0.40]
    rng = np.random.default_rng(n + 2000)
    for scale in kick_scales:
        if _remaining() < 10.0:
            break
        try:
            e, p = _kick_and_relax(best_pos, scale, n_kicks=3, rng=rng)
            if e < best_e and _min_sep_ok(p):
                best_e, best_pos = e, p
        except Exception:
            pass
    
    # Phase 4: Try the built-in relax function if available
    if _remaining() > 15.0:
        try:
            relaxed = relax(best_pos, 'dzugutov')
            e = _energy(relaxed)
            if e < best_e and _min_sep_ok(relaxed):
                best_e, best_pos = e, relaxed
        except Exception:
            pass
    
    return np.asarray(best_pos, dtype=float)
# EVOLVE-BLOCK-END
