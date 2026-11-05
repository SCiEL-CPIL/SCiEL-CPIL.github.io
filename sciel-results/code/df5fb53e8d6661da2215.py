# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_GLOBAL_START = [None]
_DEADLINE = 320.0  # safety margin before 330s kill

def _remaining():
    if _GLOBAL_START[0] is None:
        _GLOBAL_START[0] = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START[0])

def _morse_energy(pos, rho):
    n = len(pos)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            x = rho * (1.0 - r)
            ex = math.exp(x)
            E += ex * (ex - 2.0)
    return E

def _morse_grad(pos, rho):
    n = len(pos)
    g = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]
            r = np.linalg.norm(d)
            if r < 1e-10:
                continue
            x = rho * (1.0 - r)
            ex = math.exp(x)
            # dE/dr = rho * exp(x) * (2*exp(x) - 2) = 2*rho*ex*(ex-1)
            dEdr = 2.0 * rho * ex * (ex - 1.0)
            # dE/dpos_i = dEdr * (pos_j - pos_i)/r = -dEdr * d/r
            f = -dEdr / r
            g[i] += f * d
            g[j] -= f * d
    return g

def _gen_fcc(n, spacing=1.0):
    """Generate n atoms from FCC lattice, take lowest-energy subset by compactness."""
    coords = []
    # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
    basis = np.array([
        [0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]
    ])
    L = 2  # lattice constant in units of spacing
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            for iz in range(-2, 3):
                for b in basis:
                    p = (np.array([ix, iy, iz]) * L + b) * spacing
                    coords.append(p)
    coords = np.array(coords)
    # center
    coords -= coords.mean(axis=0)
    if n <= len(coords):
        # pick n closest to origin
        d2 = np.sum(coords**2, axis=1)
        idx = np.argsort(d2)[:n]
        return coords[idx]
    # not enough, pad with random
    extra = np.random.randn(n - len(coords)) * 0.1
    return np.vstack([coords, extra])

def _gen_bcc(n, spacing=1.0):
    """Generate n atoms from BCC lattice."""
    coords = []
    basis = np.array([
        [0,0,0], [0.5,0.5,0.5]
    ])
    L = 2
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            for iz in range(-2, 3):
                for b in basis:
                    p = (np.array([ix, iy, iz]) * L + b) * spacing
                    coords.append(p)
    coords = np.array(coords)
    coords -= coords.mean(axis=0)
    if n <= len(coords):
        d2 = np.sum(coords**2, axis=1)
        idx = np.argsort(d2)[:n]
        return coords[idx]
    extra = np.random.randn(n - len(coords)) * 0.1
    return np.vstack([coords, extra])

def _gen_hcp(n, spacing=1.0):
    """Generate n atoms from HCP lattice."""
    coords = []
    a = spacing  # in-plane spacing
    c = spacing * math.sqrt(8.0/3.0)  # ideal c/a
    # HCP basis in 2D hexagonal: layer A at z=0, layer B at z=c/2
    # 2D hex lattice points
    for layer in range(6):
        z = layer * c / 2.0
        for ix in range(-3, 4):
            for iy in range(-3, 4):
                x = a * (ix + 0.5 * (iy % 2))
                y = a * math.sqrt(3.0) / 2.0 * iy
                if layer % 2 == 1:
                    x += a / 2.0
                coords.append([x, y, z])
    coords = np.array(coords)
    coords -= coords.mean(axis=0)
    if n <= len(coords):
        d2 = np.sum(coords**2, axis=1)
        idx = np.argsort(d2)[:n]
        return coords[idx]
    extra = np.random.randn(n - len(coords)) * 0.1
    return np.vstack([coords, extra])

def _gen_random(n, spacing=1.0, seed=None):
    """Random compact cluster with minimum distance."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.9
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55 * spacing:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    if len(points) < n:
        # fallback: just random
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _gen_shell(n, spacing=1.0):
    """Spherical shell packing - atoms on a sphere surface."""
    # Fibonacci sphere
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    radius = spacing * n ** (1.0/3.0) * 0.5
    points = []
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1) if n > 1 else 0.0
        r = math.sqrt(max(0.0, 1.0 - z*z))
        theta = 2.0 * math.pi * i / golden
        p = np.array([r * math.cos(theta), r * math.sin(theta), z]) * radius
        points.append(p)
    return np.array(points)

def _relax_lbfgs(pos, rho, time_budget=10.0):
    """Relax using L-BFGS-B with gradient."""
    start = time.time()
    def fun(x):
        return _morse_energy(x.reshape(-1, 3), rho)
    def jac(x):
        return _morse_grad(x.reshape(-1, 3), rho).flatten()
    try:
        res = minimize(fun, pos.flatten(), jac=jac, method='L-BFGS-B',
                       options={'maxiter': 200, 'gtol': 1e-8,
                                'callback': lambda x: (time.time() - start > time_budget) and 1 or None})
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return pos, _morse_energy(pos, rho)

def _relax_with_min_dist(pos, rho, min_d=0.5):
    """Relax and enforce minimum distance."""
    pos, E = _relax_lbfgs(pos, rho)
    # Check min distance; if violated, nudge
    for _ in range(3):
        violated = False
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    violated = True
                    mid = (pos[i] + pos[j]) / 2.0
                    direction = pos[i] - pos[j]
                    if np.linalg.norm(direction) < 1e-10:
                        direction = np.array([1.0, 0, 0])
                    direction /= np.linalg.norm(direction)
                    push = (min_d - d) / 2.0 + 0.01
                    pos[i] += direction * push
                    pos[j] -= direction * push
        if not violated:
            break
        # Re-relax after nudging
        pos, E = _relax_lbfgs(pos, rho, time_budget=5.0)
    return pos, E

def construct(instance):
    """Multi-start optimization with crystalline templates and L-BFGS-B."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    if _GLOBAL_START[0] is None:
        _GLOBAL_START[0] = time.time()
    
    rem = _remaining()
    if rem < 5.0:
        # Emergency: return a simple random cluster
        return _gen_random(n, seed=42)
    
    # Generate diverse starting points
    starts = []
    
    # Crystalline templates
    starts.append(_gen_fcc(n))
    starts.append(_gen_bcc(n))
    starts.append(_gen_hcp(n))
    starts.append(_gen_shell(n))
    
    # Jittered crystals
    for template in [_gen_fcc(n), _gen_bcc(n)]:
        jitter = np.random.randn(n, 3) * 0.15
        starts.append(template + jitter)
    
    # Random compact
    for seed in [0, 1, 2]:
        starts.append(_gen_random(n, seed=seed))
    
    # Per-instance time budget
    per_start_time = max(3.0, rem / (len(starts) + 1))
    
    best_pos = None
    best_E = np.inf
    
    for si, start_pos in enumerate(starts):
        if _remaining() < 5.0:
            break
        
        # Ensure minimum distance before relaxation
        for i in range(len(start_pos)):
            for j in range(i+1, len(start_pos)):
                d = np.linalg.norm(start_pos[i] - start_pos[j])
                if d < 0.5:
                    mid = (start_pos[i] + start_pos[j]) / 2.0
                    direction = start_pos[i] - start_pos[j]
                    if np.linalg.norm(direction) < 1e-10:
                        direction = np.array([1.0, 0, 0])
                    direction /= np.linalg.norm(direction)
                    push = (0.5 - d) / 2.0 + 0.01
                    start_pos[i] += direction * push
                    start_pos[j] -= direction * push
        
        # Relax with L-BFGS-B
        t0 = time.time()
        pos, E = _relax_with_min_dist(start_pos, rho, min_d=0.5)
        
        # If L-BFGS-B was too slow, try the built-in relax as fallback
        if time.time() - t0 > per_start_time * 1.5:
            pass  # already spent time
        
        if E < best_E:
            best_E = E
            best_pos = pos.copy()
        
        # If we have good time, do a second round with perturbation of best
        if _remaining() > 30.0 and si % 3 == 0 and best_pos is not None:
            perturbed = best_pos + np.random.randn(n, 3) * 0.05
            pos2, E2 = _relax_with_min_dist(perturbed, rho, min_d=0.5)
            if E2 < best_E:
                best_E = E2
                best_pos = pos2.copy()
    
    if best_pos is None:
        # Shouldn't happen, but safety
        best_pos = _gen_random(n, seed=99)
    
    # Final verification: ensure min distance
    for i in range(len(best_pos)):
        for j in range(i+1, len(best_pos)):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < 0.5:
                direction = best_pos[i] - best_pos[j]
                if np.linalg.norm(direction) < 1e-10:
                    direction = np.array([1.0, 0, 0])
                direction /= np.linalg.norm(direction)
                push = (0.5 - d) / 2.0 + 0.01
                best_pos[i] += direction * push
                best_pos[j] -= direction * push
    
    return best_pos
# EVOLVE-BLOCK-END
