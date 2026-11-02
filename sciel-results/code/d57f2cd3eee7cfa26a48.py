"""Parallel tempering Monte Carlo for circle packing in unit square."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 26
DEADLINE = 110.0  # safety margin before 120s kill

def _initial_grid():
    """Hexagonal-ish grid of 26 equal circles, radii near max for grid."""
    # Use a 5x5 grid, drop 2 corners to get 26
    side = 5
    r0 = 0.5 / side
    pts = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            pts.append([x, y, r0])
    pts = pts[:N]
    return np.array(pts)

def _validity(p):
    """Return True if packing is valid (all circles inside square, disjoint)."""
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
        return False
    if np.any(r < 0):
        return False
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return False
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return False
    # pair distances
    d = cKDTree(p[:, :2])
    pairs = d.query_pairs(2.0)
    for i, j in pairs:
        if np.hypot(p[i,0]-p[j,0], p[i,1]-p[j,1]) < p[i,2] + p[j,2] - 1e-9:
            return False
    return True

def _energy(p):
    """Energy = sum of violations (0 if valid). Higher = worse."""
    x, y, r = p[:, 0], p[:, 1], p[:, 2]
    e = 0.0
    # wall violations
    e += np.sum(np.maximum(0, -x + r))
    e += np.sum(np.maximum(0, x - r - 1))
    e += np.sum(np.maximum(0, -y + r))
    e += np.sum(np.maximum(0, y - r - 1))
    # pairwise overlap
    d = cKDTree(p[:, :2])
    pairs = d.query_pairs(2.0)
    for i, j in pairs:
        dist = np.hypot(p[i,0]-p[j,0], p[i,1]-p[j,1])
        e += max(0.0, p[i,2] + p[j,2] - dist)
    return e

def construct():
    t_start = time.time()
    rng = np.random.default_rng(12345)

    # Build replicas: replicas 0..NREP-1, replica 0 is "coldest" (best)
    NREP = 8
    # Temperatures (beta = 1/T). Low beta = hot (explore), high beta = cold (exploit)
    betas = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])  # replica 0 = hottest? 
    # Let's define replica 0 as coldest (highest beta) for tracking best
    betas = np.array([32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5, 0.0])

    # Initialize all replicas from grid with jitter
    base = _initial_grid()
    replicas = []
    for k in range(NREP):
        p = base.copy()
        jitter = 0.02
        p[:, 0] += rng.uniform(-jitter, jitter, N)
        p[:, 1] += rng.uniform(-jitter, jitter, N)
        # vary radii a bit
        p[:, 2] *= rng.uniform(0.9, 1.0, N)
        # clamp into square
        p[:, 0] = np.clip(p[:, 0], p[:, 2], 1 - p[:, 2])
        p[:, 1] = np.clip(p[:, 1], p[:, 2], 1 - p[:, 2])
        replicas.append(p)

    energies = np.array([_energy(p) for p in replicas])
    # best = replica with lowest energy; track its value
    best_p = replicas[int(np.argmin(energies))].copy()
    best_val = -np.sum(best_p[:, 2])
    best_energy = energies.min()

    # Move sizes (anneal down)
    pos_step = 0.02
    rad_step = 0.005

    step = 0
    while time.time() - t_start < DEADLINE:
        # Each iteration: one MC move per replica, then attempt swap
        for k in range(NREP):
            p = replicas[k]
            e = energies[k]
            # pick a circle
            i = rng.integers(N)
            # decide move type: position (70%) or radius (30%)
            if rng.random() < 0.7:
                dx = rng.uniform(-pos_step, pos_step)
                dy = rng.uniform(-pos_step, pos_step)
                new_p = p.copy()
                new_p[i, 0] += dx
                new_p[i, 1] += dy
            else:
                dr = rng.uniform(-rad_step, rad_step)
                new_p = p.copy()
                new_p[i, 2] = max(1e-6, p[i, 2] + dr)
            new_e = _energy(new_p)
            delta = new_e - e
            # Metropolis
            if delta <= 0 or rng.random() < np.exp(-betas[k] * delta):
                replicas[k] = new_p
                energies[k] = new_e
            # shrink steps slowly
        # swap adjacent replicas (nearest-neighbor Metropolis swap)
        # try swapping k and k+1 for random k
        k = rng.integers(NREP - 1)
        k2 = k + 1
        # swap acceptance: exp(-beta_k*(E_k2 - E_k) - beta_k2*(E_k - E_k2))
        # = exp((beta_k - beta_k2)*(E_k - E_k2))
        log_accept = (betas[k] - betas[k2]) * (energies[k] - energies[k2])
        if log_accept >= 0 or rng.random() < np.exp(log_accept):
            replicas[k], replicas[k2] = replicas[k2], replicas[k]
            energies[k], energies[k2] = energies[k2], energies[k]

        step += 1
        # anneal: reduce step sizes over time
        frac = min(1.0, step / 5000.0)
        pos_step = 0.02 * (1.0 - 0.7 * frac)
        rad_step = 0.005 * (1.0 - 0.7 * frac)

        # track best (lowest energy / valid)
        cur_min_e = energies.min()
        if cur_min_e < best_energy:
            best_energy = cur_min_e
            bi = int(np.argmin(energies))
            best_p = replicas[bi].copy()
            if _validity(best_p):
                best_val = -np.sum(best_p[:, 2])

        # periodically try to enforce validity by clamping best replica
        if step % 200 == 0:
            bi = int(np.argmin(energies))
            p = replicas[bi]
            # clamp walls
            p[:, 0] = np.clip(p[:, 0], p[:, 2], 1 - p[:, 2])
            p[:, 1] = np.clip(p[:, 1], p[:, 2], 1 - p[:, 2])
            p = replicas[bi] = p
            energies[bi] = _energy(p)
            if energies[bi] < best_energy:
                best_energy = energies[bi]
                best_p = p.copy()
                if _validity(best_p):
                    best_val = -np.sum(best_p[:, 2])

    # Final polish: local refinement on best_p
    if _validity(best_p):
        # try to grow radii slightly while staying valid
        for _ in range(50):
            if time.time() - t_start > DEADLINE - 5:
                break
            improved = False
            for i in range(N):
                trial = best_p.copy()
                trial[i, 2] *= 1.001
                if _validity(trial):
                    best_p = trial
                    improved = True
            if not improved:
                break

    # Final validity check; if invalid, fall back to grid
    if not _validity(best_p):
        # repair: shrink radii until valid
        p = best_p.copy()
        for _ in range(100):
            if _validity(p):
                break
            p[:, 2] *= 0.98
        if not _validity(p):
            p = _initial_grid()
        best_p = p

    return best_p
