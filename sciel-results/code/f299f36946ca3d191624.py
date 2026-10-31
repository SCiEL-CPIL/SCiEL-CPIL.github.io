"""LJ38 cluster global minimum via Molecular Dynamics with adaptive cooling."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 38

def _lj_energy(pos):
    """Total Lennard-Jones energy 4*sum(r^-12 - r^-6) over all pairs."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    d2 = d * d
    iu = np.triu_indices(N, k=1)
    r2 = d2[iu]
    inv_r6 = 1.0 / (r2 * r2 * r2)
    return 4.0 * np.sum(inv_r6 * inv_r6 - inv_r6)

def _lj_grad(pos):
    """Gradient of LJ energy."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, 1.0)
    grad = np.zeros_like(pos)
    for i in range(N):
        for j in range(i + 1, N):
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            coeff = 4.0 * (12.0 / (r2 * r2 * r2 * r2 * r2 * r2) - 6.0 / (r2 * r2 * r2))
            grad[i] += coeff * rij
            grad[j] -= coeff * rij
    return grad

def _relax(pos):
    """Local L-BFGS relaxation."""
    try:
        res = minimize(_lj_energy, pos, jac=_lj_grad, method='L-BFGS-B',
                       options={'maxiter': 3000, 'gtol': 1e-9, 'ftol': 1e-12})
        return res.x
    except Exception:
        return pos

def _enforce_min_separation(pos, min_sep=0.5):
    """Push atoms apart if any pair is closer than min_sep."""
    for _ in range(100):
        moved = False
        d = cdist(pos, pos)
        np.fill_diagonal(d, np.inf)
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < min_sep:
                    rij = pos[i] - pos[j]
                    r = np.linalg.norm(rij)
                    if r < 1e-12:
                        rij = np.array([1.0, 0.0, 0.0])
                        r = 1.0
                    push = (min_sep - r) * 0.5
                    unit = rij / r
                    pos[i] += push * unit
                    pos[j] -= push * unit
                    moved = True
        if not moved:
            break
    return pos

def _md_step(pos, vel, dt, T):
    """One Langevin MD step with velocity Verlet."""
    forces = -_lj_grad(pos)
    vel_half = vel + 0.5 * dt * forces
    pos_new = pos + dt * vel_half
    forces_new = -_lj_grad(pos_new)
    gamma = 5.0
    vel_new = (vel_half + 0.5 * dt * forces_new) * (1.0 - gamma * dt)
    vel_new += np.sqrt(2.0 * gamma * T * dt) * np.random.randn(N, 3)
    return pos_new, vel_new

def _fcc_seed():
    """FCC truncated octahedron seed."""
    a = 2.0 ** (2.0 / 3.0)
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (i + bx) * a
                    y = (j + by) * a
                    z = (k + bz) * a
                    points.append([x, y, z])
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    return points[:N].copy()

def _run_md_cooling(pos0, dt=0.004, T_start=3.0, T_end=0.005, n_steps=4000, refine_every=400):
    """Run MD with linear cooling and periodic L-BFGS refinement."""
    pos = pos0.copy()
    vel = np.random.randn(N, 3) * np.sqrt(T_start / N)
    best_pos = pos.copy()
    best_e = _lj_energy(pos)
    
    for step in range(n_steps):
        frac = step / n_steps
        T = T_start * (1.0 - frac) + T_end * frac
        pos, vel = _md_step(pos, vel, dt, T)
        pos = _enforce_min_separation(pos)
        
        if step % refine_every == 0:
            relaxed = _relax(pos)
            relaxed = _enforce_min_separation(relaxed)
            e = _lj_energy(relaxed)
            if e < best_e:
                best_e = e
                best_pos = relaxed.copy()
            pos = relaxed
    
    # Final refinement
    relaxed = _relax(best_pos)
    relaxed = _enforce_min_separation(relaxed)
    e = _lj_energy(relaxed)
    if e < best_e:
        best_e = e
        best_pos = relaxed
    
    return best_pos, best_e

def construct():
    np.random.seed(42)
    
    # Multiple independent MD runs from different starting points
    best_pos = None
    best_energy = np.inf
    
    # Run 1: FCC seed
    pos0 = _fcc_seed()
    pos0 = _enforce_min_separation(pos0)
    p, e = _run_md_cooling(pos0, dt=0.004, T_start=3.0, T_end=0.005, n_steps=3000, refine_every=500)
    if e < best_energy:
        best_energy = e
        best_pos = p
    
    # Run 2: Random cluster
    pos0 = np.random.randn(N, 3) * 2.5
    pos0 = _enforce_min_separation(pos0)
    p, e = _run_md_cooling(pos0, dt=0.004, T_start=4.0, T_end=0.005, n_steps=3000, refine_every=500)
    if e < best_energy:
        best_energy = e
        best_pos = p
    
    # Run 3: Another random cluster
    pos0 = np.random.randn(N, 3) * 3.0
    pos0 = _enforce_min_separation(pos0)
    p, e = _run_md_cooling(pos0, dt=0.004, T_start=5.0, T_end=0.005, n_steps=3000, refine_every=500)
    if e < best_energy:
        best_energy = e
        best_pos = p
    
    # Final relaxation of best
    final = _relax(best_pos)
    final = _enforce_min_separation(final)
    e = _lj_energy(final)
    if e < best_energy:
        best_energy = e
        best_pos = final
    
    return best_pos
