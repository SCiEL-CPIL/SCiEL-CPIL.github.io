"""LJ38 global minimum search via FCC truncated octahedron seed + L-BFGS relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)  # avoid division by zero
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    # Sum over upper triangle only
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return 4.0 * np.sum(r12[mask] - r6[mask])

def _lj_energy_grad(pos):
    """Compute energy and gradient."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    energy = 4.0 * np.sum(r12[mask] - r6[mask])
    
    # Gradient: dE/dx_i = sum_j 4*(-12*r12_ij*r_ij + 6*r6_ij*r_ij) / r_ij^2 * (x_i - x_j)
    # More carefully: for pair (i,j), force on i is -dE/dx_i
    # E_pair = 4*(r^-12 - r^-6)
    # dE_pair/dx_i = 4*(-12*r^-13 * dr/dx_i + 6*r^-7 * dr/dx_i)
    # dr/dx_i = (x_i - x_j)/r
    # So dE_pair/dx_i = 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)
    # = 4*(6*r^-8 - 12*r^-14) * (x_i - x_j)
    
    # Vectorized: for each pair (i,j) with i<j, contribution to grad_i is:
    # 4*(6*r6_ij - 12*r12_ij) * (x_i - x_j)
    # and to grad_j is: -4*(6*r6_ij - 12*r12_ij) * (x_i - x_j) = 4*(6*r6_ij - 12*r12_ij) * (x_j - x_i)
    
    coeff = 4.0 * (6.0 * r6 - 12.0 * r12)  # shape (n, n)
    coeff = coeff * mask  # only upper triangle
    # grad_i = sum_j coeff[i,j] * (x_i - x_j)
    # = sum_j coeff[i,j] * x_i - sum_j coeff[i,j] * x_j
    # = (sum_j coeff[i,j]) * x_i - (coeff @ x)_i
    
    row_sum = coeff.sum(axis=1, keepdims=True)  # shape (n, 1)
    grad = row_sum * pos - (coeff @ pos)
    return energy, grad

def _fcc_truncated_octahedron():
    """Generate FCC truncated octahedron cluster of 38 atoms.
    
    FCC lattice points within a truncated octahedron centered at origin.
    The 38-atom FCC cluster consists of:
    - 1 center atom
    - 12 nearest neighbors (octahedral + tetrahedral in FCC)
    - 24 next-nearest neighbors
    
    Actually, the standard 38-atom FCC cluster is a truncated octahedron with
    layers. Let me construct it properly.
    
    FCC lattice: points at (a/2)(i, j, k) where i+j+k is even, or equivalently
    the union of simple cubic lattices offset by (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5).
    
    For the 38-atom cluster, we take all FCC lattice points within a certain radius.
    The nearest neighbor distance in FCC is a/sqrt(2). In reduced LJ units, sigma=1,
    and the minimum is at r=2^(1/6)*sigma ≈ 1.1225.
    
    Let me just generate FCC points and select 38 closest to center.
    """
    # FCC lattice: generate points in a box
    # In reduced units, nearest neighbor distance should be around 2^(1/6) ≈ 1.1225
    # FCC lattice constant a: nearest neighbor = a/sqrt(2), so a = sqrt(2) * 2^(1/6)
    
    a = np.sqrt(2.0) * 2.0 ** (1.0/6.0)
    
    # Generate FCC points
    points = []
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) in units of a
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for b in basis:
                    pt = a * (np.array([i, j, k]) + b)
                    points.append(pt)
    
    points = np.array(points)
    
    # Select 38 points closest to origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    selected = points[idx[:N]]
    
    return selected

def _relax(pos, maxiter=2000, ftol=1e-10, gtol=1e-8):
    """Relax positions using L-BFGS with energy and gradient."""
    from scipy.optimize import minimize
    
    def obj(x):
        e, g = _lj_energy_grad(x.reshape(N, 3))
        return e, g
    
    res = minimize(obj, pos.flatten(), method='L-BFGS-B', 
                   jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
    return res.x.reshape(N, 3)

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    # Start with FCC truncated octahedron
    pos = _fcc_truncated_octahedron()
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    # Relax from the FCC seed
    try:
        relaxed = _relax(pos, maxiter=2000, ftol=1e-10, gtol=1e-8)
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_pos = relaxed
            best_energy = e
    except Exception:
        pass
    
    # Try a few more relaxations from slightly perturbed FCC seeds if time permits
    if time.time() < deadline - 30.0:
        rng = np.random.RandomState(42)
        for trial in range(5):
            if time.time() > deadline - 15.0:
                break
            perturbed = best_pos + rng.normal(0, 0.05, size=(N, 3))
            try:
                relaxed = _relax(perturbed, maxiter=1000, ftol=1e-10, gtol=1e-8)
                e = _lj_energy(relaxed)
                if e < best_energy:
                    best_pos = relaxed
                    best_energy = e
            except Exception:
                pass
    
    # Final high-precision polish
    if time.time() < deadline - 10.0:
        try:
            polished = _relax(best_pos, maxiter=5000, ftol=1e-12, gtol=1e-10)
            e = _lj_energy(polished)
            if e < best_energy:
                best_pos = polished
                best_energy = e
        except Exception:
            pass
    
    # Ensure minimum separation of 0.5
    # Check and fix if any pairs are too close
    for _ in range(10):
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, np.inf)
        r = np.sqrt(r2)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        if np.min(r[mask]) >= 0.5:
            break
        # Nudge atoms that are too close apart
        close_pairs = np.argwhere(mask & (r < 0.5))
        for i, j in close_pairs:
            mid = (best_pos[i] + best_pos[j]) / 2.0
            direction = best_pos[i] - best_pos[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.array([1.0, 0.0, 0.0])
            best_pos[i] += 0.01 * direction
            best_pos[j] -= 0.01 * direction
    
    return best_pos
# EVOLVE-BLOCK-END
