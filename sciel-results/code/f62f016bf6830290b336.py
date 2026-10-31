import numpy as np
import time
from scipy.optimize import minimize

N = 38
DEADLINE = 120.0  # Safety margin before 150s limit

def _lj_energy(pos):
    """Calculate Lennard-Jones energy."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    # Avoid division by zero
    r2 = np.where(r2 < 1e-10, 1e-10, r2)
    r6 = r2 ** -3
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6)
    # Subtract self-interaction (diagonal is 0 distance, but we masked it)
    # Actually, the sum includes i=j terms if not careful.
    # diff[i,i] = 0, r2=1e-10, energy huge.
    # We must exclude diagonal.
    idx = np.diag_indices(n)
    energy -= 4.0 * (r12[idx] - r6[idx])
    return energy

def _lj_grad(pos):
    """Calculate Lennard-Jones gradient."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    r2 = np.where(r2 < 1e-10, 1e-10, r2)
    r6 = r2 ** -3
    r12 = r6 * r6
    # Gradient of 4*(r^-12 - r^-6) wrt pos
    # dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # dr/dpos = 2*(pos_i - pos_j)
    # dE/dpos_i = sum_j 4*(-12*r^-13 + 6*r^-7) * 2*(pos_i - pos_j)
    # = 8 * sum_j (-12*r^-13 + 6*r^-7) * (pos_i - pos_j)
    # = 8 * sum_j (6*r^-7 - 12*r^-13) * (pos_i - pos_j)
    # = 48 * sum_j (r^-7 - 2*r^-13) * (pos_i - pos_j)
    
    # Vectorized calculation
    # factor = 48 * (r6 - 2*r12)
    factor = 48.0 * (r6 - 2.0 * r12)
    # Zero out diagonal
    idx = np.diag_indices(n)
    factor[idx] = 0.0
    # grad = - factor * diff ?
    # Let's check sign.
    # E = sum 4(r^-12 - r^-6)
    # dE/dx_i = sum_j 4(-12 r^-13 + 6 r^-7) * (x_i - x_j)
    # = sum_j 24 r^-7 (1 - 2 r^-6) * (x_i - x_j)
    # = 24 sum_j (r^-7 - 2 r^-13) * (x_i - x_j)
    # My previous factor was 48. Let's re-derive.
    # V(r) = 4(r^-12 - r^-6)
    # dV/dr = 4(-12 r^-13 + 6 r^-7)
    # dV/dx_i = dV/dr * dr/dx_i = dV/dr * (x_i - x_j)/r
    # = 4(-12 r^-13 + 6 r^-7) * (x_i - x_j)/r
    # = 4(-12 r^-14 + 6 r^-8) * (x_i - x_j)
    # = 24 r^-8 (r^-6 - 2) * (x_i - x_j)
    # = 24 (r^-14 - 2 r^-8) * (x_i - x_j)
    # Let's use r2.
    # r^-14 = r2^-7
    # r^-8 = r2^-4
    # factor = 24 * (r2^-7 - 2*r2^-4)
    # grad_i = sum_j factor_ij * (x_i - x_j)
    
    factor = 24.0 * (r2 ** -7 - 2.0 * r2 ** -4)
    factor[idx] = 0.0
    grad = np.einsum('ij,ijk->ik', factor, diff)
    return grad

def construct():
    start_time = time.time()
    
    # Generate FCC lattice points
    # FCC points: (i, j, k) where i, j, k are integers and i+j+k is even?
    # No, standard FCC lattice vectors are (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    # Let's generate points (i, j, k) in a range, filter for FCC condition.
    # FCC condition: i, j, k all even or all odd? No.
    # Let's just generate a dense grid and keep those that are FCC.
    # Simpler: Generate all integer points in [-5, 5]^3.
    # FCC points are those where i+j+k is even? No, that's BCC.
    # FCC: i, j, k are all even or all odd? No.
    # Let's use the explicit basis.
    # Point = n1*a1 + n2*a2 + n3*a3
    # a1 = (0, 0.5, 0.5), a2 = (0.5, 0, 0.5), a3 = (0.5, 0.5, 0)
    
    max_coord = 5
    n1_range = range(-max_coord, max_coord + 1)
    fcc_points = []
    for n1 in n1_range:
        for n2 in n1_range:
            for n3 in n1_range:
                x = 0.5 * (n2 + n3)
                y = 0.5 * (n1 + n3)
                z = 0.5 * (n1 + n2)
                # Only keep points in a reasonable box
                if abs(x) <= 3 and abs(y) <= 3 and abs(z) <= 3:
                    fcc_points.append([x, y, z])
    
    fcc_points = np.array(fcc_points)
    # Remove duplicates
    fcc_points = np.unique(fcc_points, axis=0)
    
    # Sort by distance to origin
    dists = np.sum(fcc_points ** 2, axis=1)
    sorted_indices = np.argsort(dists)
    fcc_points_sorted = fcc_points[sorted_indices]
    
    # Take the 38 closest
    pos = fcc_points_sorted[:N].copy()
    
    # Relax with L-BFGS
    try:
        res = minimize(_lj_energy, pos, jac=_lj_grad, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
        pos = res.x
    except Exception:
        pass
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        # Polish with high precision
        try:
            res = minimize(_lj_energy, pos, jac=_lj_grad, method='L-BFGS-B', 
                           options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14})
            pos = res.x
        except Exception:
            pass
    
    return pos
