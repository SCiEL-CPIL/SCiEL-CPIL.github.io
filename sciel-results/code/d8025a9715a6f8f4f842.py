import numpy as np
from scipy.optimize import minimize
import time

N = 38

def lj_energy(pos):
    # pos shape (N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Avoid division by zero
    dist_sq[dist_sq == 0] = 1e-10
    inv_r6 = 1.0 / (dist_sq**3)
    inv_r12 = inv_r6**2
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    # Subtract self-interaction (N terms of 4*(1-1)=0, but numerically better to handle)
    # Actually, np.sum includes i=j terms.
    # For i=j, dist_sq is 0 (or 1e-10).
    # If we set dist_sq to 1e-10, energy is huge.
    # Better to mask diagonal.
    mask = ~np.eye(N, dtype=bool)
    energy = 4.0 * np.sum((inv_r12 - inv_r6)[mask])
    # The sum over mask counts each pair twice (i,j) and (j,i).
    # LJ energy is sum over i<j.
    # So divide by 2.
    return energy / 2.0

def lj_forces(pos):
    # Returns gradient of energy
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq[dist_sq == 0] = 1e-10
    # r = sqrt(dist_sq)
    # F_vec = 24 * (2/r^13 - 1/r^7) * r_vec
    # r_vec = diff
    # r^13 = (dist_sq)^(6.5)
    # r^7 = (dist_sq)^3.5
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2**3
    inv_r12 = inv_r6**2
    # Coefficient for vector
    # F = 24 * (2 * inv_r12 * inv_r2 - inv_r6 * inv_r2) * diff
    # Wait, d/dr (4(r^-12 - r^-6)) = 4(-12 r^-13 + 6 r^-7) = 24(-2 r^-13 + r^-7)
    # Force on i due to j: F_ij = - dU/dr * r_hat = - (-24(2 r^-13 - r^-7)) * r_hat
    # = 24(2 r^-13 - r^-7) * r_vec / r
    # = 24(2 r^-14 - r^-8) * r_vec
    # r^-14 = (r^2)^-7 = dist_sq^-7
    # r^-8 = (r^2)^-4 = dist_sq^-4
    coeff = 24.0 * (2.0 * inv_r2**7 - inv_r2**4)
    # coeff shape (N, N)
    # Force on i is sum over j of coeff[i,j] * (pos[i] - pos[j])
    # diff[i,j] = pos[i] - pos[j]
    forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    # Mask diagonal (force on self is 0)
    # coeff diagonal is large, but diff diagonal is 0, so product is 0.
    # However, 0 * inf might be nan if not careful.
    # We set dist_sq to 1e-10, so inv_r2 is 1e10.
    # coeff ~ 1e21. diff ~ 0. Product ~ 0.
    # It should be fine.
    return forces

def construct():
    start_time = time.time()
    deadline = 100.0 # 100s limit to be safe
    
    # 1. Generate FCC points
    a = 2**(2/3) # Lattice constant for equilibrium distance 2^(1/6)
    
    points = []
    # Iterate over integer coordinates in the FCC basis
    # Basis vectors:
    # b1 = (0.5, 0.5, 0)
    # b2 = (0.5, 0, 0.5)
    # b3 = (0, 0.5, 0.5)
    # Point = n1*b1 + n2*b2 + n3*b3
    # x = 0.5*(n1+n2)
    # y = 0.5*(n1+n3)
    # z = 0.5*(n2+n3)
    
    R = 5 # Search radius in lattice indices
    for n1 in range(-R, R+1):
        for n2 in range(-R, R+1):
            for n3 in range(-R, R+1):
                x = 0.5 * (n1 + n2)
                y = 0.5 * (n1 + n3)
                z = 0.5 * (n2 + n3)
                # Scale by a
                points.append((x*a, y*a, z*a))
                
    # Convert to numpy array
    points = np.array(points)
    
    # Remove duplicates (due to symmetry in n1,n2,n3 mapping? No, mapping is 1-to-1 for integers)
    # But we might generate same point from different n?
    # n1+n2 = 2x, n1+n3=2y, n2+n3=2z
    # 2n1 = 2x - 2y + 2z = 2(x-y+z) => n1 = x-y+z
    # Since x,y,z are half-integers or integers, x-y+z is integer.
    # So mapping is unique.
    
    # Sort by distance to origin
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    
    # Take first N
    pos0 = points[:N].copy()
    
    # 2. Optimize
    # Use L-BFGS-B to keep atoms reasonably close? No, just L-BFGS.
    # The structure is rigid enough, but local relaxation helps.
    
    def fun(x):
        pos = x.reshape(N, 3)
        return lj_energy(pos)
        
    def jac(x):
        pos = x.reshape(N, 3)
        return lj_forces(pos).flatten()
        
    try:
        res = minimize(fun, pos0.flatten(), jac=jac, method='L-BFGS-B', 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        pos_opt = res.x.reshape(N, 3)
    except Exception as e:
        pos_opt = pos0
        
    # Check time
    if time.time() - start_time > deadline:
         # Return best found so far (pos0 or pos_opt if it finished)
         pass
         
    return pos_opt
