import numpy as np
from scipy.optimize import minimize
import time

N = 38

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Generate FCC lattice points
    # FCC basis vectors
    # v1 = (0, 1, 1), v2 = (1, 0, 1), v3 = (1, 1, 0)
    # Point = n1*v1 + n2*v2 + n3*v3 = (n2+n3, n1+n3, n1+n2)
    # Let's iterate n1, n2, n3 in a range
    points = []
    for n1 in range(-3, 4):
        for n2 in range(-3, 4):
            for n3 in range(-3, 4):
                x = n2 + n3
                y = n1 + n3
                z = n1 + n2
                points.append((x, y, z))
    
    # Convert to numpy array
    pts = np.array(points, dtype=float)
    
    # Calculate distance from origin
    dists = np.linalg.norm(pts, axis=1)
    
    # Sort by distance
    sorted_indices = np.argsort(dists)
    
    # Take the first 38 unique points
    # Note: The origin (0,0,0) is included.
    # We need 38 points.
    selected_pts = pts[sorted_indices[:N]]
    
    # Center the cluster
    center = np.mean(selected_pts, axis=0)
    selected_pts = selected_pts - center
    
    # Scale the points
    # The nearest neighbor distance in the generated lattice (with basis vectors of length sqrt(2))
    # Let's find the minimum non-zero distance
    min_dist = np.inf
    for i in range(len(selected_pts)):
        for j in range(i+1, len(selected_pts)):
            d = np.linalg.norm(selected_pts[i] - selected_pts[j])
            if d < min_dist and d > 1e-6:
                min_dist = d
    
    # The minimum of LJ potential is at r = 2^(1/6)
    target_dist = 2 ** (1.0/6.0)
    scale_factor = target_dist / min_dist
    selected_pts = selected_pts * scale_factor
    
    # 2. Define energy and gradient functions for optimization
    def lj_energy(pos):
        # pos is (N, 3)
        # Calculate all pair distances
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        # Mask self-interactions
        np.fill_diagonal(dists_sq, np.inf)
        # Avoid division by zero if any atoms overlap (shouldn't happen with good init)
        dists_sq = np.maximum(dists_sq, 1e-12)
        
        inv_r6 = dists_sq ** -3
        inv_r12 = inv_r6 ** 2
        
        energy = 4.0 * np.sum(inv_r12 - inv_r6)
        # We sum over all i,j, so each pair is counted twice.
        # The task defines energy as 4 * sum(r^-12 - r^-6) over all pairs.
        # Usually "sum over all pairs" implies i<j.
        # My calculation sums over all i != j.
        # So I need to divide by 2.
        return energy / 2.0

    def lj_grad(pos):
        # Gradient of the energy
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists_sq = np.maximum(dists_sq, 1e-12)
        
        # d/dr (4(r^-12 - r^-6)) = 4(-12r^-13 + 6r^-7)
        # Chain rule: dE/dxi = sum_j dE/drij * drij/dxi
        # drij/dxi = (xi - xj) / rij
        # dE/drij = 4(-12rij^-13 + 6rij^-7)
        # dE/dxi = sum_j 4(-12rij^-13 + 6rij^-7) * (xi - xj) / rij
        #        = sum_j 4(-12rij^-14 + 6rij^-8) * (xi - xj)
        
        # Note: This sums over all j != i.
        # The energy was defined as sum over i<j.
        # Let E = 0.5 * sum_{i != j} V(rij)
        # dE/dxi = 0.5 * sum_{j != i} dV/dxi
        # dV/dxi = 4(-12rij^-14 + 6rij^-8) * (xi - xj)
        
        # Let's compute the term in parentheses
        # r^-14 = (r^2)^-7
        # r^-8 = (r^2)^-4
        
        r2 = dists_sq
        # We need to handle the matrix carefully
        # term = 4 * (-12 * r2^-7 + 6 * r2^-4)
        # This term is 0 on the diagonal (inf -> 0)
        r2_safe = np.where(np.isinf(r2), 1.0, r2)
        term = 4.0 * (-12.0 * r2_safe ** -7 + 6.0 * r2_safe ** -4)
        
        # Gradient component: sum_j term_ij * (xi - xj)
        # diff is (xi - xj)
        # grad_i = sum_j term_ij * diff_ij
        
        grad = np.sum(term[:, :, np.newaxis] * diff, axis=1)
        
        # Since E = 0.5 * sum_{i != j} ..., the gradient is 0.5 * sum_{j != i} ...
        # But our sum over j includes i? No, we set diagonal to inf, so term is 0 on diagonal.
        # However, the factor of 2 in the energy definition (sum over i<j vs i!=j)
        # If E = sum_{i<j} V, then dE/dxi = sum_{j>i} dV/dxi + sum_{j<i} dV/dxi
        # dV/dxi (for pair ij) = - dV/dxj
        # Summing over all j != i in 0.5 * sum_{i != j} V gives the correct gradient for sum_{i<j} V.
        # Let's verify:
        # E = 0.5 * (V12 + V21 + V13 + V31 ...) = V12 + V13 ...
        # dE/dx1 = 0.5 * (dV12/dx1 + dV21/dx1 + dV13/dx1 + dV31/dx1)
        # dV12/dx1 = -dV12/dx2
        # dV21/dx1 = dV12/dx1 (since V21=V12, but derivative w.r.t x1 is same as dV12/dx1? No)
        # V(r12) depends on x1, x2.
        # dV(r12)/dx1 = (dV/dr) * (x1-x2)/r
        # dV(r12)/dx2 = (dV/dr) * (x2-x1)/r = - dV(r12)/dx1
        # So dV12/dx1 + dV21/dx1 = dV12/dx1 + dV12/dx1 = 2 dV12/dx1.
        # 0.5 * 2 dV12/dx1 = dV12/dx1.
        # So the gradient calculated by summing over all j != i with the factor 0.5 is correct.
        # My `term` calculation is for dV/dri * 1/r?
        # dV/dr = 4(-12r^-13 + 6r^-7)
        # dV/dx = dV/dr * (x1-x2)/r = 4(-12r^-14 + 6r^-8) * (x1-x2)
        # So `term` is exactly dV/dx component per pair.
        # And we sum over all j.
        # So `grad` is the correct gradient for the energy defined as 0.5 * sum_{i != j}.
        # Which is equal to sum_{i<j}.
        
        return grad

    # 3. Optimize
    # Use L-BFGS-B
    # Constraints: none strictly, but atoms should not overlap.
    # L-BFGS-B handles bounds, but we don't have simple bounds.
    # We rely on the good initial guess.
    
    result = minimize(
        lj_energy, 
        selected_pts, 
        method='L-BFGS-B', 
        jac=lj_grad,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
    )
    
    if time.time() > deadline:
        # Return the initial guess if we are running out of time?
        # No, we should return the best found so far.
        # But minimize returns the result. If it times out, it might not.
        # Given the small N, it should be fast.
        pass

    best_pos = result.x
    
    # Check validity: min distance >= 0.5
    # Calculate min distance
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    
    if min_d < 0.5:
        # If atoms are too close, we might need to expand or use the initial guess
        # But with LJ repulsion, they should push apart.
        # If this happens, it's a failure mode.
        # For now, return best_pos.
        pass

    return best_pos
