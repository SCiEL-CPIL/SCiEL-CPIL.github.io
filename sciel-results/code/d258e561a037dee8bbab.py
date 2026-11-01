import numpy as np
from scipy.optimize import minimize
import time

N = 55

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit

    # 1. Analytic Construction of Mackay Icosahedron
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12)
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize vertices to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Calculate edge midpoints (30)
    # An edge exists if the distance between two vertices is the minimum non-zero distance.
    # For a regular icosahedron, all edges have the same length.
    dists = np.linalg.norm(verts_unit[:, np.newaxis] - verts_unit[np.newaxis, :], axis=2)
    # The minimum distance is the edge length
    min_dist = np.min(dists[dists > 1e-6])
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                mid = (verts_unit[i] + verts_unit[j]) / 2
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-6:
                    mid_unit = mid / mid_norm
                    edges.append(mid_unit)
    
    edges_unit = np.array(edges)
    
    # We expect 30 edges
    if len(edges_unit) != 30:
        # Fallback: If edge detection fails, just use vertices scaled for outer shell too (less optimal but valid)
        # But it should work.
        pass

    # Initial radii
    # LJ minimum is at r = 2^(1/6) * sigma = 1.122
    # Inner shell radius
    r_in = 1.1
    # Outer shell radius
    r_out = 2.2
    
    # Build positions
    # Center atom
    pos = np.array([[0.0, 0.0, 0.0]])
    
    # Inner shell: 12 atoms at verts_unit * r_in
    pos = np.vstack([pos, verts_unit * r_in])
    
    # Outer shell: 12 atoms at verts_unit * r_out + 30 atoms at edges_unit * r_out
    pos = np.vstack([pos, verts_unit * r_out])
    pos = np.vstack([pos, edges_unit * r_out])
    
    # Ensure we have exactly 55 atoms
    # If we have more, truncate. If less, pad (should not happen).
    if pos.shape[0] > N:
        pos = pos[:N]
    elif pos.shape[0] < N:
        # Pad with random points on a sphere if needed (should not happen)
        extra = N - pos.shape[0]
        random_pts = np.random.randn(extra, 3)
        random_pts = random_pts / np.linalg.norm(random_pts, axis=1, keepdims=True) * r_out
        pos = np.vstack([pos, random_pts])
        
    # 2. Local Optimization (L-BFGS)
    def energy_and_grad(x):
        # Reshape x to (N, 3)
        x = x.reshape(N, 3)
        
        # Calculate distances
        # Use broadcasting for pair distances
        # To save memory and time, we can do it in chunks or just use the full matrix for N=55
        # N=55 is small enough for (55,55,3) array
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        r2 = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        r2[range(N), range(N)] = np.inf
        
        # LJ Energy: 4 * sum(r^-12 - r^-6)
        # r^-6 = r2^-3
        # r^-12 = r2^-6
        inv_r6 = r2 ** -3
        inv_r12 = inv_r6 ** 2
        
        # Energy is sum over i < j
        # The matrix is symmetric. Sum of all elements is 2 * sum(i<j)
        # So sum(i<j) = 0.5 * sum(all off-diagonal)
        E = 4.0 * 0.5 * np.sum(inv_r12 - inv_r6)
        
        # Gradient
        # dE/dr = 4 * (-12 r^-13 + 6 r^-7) = 24 r^-7 (1 - 2 r^-6)
        # dE/dx_i = sum_{j != i} dE/dr * dr/dx_i
        # r = |x_i - x_j|
        # dr/dx_i = (x_i - x_j) / r
        # dE/dx_i = sum_{j != i} 24 r^-7 (1 - 2 r^-6) * (x_i - x_j) / r
        #          = sum_{j != i} 24 r^-8 (1 - 2 r^-6) * (x_i - x_j)
        
        # Let's compute the term 24 * r^-8 * (1 - 2*r^-6) for each pair
        # r^-8 = r2^-4
        # r^-6 = r2^-3
        coeff = 24.0 * (r2 ** -4) * (1.0 - 2.0 * (r2 ** -3))
        
        # coeff is (N, N)
        # grad_i = sum_j coeff[i, j] * (x_i - x_j)
        # grad = coeff @ (x_i - x_j matrix?)
        # Let D = x[:, np.newaxis] - x[np.newaxis, :] (N, N, 3)
        # grad[i] = sum_j coeff[i, j] * D[i, j]
        
        # We can use einsum or matrix multiplication
        # coeff is (N, N), D is (N, N, 3)
        # grad = np.einsum('ij,ijk->k', coeff, D) ? No, sum over j for each i.
        # grad[i] = sum_j coeff[i, j] * D[i, j]
        # This is like (coeff * D).sum(axis=1)
        
        # Note: D[i, j] = x[i] - x[j]
        # coeff[i, j] is the force magnitude factor.
        
        # Force on i from j is coeff[i, j] * (x[i] - x[j])
        # Total force on i is sum_j coeff[i, j] * (x[i] - x[j])
        
        # Gradient of Energy with respect to x is the Negative of Force?
        # E = sum V(r). F = -grad E.
        # dE/dx_i = -F_i.
        # Let's check signs.
        # E = 4(r^-12 - r^-6).
        # dE/dr = 4(-12r^-13 + 6r^-7) = -24r^-13 + 24r^-7 = 24r^-7(1 - 2r^-6).
        # dE/dx_i = sum_j (dE/dr) * (x_i - x_j)/r
        #          = sum_j 24r^-8(1 - 2r^-6) (x_i - x_j)
        # This is what we calculated in `coeff` (which is 24r^-8(1-2r^-6)).
        # So grad = sum_j coeff[i,j] * (x_i - x_j).
        
        # Calculation:
        # weighted_diff = coeff[:, :, np.newaxis] * diff
        # grad = np.sum(weighted_diff, axis=1)
        
        weighted_diff = coeff[:, :, np.newaxis] * diff
        grad = np.sum(weighted_diff, axis=1)
        
        return E, grad.ravel()

    # Initial guess
    x0 = pos.ravel()
    
    # Optimize
    best_x = x0
    best_E = energy_and_grad(x0)[0]
    
    try:
        res = minimize(
            energy_and_grad, 
            x0, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        if res.success or res.fun < best_E:
            best_x = res.x
            best_E = res.fun
            
        # Check time
        if time.time() > deadline - 10:
            return best_x.reshape(N, 3)
            
        # Second polish pass with tighter tolerance
        res2 = minimize(
            energy_and_grad, 
            best_x, 
            method='L-BFGS-B', 
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10, 'disp': False}
        )
        
        if res2.success or res2.fun < best_E:
            best_x = res2.x
            best_E = res2.fun
            
    except Exception:
        # If optimization fails, return the analytic construction
        pass

    return best_x.reshape(N, 3)
