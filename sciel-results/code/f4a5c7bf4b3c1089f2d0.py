import numpy as np
import time
import sys

try:
    import science_chemistry
    HAS_RELAX = True
except ImportError:
    HAS_RELAX = False

    N = 38

    def _lj_energy_and_grad(pos):
        # pos: (N, 3)
        # Returns energy, gradient
        dists = np.linalg.norm(pos[:, np.newaxis] - pos[np.newaxis, :], axis=2)
        # Avoid division by zero
        mask = dists < 1e-6
        dists[mask] = 1e-6
        
        inv_r2 = 1.0 / (dists ** 2)
        inv_r6 = inv_r2 ** 3
        inv_r12 = inv_r6 ** 2
        
        # Energy: 4 * sum(r^-12 - r^-6) over i<j
        # The matrix has diagonal 0 (masked to large? No, we set dists to 1e-6 so r^-12 is huge).
        # Better to compute off-diagonal sum.
        # Symmetric matrix. Sum is 2 * sum(i<j).
        # E = 2 * 4 * sum(i<j) (r^-12 - r^-6) ? No.
        # Pair potential is 4(r^-12 - r^-6).
        # Total E = sum_{i<j} 4(r_ij^-12 - r_ij^-6).
        
        # Using the full matrix sum:
        # Sum_all = sum_{i != j} 4(r_ij^-12 - r_ij^-6)
        # Sum_all = 2 * Sum_{i<j}
        # So E = 0.5 * Sum_all (excluding diagonal which is 0 energy if we handle it).
        
        # Let's compute the matrix term:
        # term = 4 * (inv_r12 - inv_r6)
        # Set diagonal to 0.
        term = 4.0 * (inv_r12 - inv_r6)
        np.fill_diagonal(term, 0.0)
        
        energy = 0.5 * np.sum(term)
        
        # Gradient:
        # F_i = -dE/dx_i
        # dE/dx_i = sum_{j != i} d/dx_i [4(r_ij^-12 - r_ij^-6)]
        # d/dx_i r_ij^-12 = -12 r_ij^-14 (x_i - x_j)
        # d/dx_i r_ij^-6 = -6 r_ij^-8 (x_i - x_j)
        # d/dx_i E_ij = 4 (-12 r^-14 (x_i-x_j) + 6 r^-8 (x_i-x_j))
        #              = 4 (x_i-x_j) (-12 r^-14 + 6 r^-8)
        #              = 24 (x_i-x_j) (r^-8 - 2 r^-14)
        
        # Vectorized:
        # diff = pos[:, np.newaxis] - pos[np.newaxis, :]  # (N, N, 3)
        # coeff = 24 * (inv_r8 - 2 * inv_r14) ... need inv_r8 and inv_r14
        # inv_r8 = inv_r2 ** 4
        # inv_r14 = inv_r2 ** 7
        
        # Actually, let's use the force formula directly.
        # F_ij = 48 (r^-14 - 0.5 r^-8) * r_vec ?
        # Standard LJ force: F = 48 (r^-14 - 0.5 r^-8) * r_vec / r ?
        # Let's stick to:
        # grad_i = sum_j 24 (x_i - x_j) (r_ij^-8 - 2 r_ij^-14)
        
        # We have inv_r6 and inv_r12.
        # inv_r8 = inv_r2 * inv_r6
        # inv_r14 = inv_r2 * inv_r12
        
        inv_r8 = inv_r2 * inv_r6
        inv_r14 = inv_r2 * inv_r12
        
        coeff = 24.0 * (inv_r8 - 2.0 * inv_r14)
        np.fill_diagonal(coeff, 0.0)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        
        # grad = sum over j of coeff[i,j] * (pos[i] - pos[j])
        # grad[i] = sum_j coeff[i,j] * diff[i,j]
        grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        return energy, grad

    def construct():
        start_time = time.time()
        deadline = start_time + 110.0 # 10s safety margin
        
        # 1. Generate FCC initial guess
        a = 2.0 ** (2.0/3.0) # Lattice constant for LJ min distance
        # FCC basis
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5]
        ])
        
        points = []
        # Generate a 4x4x4 block of unit cells to ensure we have > 38 atoms
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for b in basis:
                        p = np.array([i, j, k]) + b
                        points.append(p)
        
        points = np.array(points)
        # Center the points
        center = np.mean(points, axis=0)
        points = points - center
        
        # Scale by lattice constant
        points = points * a
        
        # Sort by distance to center and take N
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)
        init_pos = points[idx[:N]]
        
        # 2. Relax
        best_pos = init_pos
        best_energy = float('inf')
        
        if HAS_RELAX:
            try:
                # science_chemistry.relax returns relaxed positions? Or (pos, energy)?
                # Prompt says: "science_chemistry.relax(positions) ... is available".
                # Usually returns positions. I'll assume it returns positions.
                relaxed = science_chemistry.relax(init_pos)
                # Calculate energy to verify
                e, _ = _lj_energy_and_grad(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed
            except Exception:
                pass # Fall through to scipy if it fails
        
        if not HAS_RELAX or best_energy == float('inf'):
            # Use SciPy L-BFGS-B
            from scipy.optimize import minimize
            
            # Flatten positions
            x0 = init_pos.flatten()
            
            def fun(x):
                pos = x.reshape(N, 3)
                e, g = _lj_energy_and_grad(pos)
                return e, g.flatten()
            
            # Set deadline check in callback?
            # L-BFGS-B might take a while.
            # We'll just run it. If it's too slow, we return the initial guess.
            # But with a good start, it should be fast.
            
            try:
                res = minimize(fun, x0, jac=True, method='L-BFGS-B', options={'maxiter': 200, 'gtol': 1e-10})
                if res.success or res.fun < best_energy:
                    best_pos = res.x.reshape(N, 3)
                    best_energy = res.fun
            except Exception:
                pass
        
        # Check time
        if time.time() > deadline:
            return best_pos
        
        return best_pos
