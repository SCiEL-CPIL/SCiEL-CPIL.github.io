import numpy as np
from scipy.optimize import minimize
from itertools import combinations
import time

N = 11

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
H = np.sqrt(3) / 2.0

def barycentric_to_cartesian(bary):
    """Convert (N,3) barycentric to (N,2) cartesian."""
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])

def min_area_from_bary(bary):
    """Compute minimum triangle area from barycentric coordinates."""
    cart = barycentric_to_cartesian(bary)
    p1 = cart[tri_idx[:, 0]]
    p2 = cart[tri_idx[:, 1]]
    p3 = cart[tri_idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def soft_min_area(bary, beta=100.0):
    """Smooth surrogate for min area using log-sum-exp."""
    cart = barycentric_to_cartesian(bary)
    p1 = cart[tri_idx[:, 0]]
    p2 = cart[tri_idx[:, 1]]
    p3 = cart[tri_idx[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def objective_soft(bary_flat, beta=100.0):
    """Objective to minimize (negative of soft min area)."""
    bary = bary_flat.reshape(N, 3)
    bary_clipped = np.clip(bary, 0, None)
    row_sums = bary_clipped.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_norm = bary_clipped / row_sums
    return -soft_min_area(bary_norm, beta)

def objective_hard(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 3)
    bary_clipped = np.clip(bary, 0, None)
    row_sums = bary_clipped.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_norm = bary_clipped / row_sums
    return -min_area_from_bary(bary_norm)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    best_bary = None
    best_val = -np.inf
    
    # Generate random barycentric points
    def random_bary():
        s = np.sqrt(rng.random(N))
        t = rng.random(N)
        a = 1 - s
        b = s * t
        c = s * (1 - t)
        bary = np.column_stack([a, b, c])
        return bary.flatten()
    
    # Initial random starts with local polish
    for trial in range(10):
        if time.time() > deadline - 40:
            break
        bary0 = random_bary()
        # Quick local polish
        try:
            result = minimize(
                objective_soft,
                bary0,
                method='SLSQP',
                bounds=[(0, 1)] * (N * 3),
                args=(50.0,),
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            val = -result.fun
            if val > best_val:
                best_val = val
                best_bary = result.x.copy()
        except Exception:
            val = -objective_soft(bary0, 50.0)
            if val > best_val:
                best_val = val
                best_bary = bary0.copy()
    
    # Critical triangle optimization: focus on k-smallest triangles
    if time.time() < deadline - 20 and best_bary is not None:
        # Get current critical triangles (k smallest)
        bary_curr = best_bary.reshape(N, 3)
        bary_curr = np.clip(bary_curr, 0, None)
        row_sums = bary_curr.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        bary_curr = bary_curr / row_sums
        
        cart = barycentric_to_cartesian(bary_curr)
        p1 = cart[tri_idx[:, 0]]
        p2 = cart[tri_idx[:, 1]]
        p3 = cart[tri_idx[:, 2]]
        cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
                (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
        areas = 0.5 * np.abs(cross)
        
        # Get k smallest triangles (k=10)
        k = 10
        k_smallest_idx = np.argsort(areas)[:k]
        critical_tri = tri_idx[k_smallest_idx]
        
        # Optimize only points involved in critical triangles
        # Identify which points are in critical triangles
        critical_points = set()
        for tri in critical_tri:
            critical_points.update(tri.tolist())
        critical_points = sorted(critical_points)
        
        if len(critical_points) < N:
            # Fix non-critical points, optimize critical ones
            fixed_mask = np.ones(N, dtype=bool)
            for p in critical_points:
                fixed_mask[p] = False
            
            # Extract fixed points
            fixed_bary = bary_curr[fixed_mask]
            
            # Objective that only varies critical points
            def obj_critical(crit_flat):
                bary = np.zeros((N, 3))
                bary[fixed_mask] = fixed_bary
                crit_idx = [i for i in range(N) if not fixed_mask[i]]
                bary[crit_idx] = crit_flat.reshape(len(crit_idx), 3)
                return objective_soft(bary.flatten(), 50.0)
            
            # Initial guess for critical points
            crit_initial = bary_curr[[i for i in range(N) if not fixed_mask[i]]].flatten()
            
            try:
                result = minimize(
                    obj_critical,
                    crit_initial,
                    method='SLSQP',
                    bounds=[(0, 1)] * (len(critical_points) * 3),
                    options={'maxiter': 200, 'ftol': 1e-12}
                )
                # Reconstruct full barycentric
                bary_new = np.zeros((N, 3))
                bary_new[fixed_mask] = fixed_bary
                crit_idx = [i for i in range(N) if not fixed_mask[i]]
                bary_new[crit_idx] = result.x.reshape(len(crit_idx), 3)
                
                val = -objective_hard(bary_new.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary_new.flatten().copy()
            except Exception:
                pass
    
    # Multiple SLSQP restarts with soft objective
    if time.time() < deadline - 15:
        slsqp_starts = [best_bary.copy()]
        for i in range(5):
            perturbed = best_bary + rng.normal(0, 0.03, size=best_bary.shape)
            slsqp_starts.append(perturbed)
        
        for s in slsqp_starts:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(
                    objective_soft,
                    s,
                    method='SLSQP',
                    bounds=[(0, 1)] * (N * 3),
                    args=(80.0,),
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                if -result.fun > best_val:
                    best_val = -result.fun
                    best_bary = result.x.copy()
            except Exception:
                pass
    
    # Final high-precision polish with hard min objective
    if time.time() < deadline - 5:
        try:
            result = minimize(
                objective_hard,
                best_bary,
                method='SLSQP',
                bounds=[(0, 1)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            if -result.fun > best_val:
                best_val = -result.fun
                best_bary = result.x.copy()
        except Exception:
            pass
        
        # Nelder-Mead polish
        if time.time() < deadline - 2:
            try:
                result = minimize(
                    objective_hard,
                    best_bary,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-14}
                )
                if -result.fun > best_val:
                    best_val = -result.fun
                    best_bary = result.x.copy()
            except Exception:
                pass
    
    # Convert to cartesian
    bary_final = best_bary.reshape(N, 3)
    bary_final = np.clip(bary_final, 0, None)
    row_sums = bary_final.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    bary_final = bary_final / row_sums
    cart = barycentric_to_cartesian(bary_final)
    
    return cart
