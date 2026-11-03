import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.optimize import least_squares, minimize
import sys

N = 50
DEADLINE = 145.0

def _true_score(centers):
    """Compute the true packing radius for a given configuration."""
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5):
        return 0.0
    boundary = (0.5 - np.abs(centers)).min()
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist = np.sqrt(dists_sq.min())
    return min(boundary, min_dist / 2.0)

def _get_contact_graph(centers, r):
    """
    Identify the active contact graph.
    Returns:
    - pairs: list of (i, j) pairs where dist(i,j) is close to 2r
    - boundary_points: list of (i, k) where dist to boundary k is close to r
    """
    pairs = []
    boundary_points = []
    
    # Check pairs
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 2.0 * r + 1e-4:
                pairs.append((i, j))
                
    # Check boundaries
    # Square is [-0.5, 0.5]^2
    # dist to left: x + 0.5
    # dist to right: 0.5 - x
    # dist to bottom: y + 0.5
    # dist to top: 0.5 - y
    for i in range(N):
        x, y = centers[i]
        if x + 0.5 < r + 1e-4: boundary_points.append((i, 0))
        if 0.5 - x < r + 1e-4: boundary_points.append((i, 1))
        if y + 0.5 < r + 1e-4: boundary_points.append((i, 2))
        if 0.5 - y < r + 1e-4: boundary_points.append((i, 3))
        
    return pairs, boundary_points

def _solve_contact_system(centers, r, pairs, boundary_points, max_iter=50):
    """
    Solve for the new positions and radius such that all active constraints are tight.
    Variables: x (N*2 coords) + r (1 radius) = 101 variables.
    Constraints: 
    1. For each pair (i,j): ||c_i - c_j||^2 - (2r)^2 = 0
    2. For each boundary point (i, k): dist(c_i, boundary_k) - r = 0
    
    We use least_squares to minimize the sum of squares of these residuals.
    This is much more robust than SLSQP when the system is nearly determined by the graph.
    """
    n_vars = 2 * N + 1
    n_constraints = len(pairs) + len(boundary_points)
    
    if n_constraints == 0:
        return centers, r, False

    def residuals(x):
        c = x[:2*N].reshape(N, 2)
        t = x[-1]
        res = np.zeros(n_constraints)
        
        idx = 0
        for i, j in pairs:
            d_sq = np.sum((c[i] - c[j])**2)
            res[idx] = d_sq - 4.0 * t * t
            idx += 1
            
        for i, k in boundary_points:
            if k == 0: # left
                res[idx] = (c[i, 0] + 0.5) - t
            elif k == 1: # right
                res[idx] = (0.5 - c[i, 0]) - t
            elif k == 2: # bottom
                res[idx] = (c[i, 1] + 0.5) - t
            else: # top
                res[idx] = (0.5 - c[i, 1]) - t
            idx += 1
            
        return res

    x0 = np.concatenate([centers.ravel(), [r]])
    
    try:
        result = least_squares(
            residuals, 
            x0, 
            method='lm', # Levenberg-Marquardt is fast for well-behaved small systems
            xtol=1e-12, 
            ftol=1e-12, 
            gtol=1e-12
        )
        
        if result.success or result.cost < 1e-10:
            new_centers = result.x[:2*N].reshape(N, 2)
            new_r = result.x[-1]
            
            # Check validity: no overlaps, inside square
            if np.any(np.abs(new_centers) > 0.5 + 1e-9):
                return centers, r, False
                
            # Verify no new violations (distances < 2r)
            # If the solver moved points such that non-active pairs now overlap, it's invalid
            diffs = new_centers[:, np.newaxis, :] - new_centers[np.newaxis, :, :]
            dists_sq = (diffs ** 2).sum(axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            min_dist = np.sqrt(dists_sq.min())
            
            if min_dist < 2.0 * new_r - 1e-9:
                return centers, r, False
            
            # Verify boundary
            b_dist = (0.5 - np.abs(new_centers)).min()
            if b_dist < new_r - 1e-9:
                return centers, r, False
                
            # If valid, return the improved (or maintained) configuration
            # Note: The solver might find a solution with slightly larger r if the graph was under-constrained
            # or slightly smaller if it relaxed. We take the true score.
            true_r = _true_score(new_centers)
            if true_r >= r - 1e-9:
                return new_centers, true_r, True
    except Exception:
        pass
        
    return centers, r, False

def _init_hexagonal():
    """
    Generate a good hexagonal packing initialization.
    """
    best_pts = None
    best_r = -1
    
    # Try different scales and offsets
    for s in [0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22]:
        pts = []
        row_height = s * np.sqrt(3) / 2
        for i in range(-6, 7):
            for j in range(-6, 7):
                x = j * s + (i % 2) * s / 2
                y = i * row_height
                if abs(x) <= 0.49 and abs(y) <= 0.49:
                    pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) < N:
            continue
            
        # Select N points greedily: farthest point sampling
        selected = []
        remaining = pts.copy()
        
        center_dists = np.linalg.norm(remaining, axis=1)
        idx = np.argmin(center_dists)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
        
        for _ in range(N - 1):
            if len(remaining) == 0:
                break
            sel = np.array(selected)
            dists = np.min(np.linalg.norm(remaining[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
            idx = np.argmax(dists)
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
            
        if len(selected) == N:
            pts_sel = np.array(selected)
            r = _true_score(pts_sel)
            if r > best_r:
                best_r = r
                best_pts = pts_sel
                
    if best_pts is None:
        rows, cols = 7, 8
        xs = (np.arange(cols) + 0.5) / cols - 0.5
        ys = (np.arange(rows) + 0.5) / rows - 0.5
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]
        
    return best_pts

def _slsqp_polish(centers):
    """
    Standard SLSQP polish to ensure we are at a local optimum of the epigraph formulation.
    Used as a fallback or initial step.
    """
    def objective(x):
        return -x[-1]
    
    def gradient(x):
        g = np.zeros_like(x)
        g[-1] = -1.0
        return g
        
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    
    def constraints(x):
        c = x[:2*N].reshape(N, 2)
        t = x[-1]
        cons_b = 0.5 - np.abs(c) - t
        diffs = c[iu[0], :] - c[iu[1], :]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        cons_p = dists - 2 * t
        return np.concatenate([cons_b.ravel(), cons_p])
        
    def jacobian(x):
        c = x[:2*N].reshape(N, 2)
        t = x[-1]
        J = np.zeros((2*N + n_pairs, 2*N + 1))
        
        for i in range(N):
            for k in range(2):
                idx = 2*i + k
                s = np.sign(c[i, k])
                if s == 0: s = 1.0
                J[idx, idx] = -s
                J[idx, -1] = -1.0
        
        diffs = c[iu[0], :] - c[iu[1], :]
        dists = np.sqrt((diffs ** 2).sum(axis=1))
        dists_safe = np.where(dists > 1e-10, dists, 1e-10)
        inv_dists = 1.0 / dists_safe
        
        for p in range(n_pairs):
            i = iu[0][p]
            j = iu[1][p]
            d = diffs[p]
            inv_d = inv_dists[p]
            row = 2*N + p
            J[row, 2*i] = d[0] * inv_d
            J[row, 2*i+1] = d[1] * inv_d
            J[row, 2*j] = -d[0] * inv_d
            J[row, 2*j+1] = -d[1] * inv_d
            J[row, -1] = -2.0
        return J
        
    t0 = _true_score(centers)
    x0 = np.concatenate([centers.ravel(), [t0]])
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            jac=gradient,
            bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': constraints, 'jac': jacobian}],
            options={'maxiter': 100, 'ftol': 1e-10, 'disp': False}
        )
        if result.success:
            new_centers = result.x[:2*N].reshape(N, 2)
            true_t = _true_score(new_centers)
            if true_t >= t0 - 1e-9:
                return new_centers
    except Exception:
        pass
    return centers

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # 1. Initialization
    centers = _init_hexagonal()
    best_centers = centers.copy()
    best_score = _true_score(best_centers)
    
    # 2. Initial SLSQP Polish to get a good local optimum and define the graph
    elapsed = time.time() - start_time
    if time.time() - start_time < DEADLINE - 20:
        centers = _slsqp_polish(centers)
        score = _true_score(centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()

    # 3. Topology-Driven Refinement Loop
    # We iteratively:
    # a) Identify the contact graph for the current best radius
    # b) Solve the tight system for that graph
    # c) If the solution is valid and improves radius, accept
    # d) Repeat with a slightly perturbed graph or if stuck, use SA to jump
    
    for _ in range(50): # Max iterations
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        r = best_score
        pairs, boundary_points = _get_contact_graph(best_centers, r)
        
        # If graph is empty or too small, it's not a rigid structure, skip or expand
        if len(pairs) + len(boundary_points) < 10:
             # Try to expand radius slightly to find contacts? 
             # Or just do a small random move
             break
             
        new_centers, new_r, success = _solve_contact_system(best_centers, r, pairs, boundary_points)
        
        if success and new_r > best_score + 1e-12:
            best_score = new_r
            best_centers = new_centers.copy()
            
            # If we improved, we might need to re-polish with SLSQP to fix any slight numerical drift
            # or to allow the graph to change naturally
            if time.time() - start_time < DEADLINE - 5:
                 polished = _slsqp_polish(best_centers)
                 ps = _true_score(polished)
                 if ps > best_score:
                     best_score = ps
                     best_centers = polished.copy()
        
        # If we didn't improve, or we are stuck, we can try a "Graph Perturbation"
        # Remove a random active constraint and re-solve? Or add a nearby non-active constraint?
        # For now, let's try a small random kick to escape the local graph minimum
        # This is a cheap move compared to full SA
        if time.time() - start_time < DEADLINE - 5:
            # Random kick to one point
            i = np.random.randint(N)
            kick = np.random.randn(2) * 0.005
            perturbed = best_centers.copy()
            perturbed[i] += kick
            perturbed[i] = np.clip(perturbed[i], -0.5, 0.5)
            
            # Polish the perturbed version
            p_polished = _slsqp_polish(perturbed)
            p_score = _true_score(p_polished)
            
            if p_score > best_score:
                best_score = p_score
                best_centers = p_polished.copy()
                
    # 4. Final Polish
    if time.time() - start_time < DEADLINE - 2:
        final = _slsqp_polish(best_centers)
        final_score = _true_score(final)
        if final_score > best_score:
            best_score = final_score
            best_centers = final.copy()

    best_centers = np.clip(best_centers, -0.5, 0.5)
    return best_centers

if __name__ == "__main__":
    t0 = time.time()
    c = construct()
    print(f"Time: {time.time()-t0:.2f}s")
    print(f"Score: {_true_score(c)}")
    print(f"Shape: {c.shape}")
