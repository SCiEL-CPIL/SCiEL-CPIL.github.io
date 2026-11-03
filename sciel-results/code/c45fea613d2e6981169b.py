# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    N = 26

    def is_valid(x, tol=1e-8):
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        for i in range(N):
            if xs[i] - rs[i] < -tol or xs[i] + rs[i] > 1 + tol:
                return False
            if ys[i] - rs[i] < -tol or ys[i] + rs[i] > 1 + tol:
                return False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j] - tol:
                    return False
        return True

    def compute_hard_radii(centers):
        """Compute exact maximum radii for given centers."""
        centers = np.asarray(centers)
        n = len(centers)
        radii = np.zeros(n)
        
        for i in range(n):
            radii[i] = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                max_r = dist / 2.0
                if max_r < radii[i]:
                    radii[i] = max_r
                if max_r < radii[j]:
                    radii[j] = max_r
        return radii

    def initial_grid_centers():
        side = int(np.ceil(np.sqrt(N)))
        centers = []
        for i in range(side):
            for j in range(side):
                if len(centers) < N:
                    centers.append([(i + 0.5) / side, (j + 0.5) / side])
        return np.array(centers[:N])

    def random_centers(seed=0):
        rng = np.random.RandomState(seed)
        centers = rng.uniform(0.1, 0.9, size=(N, 2))
        return centers

    best_result = None
    best_score = -np.inf

    def solve_with_active_set(c_init, max_iters=50):
        """
        Solve packing using SLSQP with an adaptive active set of constraints.
        Variables: x = [c_1, c_2, ..., c_N, r_1, r_2, ..., r_N]
        """
        c = c_init.copy()
        r = compute_hard_radii(c)
        x = np.concatenate([c.flatten(), r])
        
        # Active set: list of pairs (i, j) that are constrained
        # We start with Delaunay edges
        active_pairs = set()
        
        # Initialize active set with Delaunay edges
        try:
            tri = Delaunay(c)
            for e in tri.edges:
                active_pairs.add(tuple(sorted(e)))
        except Exception:
            pass

        # Boundary constraints are always active
        # We will encode boundary constraints in the 'bounds' of minimize
        
        # Objective: Maximize sum(r) -> Minimize -sum(r)
        def objective(x):
            # r is the second half of x
            r_vec = x[N*2:]
            return -np.sum(r_vec)

        # Constraints: r_i + r_j <= ||c_i - c_j|| for active pairs
        # In SLSQP, constraints are of the form fun(x) >= 0
        # So: ||c_i - c_j|| - r_i - r_j >= 0
        
        def make_constraints(active_pairs):
            cons = []
            for (i, j) in active_pairs:
                def con(x, i=i, j=j):
                    ci = x[i*2:i*2+2]
                    cj = x[j*2:j*2+2]
                    ri = x[N*2 + i]
                    rj = x[N*2 + j]
                    dist = np.linalg.norm(ci - cj)
                    return dist - ri - rj
                cons.append({'type': 'ineq', 'fun': con})
            return cons

        # Bounds:
        # c_i in [0, 1]
        # r_i in [0, 0.5] (max possible radius in unit square)
        bounds = []
        for i in range(N):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
        for i in range(N):
            bounds.append((0, 0.5)) # r

        # However, we also have boundary constraints:
        # r_i <= c_ix, r_i <= 1-c_ix, etc.
        # It is better to handle these as inequality constraints or just clip and rely on the fact that r is limited by neighbors.
        # Actually, if we don't constrain r to boundary, the optimizer might push c to 0 and r to 0.5? No, if c=0, r must be 0 to satisfy r <= c_ix.
        # So we MUST include boundary constraints.
        
        def make_boundary_constraints():
            bcons = []
            for i in range(N):
                # r_i <= c_ix
                def con_left(x, i=i):
                    return x[i*2] - x[N*2+i]
                bcons.append({'type': 'ineq', 'fun': con_left})
                
                # r_i <= 1 - c_ix
                def con_right(x, i=i):
                    return (1.0 - x[i*2]) - x[N*2+i]
                bcons.append({'type': 'ineq', 'fun': con_right})
                
                # r_i <= c_iy
                def con_bottom(x, i=i):
                    return x[i*2+1] - x[N*2+i]
                bcons.append({'type': 'ineq', 'fun': con_bottom})
                
                # r_i <= 1 - c_iy
                def con_top(x, i=i):
                    return (1.0 - x[i*2+1]) - x[N*2+i]
                bcons.append({'type': 'ineq', 'fun': con_top})
            return bcons

        current_x = x
        for iter_idx in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            cons = make_constraints(active_pairs) + make_boundary_constraints()
            
            try:
                res = minimize(objective, current_x, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 100, 'ftol': 1e-8})
            except Exception:
                break

            if not res.success:
                # If SLSQP fails, maybe relax or break
                # Try to extract x if it's better
                if res.fun < objective(current_x):
                    current_x = res.x
                else:
                    break

            c_new = res.x[:N*2].reshape(N, 2)
            r_new = res.x[N*2:]
            
            # Check validity
            final_x = np.column_stack([c_new, r_new])
            if is_valid(final_x, tol=1e-6):
                return final_x

            # Update active set
            # Add any violated pairs to active set
            # We check all pairs to find violations
            for i in range(N):
                for j in range(i+1, N):
                    if (i, j) in active_pairs:
                        continue
                    dist = np.linalg.norm(c_new[i] - c_new[j])
                    if dist < r_new[i] + r_new[j] - 1e-8:
                        active_pairs.add((i, j))
            
            # Also, remove pairs that are no longer close? 
            # For simplicity, we only add. This ensures we never miss a constraint.
            # The number of constraints will grow to at most N(N-1)/2, which is 325.
            # SLSQP with 325 constraints and 52 variables might be slow.
            # Optimization: Only keep pairs that are "tight" (dist < r_i + r_j + eps) or in Delaunay.
            # Let's refine active set:
            new_active = set()
            # Always keep Delaunay
            try:
                tri = Delaunay(c_new)
                for e in tri.edges:
                    new_active.add(tuple(sorted(e)))
            except Exception:
                pass
            
            # Keep previously active pairs that are still somewhat close
            for (i, j) in active_pairs:
                dist = np.linalg.norm(c_new[i] - c_new[j])
                # If they are close, keep them
                if dist < r_new[i] + r_new[j] + 0.05:
                    new_active.add((i, j))
            
            active_pairs = new_active
            
            current_x = res.x

        # Return best found so far in this call
        c_final = current_x[:N*2].reshape(N, 2)
        r_final = current_x[N*2:]
        final_x = np.column_stack([c_final, r_final])
        if is_valid(final_x, tol=1e-6):
            return final_x
        return None

    # Multiple restarts
    seeds = [0, 1, 7, 13, 23, 37, 53, 71, 89, 42, 99, 123, 101, 113]
    
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        
        if seed == 0:
            centers = initial_grid_centers()
        else:
            centers = random_centers(seed)
        
        result = solve_with_active_set(centers, max_iters=30)
        
        if result is not None:
            score = np.sum(result[:, 2])
            if score > best_score:
                best_score = score
                best_result = result

    # Fallback if no valid result found (should not happen with grid)
    if best_result is None:
        centers = initial_grid_centers()
        radii = compute_hard_radii(centers)
        best_result = np.column_stack([centers, radii])
        
    return best_result

# EVOLVE-BLOCK-END
