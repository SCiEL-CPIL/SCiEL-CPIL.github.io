"""
Strategy: Geometric Primal-Dual / Force-Directed Relaxation.
1. Fix positions. Solve for max radii such that circles don't overlap or leave box.
   This is a linear program (LP) or can be approximated by a simple iterative relaxation.
   Actually, maximizing sum of radii subject to d_ij >= r_i + r_j and r_i <= dist_to_wall_i is not strictly LP because of the sum in the objective, but the constraints are linear in r_i if we fix positions? No, d_ij is fixed, so r_i + r_j <= d_ij is linear. r_i <= x_i is linear. So yes, it's an LP.
   However, solving an LP for every position step is expensive.
   Alternative: "Greedy Radius Assignment".
   For fixed positions, we can compute the maximum possible radius for each circle independently as min(dist_to_wall, min_j (d_ij - r_j)). This is cyclic.
   Better: Use a simple iterative scheme:
   r_i^{k+1} = min( dist_to_wall_i, min_{j != i} (d_ij - r_j^k) )
   This is not guaranteed to converge to the global max for fixed positions, but it's a good heuristic.
   
   Actually, a simpler and very effective strategy for packings is:
   1. Start with a grid or random positions.
   2. Iteratively:
      a. For each circle, compute the "limiting" distance to other circles and walls.
      b. Adjust positions to reduce "stress" (overlaps) by moving circles apart.
      c. Adjust radii to fill available space.
      
   Let's try a different structural change: **Simulated Annealing with a specialized move set**.
   The current program uses L-BFGS-B on the whole vector. This is sensitive to initial conditions and penalty parameters.
   Simulated Annealing (SA) with:
   - Moves: Move a single circle, Resize a single circle, Swap two circles (if radii differ).
   - Objective: Sum of radii.
   - Constraint handling: Penalty for overlaps and out-of-bounds.
   - Cooling schedule: Geometric.
   
   Why SA? It explores the landscape more broadly and is less likely to get stuck in the specific "valleys" created by the penalty method's gradient flow. It can jump over barriers.
   
   To make it fast:
   - Pre-compute neighbors? No, N=21 is small, O(N^2) per move is cheap.
   - Use a "hard" constraint check: reject moves that cause overlaps (or penalize heavily).
   - Actually, a "soft" penalty is better for SA to allow exploration.
   
   Let's combine:
   1. Initialize with a good heuristic (e.g., grid).
   2. Run SA for a fixed number of steps or time budget.
   3. Polish the best found configuration with L-BFGS-B (using the SA result as start).
   
   This is a hybrid: Global Search (SA) + Local Refinement (L-BFGS-B).
   The current program only does Local Refinement from a few fixed starts.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_score = -1
    best_result = None
    
    # Perimeter 4 => W + H = 2
    # We will try a few aspect ratios.
    aspect_ratios = [1.0, 0.9, 1.1, 0.8, 1.25]
    
    # Pre-compute pairs for overlap constraints
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j))
    pairs = np.array(pairs)
    ii, jj = pairs[:, 0], pairs[:, 1]
    
    def get_score_and_validity(cx, cy, r, w, h):
        """
        Calculate score and check validity.
        Returns (score, is_valid, result_array).
        """
        # Check bounds
        if np.any(cx - r < -1e-6) or np.any(cx + r - w > 1e-6) or \
           np.any(cy - r < -1e-6) or np.any(cy + r - h > 1e-6):
            return -1, False, None
        
        # Check overlaps
        dx = cx[ii] - cx[jj]
        dy = cy[ii] - cy[jj]
        dist = np.sqrt(dx*dx + dy*dy)
        r_sum = r[ii] + r[jj]
        
        if np.any(dist < r_sum - 1e-6):
            return -1, False, None
            
        score = np.sum(r)
        result = np.column_stack((cx, cy, r))
        return score, True, result

    def polish(cx, cy, r, w, h):
        """
        Refine a valid configuration using L-BFGS-B.
        """
        x0 = np.concatenate([cx, cy, r])
        r_max = min(w, h) / 2.0
        bounds = [(0.0, w)] * N + [(0.0, h)] * N + [(0.0, r_max)] * N
        
        def objective_and_grad(x):
            cx_v = x[:N]
            cy_v = x[N:2*N]
            r_v = x[2*N:]
            
            obj = -np.sum(r_v)
            grad = np.zeros(3*N)
            grad[2*N:] = -1.0
            
            # Overlap penalty
            dx = cx_v[ii] - cx_v[jj]
            dy = cy_v[ii] - cy_v[jj]
            dist_sq = dx*dx + dy*dy
            dist_sq[dist_sq < 1e-12] = 1e-12
            dist = np.sqrt(dist_sq)
            
            r_sum = r_v[ii] + r_v[jj]
            
            viol = r_sum - dist
            viol_pos = np.maximum(0, viol)
            
            penalty = 100.0
            obj += penalty * np.sum(viol_pos**2)
            
            # Gradient
            active = viol_pos > 0
            if np.any(active):
                ii_act = ii[active]
                jj_act = jj[active]
                viol_act = viol_pos[active]
                d_act = dist[active]
                
                factor = -2.0 * penalty * viol_act / d_act
                dx_act = cx_v[ii_act] - cx_v[jj_act]
                dy_act = cy_v[ii_act] - cy_v[jj_act]
                
                np.add.at(grad, ii_act, factor * dx_act)
                np.add.at(grad, jj_act, -factor * dx_act)
                np.add.at(grad, N + ii_act, factor * dy_act)
                np.add.at(grad, N + jj_act, -factor * dy_act)
                
                np.add.at(grad, 2*N + ii_act, 2.0 * penalty * viol_act)
                np.add.at(grad, 2*N + jj_act, 2.0 * penalty * viol_act)
            
            return obj, grad
        
        try:
            res = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            cx_opt = res.x[:N]
            cy_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            
            score, valid, result = get_score_and_validity(cx_opt, cy_opt, r_opt, w, h)
            if valid and score > best_score:
                return score, result
            return -1, None
        except Exception:
            return -1, None

    def simulated_annealing(cx, cy, r, w, h, steps=5000):
        """
        Run SA to improve the configuration.
        """
        current_score, current_valid, _ = get_score_and_validity(cx, cy, r, w, h)
        if not current_valid:
            # If start is invalid, try to fix it or return None
            # For simplicity, if start is invalid, we skip SA and rely on polish from valid starts
            return -1, None
            
        best_local_score = current_score
        best_local_result = np.column_stack((cx, cy, r))
        
        T = 0.05 # Initial temperature
        cooling = 0.995
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Choose a move
            move_type = np.random.choice(['move', 'resize', 'swap'])
            i = np.random.randint(N)
            
            cx_new = cx.copy()
            cy_new = cy.copy()
            r_new = r.copy()
            
            if move_type == 'move':
                # Move circle i by small random amount
                delta = np.random.normal(0, T, 2)
                cx_new[i] += delta[0]
                cy_new[i] += delta[1]
            elif move_type == 'resize':
                # Resize circle i
                delta = np.random.normal(0, T/10)
                r_new[i] = max(1e-4, r_new[i] + delta)
            else:
                # Swap positions of i and j (keeps radii with indices? No, usually swap circles)
                j = np.random.randint(N)
                if i == j:
                    continue
                cx_new[i], cx_new[j] = cx_new[j], cx_new[i]
                cy_new[i], cy_new[j] = cy_new[j], cy_new[i]
                # Radii stay with the "slots"? Or move with the circle?
                # If we swap circles, the radii should move too if we consider them part of the circle identity.
                # But in packing, we just want a set of circles.
                # Let's swap the entire circle data.
                r_new[i], r_new[j] = r_new[j], r_new[i]
            
            # Check validity and score
            new_score, new_valid, new_result = get_score_and_validity(cx_new, cy_new, r_new, w, h)
            
            # Penalty for invalidity
            if not new_valid:
                # Calculate penalty magnitude
                # Overlaps
                dx = cx_new[ii] - cx_new[jj]
                dy = cy_new[ii] - cy_new[jj]
                dist = np.sqrt(dx*dx + dy*dy)
                r_sum = r_new[ii] + r_new[jj]
                overlap_penalty = np.sum(np.maximum(0, r_sum - dist)**2)
                
                # Bounds
                bound_pen = np.sum(np.maximum(0, cx_new - r_new)**2) + \
                            np.sum(np.maximum(0, cx_new + r_new - w)**2) + \
                            np.sum(np.maximum(0, cy_new - r_new)**2) + \
                            np.sum(np.maximum(0, cy_new + r_new - h)**2)
                
                total_penalty = 10 * (overlap_penalty + bound_pen)
                new_score = -total_penalty
            
            # Acceptance
            if new_score > current_score:
                accept = True
            else:
                delta_e = current_score - new_score
                if T > 1e-6:
                    accept = np.random.rand() < math.exp(-delta_e / T)
                else:
                    accept = False
                    
            if accept:
                cx, cy, r = cx_new, cy_new, r_new
                current_score = new_score
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_result = np.column_stack((cx, cy, r))
            
            T *= cooling
            
        return best_local_score, best_local_result

    # Main Loop
    for w in aspect_ratios:
        if time.time() > deadline - 10:
            break
        h = 2.0 - w
        if h <= 0:
            continue
            
        # Initialize with a grid
        # 5x5 grid
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) * (w / side)
                cy = (j + 0.5) * (h / side)
                r = 0.5 * min(w, h) / side * 0.9
                cells.append((cx, cy, r))
        
        cx_init = np.array([c[0] for c in cells[:N]])
        cy_init = np.array([c[1] for c in cells[:N]])
        r_init = np.array([c[2] for c in cells[:N]])
        
        # 1. Polish the grid
        score, res = polish(cx_init, cy_init, r_init, w, h)
        if score > best_score:
            best_score = score
            best_result = res
            
        # 2. Run SA from the grid (or from the polished result if valid)
        if time.time() > deadline - 5:
            continue
            
        # Use the polished result as start for SA if it's valid, else grid
        sa_cx, sa_cy, sa_r = cx_init, cy_init, r_init
        if best_result is not None and best_score > 0:
             # Try to use the best result found so far? 
             # No, SA should explore. Let's start SA from the grid again but with random jitter
             # to avoid getting stuck in the same local optimum as polish.
             cx_sa = cx_init + np.random.normal(0, 0.01, N)
             cy_sa = cy_init + np.random.normal(0, 0.01, N)
             r_sa = r_init.copy()
        else:
             cx_sa = cx_init
             cy_sa = cy_init
             r_sa = r_init
            
        sa_score, sa_res = simulated_annealing(cx_sa, cy_sa, r_sa, w, h, steps=3000)
        
        if sa_score > best_score and sa_res is not None:
            best_score = sa_score
            best_result = sa_res
            
        # 3. Polish the SA result
        if sa_res is not None and time.time() < deadline - 5:
            sa_cx = sa_res[:, 0]
            sa_cy = sa_res[:, 1]
            sa_r = sa_res[:, 2]
            score2, res2 = polish(sa_cx, sa_cy, sa_r, w, h)
            if score2 > best_score:
                best_score = score2
                best_result = res2

    # Fallback
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append((cx, cy, r))
        best_result = np.array(cells[:N])
        
    return best_result
