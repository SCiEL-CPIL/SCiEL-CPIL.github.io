"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    """Pack N circles in unit square, maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Vectorized constraint penalty calculation
    # x: (N, 3) array of [cx, cy, r]
    # Returns: scalar penalty value (0 if valid)
    # We minimize -sum(r) + penalty
    # Penalty is sum of max(0, violation)^2 for all constraints
    
    def penalty_and_score(x):
        """
        Calculate penalty and score.
        x shape: (N, 3)
        Returns: (penalty, score)
        """
        # x: centers (N, 2), r: radii (N,)
        cx = x[:, 0]
        cy = x[:, 1]
        r = x[:, 2]
        
        score = np.sum(r)
        
        # Boundary penalties
        # We want cx - r >= 0, cx + r <= 1, etc.
        # Violation if cx - r < 0 -> max(0, r - cx)
        # Violation if cx + r > 1 -> max(0, cx + r - 1)
        # Same for y
        
        pen = 0.0
        
        # Lower bounds
        pen += np.sum(np.maximum(0, r - cx)**2)
        pen += np.sum(np.maximum(0, r - cy)**2)
        
        # Upper bounds
        pen += np.sum(np.maximum(0, cx + r - 1)**2)
        pen += np.sum(np.maximum(0, cy + r - 1)**2)
        
        # Overlap penalties
        # dist(i,j) >= r(i) + r(j)
        # violation = max(0, r(i) + r(j) - dist(i,j))
        
        # Vectorized distance calculation
        # cx_i - cx_j
        dx = cx[:, np.newaxis] - cx[np.newaxis, :]
        dy = cy[:, np.newaxis] - cy[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # r_i + r_j
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # We only need upper triangle to avoid double counting, but 
        # summing all is fine if we divide by 2 or just accept higher penalty weight.
        # Let's sum all i != j and divide by 2 to keep scale consistent.
        # Actually, max(0, violation) is 0 on diagonal.
        
        violations = np.maximum(0, r_sum - dist)
        
        # Exclude diagonal
        np.fill_diagonal(violations, 0)
        
        # Sum of squares
        pen += np.sum(violations**2)
        
        return pen, score

    def objective(x):
        """
        Objective function for minimization.
        We want to maximize score, so minimize -score + lambda * penalty.
        """
        pen, score = penalty_and_score(x)
        # Adaptive lambda? Fixed large lambda is safer for L-BFGS.
        # If penalty is large, it dominates.
        # We can use a barrier or just a large multiplier.
        # Let's use lambda = 10.0
        return -score + 10.0 * pen

    def bounds():
        """
        Define bounds for L-BFGS-B.
        cx, cy in [0, 1]
        r in [0, 0.5] (max radius in unit square is 0.5)
        """
        b = []
        for i in range(N):
            b.append((0, 1)) # cx
            b.append((0, 1)) # cy
            b.append((1e-6, 0.5)) # r
        return b

    def initial_guesses():
        """
        Generate diverse initial guesses.
        """
        guesses = []
        
        # 1. Grid
        side = int(np.ceil(np.sqrt(N)))
        # For N=26, side=6. 6x6 grid.
        # We take first N points.
        # To make it better, we can center the grid if N < side^2.
        # But simple grid is a good baseline.
        x_grid = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            x_grid[i, 0] = (col + 0.5) / side
            x_grid[i, 1] = (row + 0.5) / side
            # Initial radius: slightly less than half cell width
            x_grid[i, 2] = 0.5 / side * 0.9
        guesses.append(x_grid)
        
        # 2. Hexagonal lattice approximation
        # Hexagonal packing is denser.
        # For N=26, let's try to arrange in rows.
        # Row 1: 5 circles, Row 2: 5, Row 3: 5, Row 4: 5, Row 5: 5, Row 6: 1?
        # Or 5,5,5,5,4,2?
        # Let's just use a perturbed grid with hexagonal offsets.
        x_hex = x_grid.copy()
        for i in range(N):
            row = i // side
            col = i % side
            if row % 2 == 1:
                x_hex[i, 0] += 0.5 / side # Shift every other row
            # Keep radius same for now
        guesses.append(x_hex)
        
        # 3. Random
        np.random.seed(42)
        x_rand = np.random.uniform(0.1, 0.9, (N, 2))
        r_rand = np.random.uniform(0.05, 0.1, N)
        x_rand_full = np.hstack([x_rand, r_rand.reshape(-1, 1)])
        guesses.append(x_rand_full)
        
        # 4. Another random with different seed
        np.random.seed(123)
        x_rand2 = np.random.uniform(0.1, 0.9, (N, 2))
        r_rand2 = np.random.uniform(0.05, 0.1, N)
        x_rand2_full = np.hstack([x_rand2, r_rand2.reshape(-1, 1)])
        guesses.append(x_rand2_full)
        
        return guesses

    best_result = None
    best_score = -np.inf
    best_penalty = np.inf
    
    guesses = initial_guesses()
    
    for i, x0 in enumerate(guesses):
        if time.time() > deadline - 10:
            break
            
        # Run L-BFGS-B
        try:
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds(),
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
            )
            
            if res.success or res.fun < best_score: # Note: best_score is actually min objective
                # Calculate actual score and penalty for this result
                pen, score = penalty_and_score(res.x)
                
                # Check validity strictly
                # We need to ensure the solution is actually valid (penalty ~ 0)
                # If penalty is not close to 0, it's not a valid packing.
                # However, L-BFGS-B minimizes the combined objective.
                # If the penalty is large, the score will be low (due to -score + 10*pen).
                # So a low objective usually implies low penalty or high score.
                # But we want to maximize score subject to penalty=0.
                
                # Let's check if penalty is negligible.
                # If penalty > 1e-6, we might consider it invalid or try to repair.
                
                # For now, we track the best valid-looking solution.
                # "Valid-looking" means penalty is very small.
                
                if pen < 1e-5: # Threshold for validity
                    if score > best_score:
                        best_score = score
                        best_result = res.x
                        best_penalty = pen
                else:
                    # If penalty is not small, we might still keep it if it's the best "potential"
                    # but we should prioritize valid ones.
                    # Let's store the best objective value regardless, but only return valid ones.
                    pass
                    
        except Exception as e:
            pass # Continue to next guess
            
    # If no valid solution found, try to repair the best candidate
    if best_result is None:
        # Fallback: use the result with the lowest penalty, even if not strictly valid?
        # Or just return a grid?
        # Let's try to find the one with lowest penalty among all runs.
        # We didn't store all. Let's re-run the best guess if needed?
        # For now, if best_result is None, return a simple grid.
        x_grid = guesses[0]
        best_result = x_grid
        # Calculate score for reporting
        _, best_score = penalty_and_score(best_result)

    # Final check: ensure the result is strictly valid.
    # If not, we might need a local projection step to fix tiny violations.
    pen, score = penalty_and_score(best_result)
    if pen > 1e-8:
        # Attempt a simple projection to fix boundaries and overlaps
        # This is a fallback repair
        cx = best_result[:, 0].copy()
        cy = best_result[:, 1].copy()
        r = best_result[:, 2].copy()
        
        # Boundary clamp
        cx = np.clip(cx, r, 1 - r)
        cy = np.clip(cy, r, 1 - r)
        
        # Overlap resolution (simple iterative)
        for _ in range(50):
            moved = False
            for i in range(N):
                for j in range(i+1, N):
                    dx = cx[j] - cx[i]
                    dy = cy[j] - cy[i]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_dist = r[i] + r[j]
                    if dist < min_dist and dist > 1e-10:
                        overlap = min_dist - dist
                        push_x = (dx / dist) * overlap * 0.5
                        push_y = (dy / dist) * overlap * 0.5
                        cx[i] -= push_x
                        cy[i] -= push_y
                        cx[j] += push_x
                        cy[j] += push_y
                        moved = True
            # Re-clamp boundaries after moves
            cx = np.clip(cx, r, 1 - r)
            cy = np.clip(cy, r, 1 - r)
            if not moved:
                break
        
        best_result[:, 0] = cx
        best_result[:, 1] = cy
        best_result[:, 2] = r
        
        # Re-check
        pen, score = penalty_and_score(best_result)
        if pen > 1e-6:
            # Still invalid? Return grid as last resort
            best_result = guesses[0]
            _, score = penalty_and_score(best_result)
            best_score = score

    return best_result
# EVOLVE-BLOCK-END
