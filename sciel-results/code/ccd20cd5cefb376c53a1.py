import numpy as np
import time

def squares_in_square10() -> tuple:
    """
    Maximizes the total side length of 10 non-overlapping axis-aligned squares
    inside the unit square [0, 1]^2 using Hierarchical Block Packing and
    Feasibility-Preserving Coordinate Descent.
    """
    n = 10
    start_time = time.time()
    hard_deadline = start_time + 600.0 - 15.0  # 15s safety margin

    # --- 1. Deterministic Hierarchical Block Packing Initialization ---
    # Known optimal structure for 10 squares is often 2 large + 8 small.
    # Layout:
    # Left half (width 0.5):
    #   Top: One large square (side 0.5) at (0.25, 0.75)
    #   Bottom: 2x2 grid of 4 small squares (side 0.25) in [0, 0.5]x[0, 0.5]
    # Right half (width 0.5):
    #   Top: One large square (side 0.5) at (0.75, 0.75)
    #   Bottom: 2x2 grid of 4 small squares (side 0.25) in [0.5, 1.0]x[0, 0.5]
    
    cx = np.array([
        0.25, # Large 1
        0.75, # Large 2
        0.125, 0.375, 0.125, 0.375, # Small 1-4 (Left Bottom 2x2)
        0.625, 0.875, 0.625, 0.875  # Small 5-8 (Right Bottom 2x2)
    ])
    cy = np.array([
        0.75, # Large 1
        0.75, # Large 2
        0.125, 0.125, 0.375, 0.375, # Small 1-4
        0.125, 0.125, 0.375, 0.375  # Small 5-8
    ])
    s = np.array([
        0.5, 0.5,
        0.25, 0.25, 0.25, 0.25,
        0.25, 0.25, 0.25, 0.25
    ])

    # --- 2. Feasibility-Preserving Coordinate Descent ---
    # We will iterate over each square and each coordinate (cx, cy, s).
    # For each coordinate, we try to move it in the direction that increases
    # the total side length sum, while ensuring the new configuration remains
    # strictly feasible (within bounds and non-overlapping).
    
    # To make "increasing total side length" the primary driver, we can treat
    # expanding side length 's' as the most valuable move, followed by moving
    # centers to allow expansion. However, a simpler robust approach is:
    # 1. Try to expand each square's side length 's' as much as possible without
    #    violating constraints, while keeping centers fixed.
    # 2. If a square is blocked, try moving its center to create space for expansion.
    
    # Let's implement a generic coordinate descent that evaluates the "score"
    # as sum(sides) and rejects any move that makes the configuration infeasible.
    # Since we start feasible, and only accept feasible moves, we stay feasible.
    # The objective is simply to maximize sum(s).
    
    tol = 1e-7

    def is_feasible(cx, cy, s):
        # Check bounds
        if np.any(cx - s/2 < -tol) or np.any(cx + s/2 > 1 + tol):
            return False
        if np.any(cy - s/2 < -tol) or np.any(cy + s/2 > 1 + tol):
            return False
        if np.any(s <= 0):
            return False
            
        # Check overlaps
        # Vectorized check
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = np.abs(cx[i_idx] - cx[j_idx])
        dy = np.abs(cy[i_idx] - cy[j_idx])
        sep = (s[i_idx] + s[j_idx]) / 2.0 - tol
        
        # Violation if dx < sep AND dy < sep
        if np.any((dx < sep) & (dy < sep)):
            return False
            
        return True

    def get_score(cx, cy, s):
        return np.sum(s)

    # Optimization Loop
    # We will perform multiple passes. In each pass, we try to improve each square.
    # For square i, we try:
    # 1. Increase s_i
    # 2. Decrease s_i (if it allows others to grow later? No, usually we want to grow)
    # 3. Move cx_i
    # 4. Move cy_i
    
    # A more effective strategy for packing is "expansion" followed by "shifting".
    # Let's try to expand each square to its maximum possible size given the current
    # positions of others. Then, if stuck, try moving a square to a "hole".
    
    # Simpler robust CD:
    # For each square i:
    #   - Try increasing s_i by small steps. If feasible, keep it.
    #   - Try moving cx_i left/right. If feasible AND allows s_i to increase (or just if it's part of a swap?), keep it.
    #     Actually, just moving cx_i doesn't increase score directly. It only helps if it enables expansion.
    #     So, a good heuristic:
    #     1. For each square, maximize s_i given fixed centers.
    #     2. For each square, try moving centers to positions that allow larger s_i.
    
    # Let's implement a specific heuristic:
    # 1. Maximize sizes given centers.
    # 2. For each square, try moving it to one of the "corners" or "edges" of the available space
    #    defined by other squares, then re-maximize sizes.
    
    # To keep it simple and fast, let's do:
    # 1. Expand all squares to max feasible size (iteratively, as expanding one might block another, 
    #    but usually expanding is monotonic in feasibility if we check carefully? No, expanding one 
    #    can make it infeasible with another. So we must check feasibility of the WHOLE system).
    
    # Correct approach:
    # We maintain the state (cx, cy, s).
    # We try a move: change one parameter of one square.
    # If the new state is feasible, accept it.
    # We prioritize moves that increase sum(s).
    
    # Since moving centers doesn't change sum(s), we only accept center moves if they are 
    # "preparatory" for expansion. To simplify, we can alternate:
    # Phase A: Expand sides.
    # Phase B: Move centers to "tight" spots (e.g., against boundaries or other squares) to allow further expansion.
    
    # Let's use a simple random-restart CD with time limit.
    
    best_cx, best_cy, best_s = cx.copy(), cy.copy(), s.copy()
    best_score = get_score(best_cx, best_cy, best_s)
    
    # Helper to check if a specific square i can be expanded to s_new given others
    def max_feasible_size(i, cx, cy, s):
        # Binary search for max s_i
        lo = s[i]
        hi = 1.0 # Max possible side
        # Actually, max side is limited by distance to boundaries and other squares.
        # But binary search on [0, 1] is safe.
        for _ in range(20):
            mid = (lo + hi) / 2
            s_test = s.copy()
            s_test[i] = mid
            if is_feasible(cx, cy, s_test):
                lo = mid
            else:
                hi = mid
        return lo

    # Main loop
    iterations = 0
    while time.time() < hard_deadline:
        iterations += 1
        improved = False
        
        # 1. Try to expand each square
        for i in range(n):
            if time.time() > hard_deadline:
                break
            new_s_i = max_feasible_size(i, cx, cy, s)
            if new_s_i > s[i] + 1e-9:
                s[i] = new_s_i
                improved = True
                score = get_score(cx, cy, s)
                if score > best_score:
                    best_score = score
                    best_cx, best_cy, best_s = cx.copy(), cy.copy(), s.copy()
        
        if not improved:
            # 2. Try to move centers to enable expansion
            # For each square, try moving it to "snap" against boundaries or other squares
            # This is complex. Instead, let's try random perturbations of centers
            # and then re-expand.
            
            # Pick a random square
            i = np.random.randint(n)
            # Try moving it slightly
            dx = np.random.uniform(-0.1, 0.1)
            dy = np.random.uniform(-0.1, 0.1)
            
            cx_new = cx.copy()
            cy_new = cy.copy()
            cx_new[i] += dx
            cy_new[i] += dy
            
            # Check if this move is feasible (with current sizes)
            if is_feasible(cx_new, cy_new, s):
                # Accept the move? Only if it leads to improvement after re-expansion.
                # So, we tentatively accept, re-expand, and check score.
                cx_tent, cy_tent, s_tent = cx_new.copy(), cy_new.copy(), s.copy()
                
                # Re-expand all squares
                for j in range(n):
                    new_s_j = max_feasible_size(j, cx_tent, cy_tent, s_tent)
                    s_tent[j] = new_s_j
                    
                score_tent = get_score(cx_tent, cy_tent, s_tent)
                if score_tent > best_score + 1e-9:
                    cx, cy, s = cx_tent, cy_tent, s_tent
                    best_score = score_tent
                    best_cx, best_cy, best_s = cx.copy(), cy.copy(), s.copy()
                    improved = True
            # If not feasible, we could try to shrink s[i] to make it feasible, but that reduces score.
            # So we skip.
            
        if not improved and iterations > 100:
            # Stagnation: try a more aggressive restart or perturbation
            # Perturb all centers slightly
            cx += np.random.uniform(-0.05, 0.05, n)
            cy += np.random.uniform(-0.05, 0.05, n)
            # Clamp to bounds roughly
            cx = np.clip(cx, 0.0, 1.0)
            cy = np.clip(cy, 0.0, 1.0)
            
            # Re-expand
            for j in range(n):
                new_s_j = max_feasible_size(j, cx, cy, s)
                s[j] = new_s_j
                
            score = get_score(cx, cy, s)
            if score > best_score:
                best_score = score
                best_cx, best_cy, best_s = cx.copy(), cy.copy(), s.copy()
            else:
                # Revert to best if perturbation failed
                cx, cy, s = best_cx.copy(), best_cy.copy(), best_s.copy()
                
    # Final check and return
    if not is_feasible(best_cx, best_cy, best_s):
        # Fallback to initial valid solution
        best_cx = np.array([0.25, 0.75, 0.125, 0.375, 0.125, 0.375, 0.625, 0.875, 0.625, 0.875])
        best_cy = np.array([0.75, 0.75, 0.125, 0.125, 0.375, 0.375, 0.125, 0.125, 0.375, 0.375])
        best_s = np.array([0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])

    centers = np.column_stack((best_cx, best_cy))
    sides = best_s
    
    return centers, sides

# EVOLVE-BLOCK-END
