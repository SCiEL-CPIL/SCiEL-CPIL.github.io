# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
import itertools

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 to maximize
    the sum of their side lengths.
    
    Strategy:
    1. Enumerate discrete grid topologies (R rows, C cols).
    2. For each topology, try different assignments of squares to cells.
    3. For a fixed assignment, optimize the row heights and column widths 
       to maximize the sum of min(h_i, w_j).
    4. Return the best feasible configuration found.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 610  # Leave 20s margin for safety

    # Pre-defined topologies: (rows, cols)
    # We need to cover 10 cells.
    # 1x10, 2x5, 3x4 (one empty), 4x3 (one empty), 5x2, 10x1
    # Also 3x3 (9 cells, one square must be split? No, squares are atomic).
    # So we need topologies with exactly 10 cells or we handle "empty" cells.
    # Let's use topologies that can hold 10 squares in a grid-like structure.
    # 2x5: 10 cells.
    # 5x2: 10 cells.
    # 1x10: 10 cells.
    # 10x1: 10 cells.
    # 3x4: 12 cells, use 10.
    # 4x3: 12 cells, use 10.
    
    topologies = [
        (2, 5),
        (5, 2),
        (1, 10),
        (10, 1),
        (3, 4),
        (4, 3),
        (3, 3), # 9 cells, we can't fit 10. Skip.
        (2, 4), # 8 cells. Skip.
    ]
    
    # Filter valid topologies that have at least 10 cells
    valid_topologies = [(r, c) for r, c in topologies if r * c >= 10]

    best_centers = None
    best_sides = None
    best_score = -np.inf

    # We will try a few random permutations of square indices for each topology
    # to break symmetry.
    rng = np.random.default_rng(42)
    
    # Square sizes are not fixed in advance in the general problem, 
    # but in the grid model, s_ij = min(h_i, w_j).
    # The "squares" are determined by the grid dimensions.
    
    for r, c in valid_topologies:
        if time.time() > deadline:
            break
            
        num_cells = r * c
        # We need to pick 10 cells out of num_cells if num_cells > 10
        # For simplicity, let's assume we use the first 10 cells in row-major order
        # and leave the rest empty (or merge them? No, just empty).
        # Actually, for 3x4, we have 12 slots. We use 10.
        
        # Try a few random permutations of which "logical squares" go into which "cells"
        # Since all squares are identical in the grid model (size determined by cell),
        # the permutation doesn't change the *geometry* of the grid, but it might
        # if we were assigning fixed sizes. Here, sizes are derived.
        # So for a fixed (r,c), the optimal grid dimensions are unique (up to symmetry).
        # The "permutation" aspect is less relevant for pure grid optimization 
        # unless we consider non-grid packings.
        
        # Let's optimize the grid dimensions for this topology.
        # Variables: h_0..h_{r-1}, w_0..w_{c-1}
        # Constraints: sum(h) = 1, sum(w) = 1, h_i >= 0, w_j >= 0.
        # Objective: Maximize sum_{i,j in used_cells} min(h_i, w_j)
        
        # To handle "used cells", we define a mask.
        # For 2x5, all 10 are used.
        # For 3x4, we use 10. Let's use the first 10 in row-major order.
        mask = np.zeros((r, c), dtype=bool)
        mask_flat = mask.flatten()
        mask_flat[:10] = True
        mask = mask_flat.reshape(r, c)
        
        def grid_objective(x):
            # x = [h_0..h_{r-1}, w_0..w_{c-1}]
            h = x[:r]
            w = x[r:r+c]
            
            # Ensure non-negative (handled by bounds)
            # Calculate sum of min(h_i, w_j) for used cells
            total = 0.0
            for i in range(r):
                for j in range(c):
                    if mask[i, j]:
                        total += min(h[i], w[j])
            return -total # Minimize negative sum
        
        # Initial guess: uniform
        x0 = np.concatenate([np.full(r, 1.0/r), np.full(c, 1.0/c)])
        
        bounds = [(0.0, 1.0) for _ in range(r + c)]
        
        try:
            res = minimize(grid_objective, x0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.success or res.fun < 0: # Check if we got a valid negative value (positive sum)
                x_opt = res.x
                h = x_opt[:r]
                w = x_opt[r:r+c]
                
                # Normalize to ensure sum is exactly 1 (L-BFGS-B might drift slightly)
                if np.sum(h) > 0: h = h / np.sum(h)
                if np.sum(w) > 0: w = w / np.sum(w)
                
                # Construct centers and sides
                centers = []
                sides = []
                current_y = 0.0
                for i in range(r):
                    current_x = 0.0
                    for j in range(c):
                        if mask[i, j]:
                            s = min(h[i], w[j])
                            # Center of this square
                            cx = current_x + w[j] / 2.0
                            cy = current_y + h[i] / 2.0
                            centers.append([cx, cy])
                            sides.append(s)
                        current_x += w[j]
                    current_y += h[i]
                
                # Check if we have exactly 10 squares
                if len(centers) != 10:
                    continue
                    
                centers = np.array(centers)
                sides = np.array(sides)
                
                # Validate
                if hard_feasibility(centers, sides):
                    score = np.sum(sides)
                    if score > best_score:
                        best_score = score
                        best_centers = centers
                        best_sides = sides
                        
        except Exception:
            pass

    # If no grid solution found (unlikely), fallback to simple grid
    if best_centers is None:
        k1, k2 = 2, 5
        side = 0.2
        cx, cy = np.meshgrid(np.linspace(side/2, 1-side/2, k2), np.linspace(side/2, 1-side/2, k1))
        best_centers = np.column_stack((cx.flatten(), cy.flatten()))
        best_sides = np.full(n, side)
        best_score = np.sum(best_sides)

    return best_centers, best_sides

def hard_feasibility(centers, sides):
    """Strict check for the final candidate."""
    n = len(centers)
    tol = 1e-7
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i][0] - centers[j][0]) < t and abs(centers[i][1] - centers[j][1]) < t:
                return False
    return True

# EVOLVE-BLOCK-END
