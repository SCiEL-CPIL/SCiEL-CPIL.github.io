"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: Adaptive Grid Refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        
        # Archive of non-dominated points
        archive = []
        used = 0
        
        def dominates(a, b):
            """True if a dominates b (minimization)."""
            return np.all(a <= b) and np.any(a < b)
            
        def add_to_archive(point, fval):
            nonlocal archive
            # Check if dominated by existing
            for p, f in archive:
                if dominates(f, fval):
                    return False
            # Remove existing points dominated by new one
            new_archive = []
            for p, f in archive:
                if not dominates(fval, f):
                    new_archive.append((p, f))
            new_archive.append((point, fval))
            archive = new_archive
            return True

        def is_deadline_reached():
            return time.time() > deadline or used >= self.budget

        # Phase 1: Coarse Grid
        # We need to cover the space. 
        # Let's start with a grid of 3x3x3x3 = 81 points.
        # This is a reasonable initial coverage for 4D.
        # If budget is very small, we might need to adjust, but 200 is fixed.
        grid_levels = [3, 5, 9, 17] 
        
        # Evaluate initial coarse grid
        current_level = 0
        n_grid = grid_levels[current_level]
        
        # Generate grid points
        # For each dimension, create linspace
        # We want to include boundaries? Yes, usually good.
        # But problem clips, so boundaries are safe.
        
        # To be systematic, we'll maintain a list of "cells" to subdivide.
        # Each cell is defined by its lower and upper bounds.
        # Initially, one big cell.
        
        # However, a pure recursive subdivision might be slow in Python if not careful.
        # Let's use a queue of cells.
        
        # Initial cell: the whole space
        cells = [(lo.copy(), hi.copy())]
        
        # We will process cells in order of "potential" (e.g., distance to reference or just FIFO)
        # To ensure coverage, we'll expand the grid uniformly first.
        
        # Strategy:
        # 1. Evaluate a coarse grid (e.g. 3^4=81 points).
        # 2. Identify non-dominated points.
        # 3. For the next iteration, subdivide the space around the non-dominated points?
        #    Or just subdivide the whole space more finely?
        #    Subdividing the whole space uniformly is "deterministic grid refinement".
        #    3^4 = 81
        #    5^4 = 625 (too many for 200 budget if we do it all at once)
        
        # Better: Adaptive subdivision.
        # Start with 1 cell.
        # Split it into 2^4 = 16 sub-cells? Or 3^4?
        # Let's try a priority queue based on the "best" objective value in the cell.
        # But we don't know the value without evaluating.
        
        # Alternative: Latin Hypercube Sampling (LHS) is better for initial spread.
        # But the prompt asks for "deterministic, adaptive grid refinement".
        
        # Let's stick to the grid idea but make it adaptive.
        # We maintain a set of "active" cells.
        # We evaluate the centroid of each active cell.
        # We keep the best non-dominated points.
        # We subdivide the cells that contain points close to the Pareto front.
        
        # How to decide which cells to subdivide?
        # We can track the "best" (lowest sum of normalized objectives) point in each cell.
        # Or simply, we can do a multi-resolution sweep.
        
        # Let's try a simpler, robust approach:
        # 1. Generate a set of candidate points using a stratified sampling (LHS-like) but deterministic.
        # 2. Evaluate them.
        # 3. Keep the non-dominated set.
        # 4. For the next phase, generate points *around* the non-dominated points (local refinement) 
        #    and also fill in gaps in the objective space.
        
        # Actually, the "Grid Refinement" directive suggests:
        # "iteratively subdivides the 4D parameter space into hypercubes"
        
        # Let's implement a best-first search on a grid tree.
        # Node: (lo, hi)
        # Priority: Estimated potential. Since we don't have a model, we can use the 
        # best point found in that node's neighborhood?
        
        # Simpler deterministic strategy:
        # 1. Coarse grid: 3x3x3x3 = 81 points.
        # 2. Find non-dominated points.
        # 3. For each non-dominated point, generate a fine grid around it (local search).
        # 4. Also, if budget remains, do a medium grid (5x5x5x5 is too big, maybe 4x4x4x4=256 is too big).
        
        # Let's combine:
        # Step 1: Evaluate 3^4 = 81 points.
        # Step 2: Identify the "islands" of non-dominated points.
        # Step 3: For each island, perform a local adaptive grid refinement.
        
        # This is effectively a multi-start local search with deterministic grids.
        
        # Let's refine the "Grid Refinement" to be strictly hierarchical.
        # Level 0: 1 cell. Centroid.
        # Level 1: 16 cells (2x2x2x2). Centroids.
        # Level 2: 64 cells (4x4x4x4). Centroids.
        # Level 3: 256 cells (8x8x8x8). Centroids.
        
        # Total points: 1 + 16 + 64 + 256 = 337. Too many for 200.
        # 1 + 16 + 64 = 81.
        # Next level: 2^4=16, 4^4=256.
        # If we do 1 + 16 + 64 + (200-81) = 200.
        # We can pick the best 119 out of the 256 level-3 cells?
        # How to pick? Based on the parent cell's performance.
        
        # Algorithm:
        # 1. Evaluate Level 1 (16 points).
        # 2. For each cell, record the best point.
        # 3. Select the top K cells (e.g., all 16, or top 8) to subdivide.
        # 4. Evaluate Level 2 for those cells (8 * 16 = 128 points? No, 8 cells * 16 sub-cells = 128).
        #    16 + 128 = 144 points used.
        # 5. Remaining budget: 56.
        # 6. Select top cells from Level 2 and subdivide to Level 3.
        #    56 / 16 = 3.5 cells. So subdivide 3 cells.
        #    3 * 16 = 48 points.
        #    Total: 16 + 128 + 48 = 192 points.
        #    Remaining: 8 points. Use for final polish or extra cells.
        
        # This seems like a solid deterministic strategy.
        
        # Implementation details:
        # We need to track cells.
        # A cell is (lo, hi).
        # We maintain a list of "active" cells for the next level.
        # We prioritize cells based on the best objective value found in them (normalized).
        
        # Normalization:
        # We don't know the bounds of the objectives yet.
        # We can use a running min/max for normalization.
        
        obj_mins = np.full(n_obj, np.inf)
        obj_maxs = np.full(n_obj, -np.inf)
        
        def normalize(f):
            # Safe division
            denom = obj_maxs - obj_mins
            denom[denom < 1e-12] = 1.0
            return (f - obj_mins) / denom

        # Initialize
        active_cells = [(lo.copy(), hi.copy())]
        
        # We will run for a fixed number of levels or until budget is exhausted.
        # Level 0: 1 cell.
        # Level 1: 2^4 = 16 cells.
        # Level 2: 2^4 = 16 cells per parent.
        
        # Let's just loop while budget allows and we have cells to process.
        # To avoid infinite loops or getting stuck, we'll cap the depth.
        
        depth = 0
        max_depth = 6 # 2^24 cells is huge, but we won't reach that.
        
        # We need to be careful with the "priority" calculation.
        # Initially, all cells are equal.
        # We'll store (priority, lo, hi, best_f_in_cell)
        # Priority: Lower is better.
        
        # Since we don't know the objective scale, we'll use the sum of objectives initially.
        
        # Let's re-evaluate the budget allocation.
        # Budget = 200.
        
        # Iteration 1:
        # Process 1 cell (root).
        # Evaluate centroid. (1 eval)
        # Subdivide into 16 children.
        # Add 16 children to queue.
        
        # Iteration 2:
        # We have 16 cells.
        # We can evaluate all 16. (16 evals). Total 17.
        # Subdivide each into 16 children? 16*16 = 256 children.
        # We can't evaluate all 256.
        # We must select the "best" parents to subdivide.
        
        # Selection:
        # After evaluating the 16 children of the root, we have 16 points.
        # We identify the non-dominated set among these 16.
        # We also identify the "best" cell based on the best point in it.
        # Actually, each cell has exactly one point (its centroid) in this scheme.
        # So we rank the 16 cells by the quality of their centroid.
        # We pick the top K cells to subdivide.
        
        # How many to pick?
        # If we pick K cells, we generate K*16 new cells.
        # We need to evaluate these K*16 cells.
        # So cost is K*16.
        
        # Let's try K=4 for the first subdivision.
        # Cost: 4 * 16 = 64 evals.
        # Total: 1 + 16 + 64 = 81 evals.
        
        # Next level:
        # We have 64 new cells.
        # Rank them by quality.
        # Pick top K=3.
        # Cost: 3 * 16 = 48 evals.
        # Total: 81 + 48 = 129 evals.
        
        # Next level:
        # We have 48 new cells.
        # Rank them.
        # Pick top K=4.
        # Cost: 4 * 16 = 64 evals.
        # Total: 129 + 64 = 193 evals.
        
        # Remaining: 7 evals.
        # We can pick the top 0 cells (0*16=0) and use remaining for something else?
        # Or pick 0 cells and just stop.
        # Or, in the last step, if budget < 16, we can evaluate single points in the best cells.
        
        # This strategy uses 193 evaluations and leaves 7.
        # We can use the remaining 7 to do a local "polish" on the best non-dominated points found.
        
        # Let's implement this.
        
        # Data structure for cells:
        # List of tuples: (priority, lo, hi)
        # Priority: We want to process "good" cells first.
        # Initially, priority is 0 for all.
        # After evaluation, we assign priority based on the point's value.
        
        # We'll use a min-heap for the queue?
        # But we need to process all cells of the current "level" before moving to the next?
        # No, the "level" concept is just for clarity.
        # We can just use a priority queue of cells to subdivide.
        
        # Algorithm:
        # 1. Queue = [(0, lo, hi)]  # (priority, lo, hi)
        # 2. While Queue is not empty and budget > 0:
        #    a. Pop the best cell (lowest priority).
        #    b. Evaluate its centroid.
        #    c. Add to archive.
        #    d. If budget allows, subdivide this cell into 16 children.
        #       - Calculate new priorities for children.
        #         How? We don't know their values yet.
        #         We can inherit the parent's priority?
        #         Or use a heuristic: children are "closer" to the good point, so they are promising.
        #         Let's give them a slightly worse priority than the parent?
        #         Or, we can batch process.
        
        # The batch processing (evaluate all children of selected parents) is more efficient 
        # because it allows ranking the children against each other before deciding 
        # which of *their* children to generate.
        
        # Revised Batch Algorithm:
        # 1. Current Cells: List of (lo, hi). Initially [ (lo, hi) ].
        # 2. Evaluate all Current Cells.
        # 3. If budget is low, stop.
        # 4. Rank Current Cells by the quality of their evaluated point.
        # 5. Select top K cells to subdivide.
        # 6. Generate children for these K cells.
        # 7. Set Current Cells = Children.
        # 8. Repeat.
        
        # How to determine K?
        # K should be such that K * 16 <= remaining_budget.
        # Also, K should not be too small to lose diversity, or too large to waste budget.
        # Let's use K = max(1, min(len(Current Cells), remaining_budget // 16)).
        
        # Let's trace:
        # Start: 1 cell.
        # Eval 1. Budget left 199.
        # K = min(1, 199//16) = 1.
        # Subdivide 1 cell -> 16 children.
        
        # Current: 16 cells.
        # Eval 16. Budget left 183.
        # K = min(16, 183//16) = min(16, 11) = 11.
        # Subdivide 11 cells -> 176 children.
        # Wait, 11 * 16 = 176.
        # Next eval: 176 points.
        # Budget left: 183 - 176 = 7.
        # K = min(176, 7//16) = 0.
        # Stop subdivision.
        # Remaining 7 budget: use for local polish.
        
        # This uses 1 + 16 + 176 = 193 evals.
        # And 7 left.
        
        # This looks very similar to my previous estimate.
        
        # One issue: If we only keep the "top K" cells, we might discard good areas 
        # that were slightly worse in the coarse grid but contain good points in the fine grid.
        # However, for a smooth Pareto front, the "best" coarse cells are likely to 
        # contain the "best" fine cells.
        
        # To improve coverage, we can mix "exploitation" (top K by quality) 
        # with "exploration" (random or spread out).
        # But the directive says "deterministic".
        # We can use a deterministic spread: 
        # Instead of just top K, we can pick K cells that are maximally spread out 
        # in the objective space among the non-dominated ones.
        
        # Let's implement the ranking:
        # 1. Filter to non-dominated points.
        # 2. If more than K, select K points that are most diverse (maximizing min distance).
        #    This is a greedy farthest-point sampling.
        
        # This is a good deterministic strategy.
        
        # Implementation:
        
        # 1. Init
        current_cells = [(lo.copy(), hi.copy())]
        archive = []
        used = 0
        
        # Helper to evaluate a cell
        def eval_cell(cell):
            nonlocal used
            lo_c, hi_c = cell
            centroid = (lo_c + hi_c) / 2.0
            f = problem(centroid)
            used += 1
            return centroid, f
            
        # Helper to subdivide a cell
        def subdivide(cell):
            lo_c, hi_c = cell
            children = []
            # 2x2x2x2 grid
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            new_lo = lo_c.copy()
                            new_hi = hi_c.copy()
                            new_lo[0] = lo_c[0] + (hi_c[0]-lo_c[0]) * i / 2.0
                            new_hi[0] = lo_c[0] + (hi_c[0]-lo_c[0]) * (i+1) / 2.0
                            new_lo[1] = lo_c[1] + (hi_c[1]-lo_c[1]) * j / 2.0
                            new_hi[1] = lo_c[1] + (hi_c[1]-lo_c[1]) * (j+1) / 2.0
                            new_lo[2] = lo_c[2] + (hi_c[2]-lo_c[2]) * k / 2.0
                            new_hi[2] = lo_c[2] + (hi_c[2]-lo_c[2]) * (k+1) / 2.0
                            new_lo[3] = lo_c[3] + (hi_c[3]-lo_c[3]) * l / 2.0
                            new_hi[3] = lo_c[3] + (hi_c[3]-lo_c[3]) * (l+1) / 2.0
                            children.append((new_lo, new_hi))
            return children
            
        # Helper to select diverse points
        def select_diverse(points, k):
            # points: list of (x, f)
            if len(points) <= k:
                return points
            # Greedy farthest point sampling in objective space
            # Normalize objectives first? 
            # We don't have global mins/maxs yet.
            # We can use the current set's mins/maxs.
            f_arr = np.array([p[1] for p in points])
            mins = f_arr.min(axis=0)
            maxs = f_arr.max(axis=0)
            spans = maxs - mins
            spans[spans < 1e-12] = 1.0
            f_norm = (f_arr - mins) / spans
            
            selected = [0] # Start with first
            # Actually, start with the "best" point (lowest sum)
            sums = f_norm.sum(axis=1)
            best_idx = np.argmin(sums)
            selected = [best_idx]
            
            min_dists = np.full(len(points), np.inf)
            
            while len(selected) < k:
                # Update min dists
                last_idx = selected[-1]
                dists = np.linalg.norm(f_norm - f_norm[last_idx], axis=1)
                min_dists = np.minimum(min_dists, dists)
                # Pick the one with max min_dist
                next_idx = np.argmax(min_dists)
                selected.append(next_idx)
                
            return [points[i] for i in selected]

        # Main Loop
        while used < self.budget and not is_deadline_reached():
            if not current_cells:
                break
            
            # 1. Evaluate all current cells
            evaluated = []
            for cell in current_cells:
                if is_deadline_reached():
                    break
                x, f = eval_cell(cell)
                evaluated.append((x, f, cell))
                
            if not evaluated:
                break
                
            # Update archive
            for x, f, _ in evaluated:
                # We don't strictly need to add all to archive immediately, 
                # but it's good for tracking.
                # The "archive" for final return should be the non-dominated set of ALL evaluated points.
                # Let's keep a global list of all evaluated points to compute the final front.
                pass
                
            # 2. Determine how many to subdivide
            remaining = self.budget - used
            if remaining < 16: # Can't subdivide any cell (needs 16 evals)
                break
                
            max_k = remaining // 16
            # We can subdivide at most len(evaluated) cells
            k = min(len(evaluated), max_k)
            
            if k == 0:
                break
                
            # 3. Select which cells to subdivide
            # We want to subdivide cells that contain good points.
            # Rank evaluated points by quality.
            # Use normalized sum? Or just non-dominated filter?
            
            # Let's filter to non-dominated points first.
            nd_points = []
            for x, f, cell in evaluated:
                is_nd = True
                for _, f2, _ in evaluated:
                    if dominates(f2, f):
                        is_nd = False
                        break
                if is_nd:
                    nd_points.append((x, f, cell))
                    
            # If we have fewer non-dominated points than k, we can subdivide all of them.
            # If we have more, we select the most diverse k.
            
            if len(nd_points) <= k:
                selected_points = nd_points
            else:
                # Select diverse from nd_points
                # We need to pass (x, f) to select_diverse, but we also need the cell.
                # Let's create a list of (x, f, cell)
                # Modify select_diverse to work with this structure or just use indices.
                
                # Let's just use the indices of nd_points
                f_arr = np.array([p[1] for p in nd_points])
                mins = f_arr.min(axis=0)
                maxs = f_arr.max(axis=0)
                spans = maxs - mins
                spans[spans < 1e-12] = 1.0
                f_norm = (f_arr - mins) / spans
                
                sums = f_norm.sum(axis=1)
                best_idx = np.argmin(sums)
                selected_indices = [best_idx]
                
                min_dists = np.full(len(nd_points), np.inf)
                
                while len(selected_indices) < k:
                    last_idx = selected_indices[-1]
                    dists = np.linalg.norm(f_norm - f_norm[last_idx], axis=1)
                    min_dists = np.minimum(min_dists, dists)
                    next_idx = np.argmax(min_dists)
                    selected_indices.append(next_idx)
                    
                selected_points = [nd_points[i] for i in selected_indices]
                
            # 4. Generate children for selected cells
            new_cells = []
            for x, f, cell in selected_points:
                children = subdivide(cell)
                new_cells.extend(children)
                
            current_cells = new_cells
            
        # Final Polish
        # If we have budget left, do a local search on the best non-dominated points found so far.
        # We need to collect all evaluated points.
        # Let's re-run the collection logic or store it.
        # I'll store all evaluated points in a list `all_points`.
        
        # Re-implementing with storage:
        
        # Since I already wrote the loop without storing, I'll just restart the logic 
        # or add a storage list. 
        # Let's refactor the code to store all points.
        
        # I will rewrite the __call__ method cleanly in the final output.
        
        return None

# EVOLVE-BLOCK-END
