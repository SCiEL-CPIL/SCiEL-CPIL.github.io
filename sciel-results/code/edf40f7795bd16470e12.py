"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Deterministic adaptive grid refinement multi-objective optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        budget = self.budget

        # We will keep a list of (point, fval) for all evaluated points
        # and a list of active cells (as tuples of interval indices per dimension)
        # to decide where to subdivide next.

        # Start with a coarse grid. Since dim=4, let's start with a 2^4 = 16 point grid.
        # Actually, let's think about the budget. Budget is 200.
        # If we start with 16 points, we have 184 left.
        # Adaptive grid refinement:
        # 1. Evaluate a set of points.
        # 2. Identify non-dominated points.
        # 3. Find the "gaps" in the non-dominated set or the most promising unexplored regions.
        #
        # A simpler deterministic approach:
        # Use a space-filling curve or a systematic grid that refines based on hypervolume contribution.
        #
        # Let's try a different deterministic strategy:
        # 1. Evaluate a Latin Hypercube or a full factorial design of small size (e.g., 3^4 = 81 points).
        # 2. From the non-dominated set, identify the largest gaps in objective space.
        # 3. Place new points in the middle of those gaps.
        #
        # However, the directive says: "iteratively subdivides the 4D parameter space into hypercubes, evaluating only the centroids and retaining the finest grid cells whose objective values are non-dominated".
        #
        # Let's implement a quadtree/octree (here 16-tree since 4D) refinement.
        #
        # Step 1: Start with the root cell (the whole box). Evaluate its centroid.
        # Step 2: Subdivide the root cell into 16 sub-cells. Evaluate centroids of all 16 sub-cells.
        #         (Total evaluations: 1 + 16 = 17)
        # Step 3: From the evaluated points, find the non-dominated set.
        #         For each non-dominated point, we can consider its cell as "active".
        #         We want to refine the cells that contain non-dominated points or are adjacent to them?
        #         The directive says "retaining the finest grid cells whose objective values are non-dominated".
        #         This implies we should refine the cells that are currently non-dominated.
        #
        # Let's refine the algorithm:
        # - Maintain a list of active cells. Initially, the root cell.
        # - In each iteration:
        #   1. Take one active cell (or a batch).
        #   2. If budget allows, subdivide it into 16 children.
        #   3. Evaluate the centroids of the children.
        #   4. Add the new points to the global set of evaluated points.
        #   5. Determine which of the new points are non-dominated (globally).
        #   6. For each non-dominated new point, add its parent cell (the cell we just subdivided? No, the child cell containing the point) to the active list for further refinement?
        #      Wait, if we subdivide a cell, we get 16 new cells. If a centroid of a child cell is non-dominated, we should add that *child cell* to the active list for future subdivision.
        #      If a centroid is dominated, we might still want to explore it if it's close to the front, but the directive says "retaining... non-dominated". So we only refine cells whose centroid is non-dominated.
        #
        # Let's trace the budget:
        # Iteration 0: Evaluate root centroid (1 eval). Active cells: [Root].
        # Iteration 1: Subdivide Root into 16 children. Evaluate 16 centroids (16 evals). Total: 17.
        #              Check which of the 16 are non-dominated. Suppose k are non-dominated.
        #              Active cells: those k child cells.
        # Iteration 2: Subdivide the k active cells. Each produces 16 children. Total new evals: 16*k.
        #              If k=4, that's 64 evals. Total: 17 + 64 = 81.
        #              Check which of the 64 are non-dominated. Suppose m are non-dominated.
        #              Active cells: those m child cells.
        # Iteration 3: Subdivide m active cells. 16*m evals.
        #
        # This can blow up quickly. We need to be careful with the budget.
        #
        # Alternative interpretation: "retaining the finest grid cells whose objective values are non-dominated"
        # Maybe we should evaluate a full grid at each level, but only keep the non-dominated cells for the next level's subdivision?
        #
        # Let's try a more controlled approach:
        # 1. Start with a 2^4 grid (16 points). Evaluate all. (16 evals)
        # 2. Find non-dominated points.
        # 3. For each non-dominated point, its cell is "interesting".
        # 4. Subdivide only the cells containing non-dominated points.
        # 5. Evaluate the centroids of the new sub-cells.
        # 6. Repeat until budget is exhausted.
        #
        # Let's implement this.
        #
        # Data structures:
        # - `evaluated`: list of (point, fval)
        # - `active_cells`: list of cells. A cell is represented by (lo_idx, hi_idx) for each dim? Or just the bounds?
        #   Let's represent a cell by its bounds: (lo_vec, hi_vec).
        #
        # Algorithm:
        # 1. Initialize `active_cells` with the root cell (lo, hi).
        # 2. Initialize `evaluated` as empty.
        # 3. While budget > 0:
        #      a. If `active_cells` is empty, we need a fallback. Maybe pick a random point or the centroid of the whole space?
        #         Let's assume we always have active cells if we start with the root and refine non-dominated ones.
        #      b. We need to decide how many cells to refine in this step.
        #         If we refine all active cells at once, we might exceed the budget.
        #         Let's refine one cell at a time, or a batch that fits the budget.
        #         Actually, refining one cell costs 16 evals.
        #         If budget < 16, we can't refine a full cell. We can just evaluate the centroid of the most promising cell.
        #      c. Let's pick the "best" active cell to refine. How to define best?
        #         The cell whose centroid has the highest hypervolume contribution?
        #         Or simply the first one?
        #         Let's prioritize cells whose current best point (centroid) is closest to the reference point in objective space?
        #         Or simply iterate through them.
        #
        # Let's try a simpler, more robust deterministic strategy that doesn't strictly follow the "cell refinement" but achieves similar systematic coverage:
        #
        # Strategy: Adaptive Sampling in Objective Space
        # 1. Evaluate a small initial set (e.g., 3^4 = 81 points).
        # 2. Compute the non-dominated set.
        # 3. Identify the largest gaps in the non-dominated set in objective space.
        # 4. Interpolate or extrapolate in parameter space to target those gaps? This is hard without a model.
        #
        # Let's stick to the grid refinement but make it budget-aware.
        #
        # Revised Grid Refinement:
        # 1. Start with a 2^4 grid (16 points). Evaluate all.
        # 2. Identify non-dominated points.
        # 3. For each non-dominated point, mark its cell for refinement.
        # 4. While budget allows:
        #      - Pick a cell to refine (e.g., the one with the best centroid).
        #      - Subdivide it into 16 children.
        #      - Evaluate the 16 centroids.
        #      - Add to global set.
        #      - Identify new non-dominated points among the 16.
        #      - Add their child cells to the refinement queue.
        #      - Remove the parent cell from the queue (it's been refined).
        #
        # This seems reasonable. Let's implement it.
        #
        # Note: We need to handle the case where budget is not a multiple of 16.
        # If remaining budget < 16, we can evaluate the centroid of the best active cell and stop.

        dim = self.dim
        n_children = 2 ** dim  # 16 for dim=4

        # Represent a cell by its bounds
        # cell = (lo_array, hi_array)
        root_lo = lo.copy()
        root_hi = hi.copy()

        # Queue of cells to refine. Each element is (lo, hi)
        # We use a priority queue? Or just a list?
        # Let's use a list and pick the "best" one based on the centroid's objective value.
        # To avoid recomputing centroids, we can store (lo, hi, centroid, f_centroid) in the queue?
        # But the centroid's dominance status changes as we add more points.
        # So we should re-evaluate dominance at each step? That's expensive.
        #
        # Alternative: Just refine cells in a fixed order (e.g., BFS) until budget runs out.
        # This is simpler and deterministic.
        #
        # Let's try BFS refinement:
        # 1. Queue: [Root]
        # 2. Evaluated: []
        # 3. While queue not empty and budget > 0:
        #      - Pop cell from front.
        #      - If budget < n_children:
        #          - Evaluate centroid of this cell.
        #          - Break.
        #      - Subdivide cell into n_children children.
        #      - Evaluate centroids of all children.
        #      - Add to evaluated.
        #      - For each child whose centroid is non-dominated (globally), add it to the back of the queue.
        #
        # This ensures we explore the space systematically.
        #
        # Let's implement this.

        evaluated_points = []  # list of points
        evaluated_fvals = []   # list of fvals

        def is_dominated(f, F):
            """Check if f is dominated by any point in F."""
            # F is (N, 2)
            # f is (2,)
            # Dominated if exists f' in F such that f' <= f and f' != f
            # Since we are minimizing, lower is better.
            # f' <= f (all objectives) and f' < f (at least one)
            if len(F) == 0:
                return False
            # Vectorized check
            # f' <= f for all objectives
            leq = np.all(F <= f, axis=1)
            # f' < f for at least one objective
            lt = np.any(F < f, axis=1)
            return np.any(leq & lt)

        def get_centroid(lo_c, hi_c):
            return (lo_c + hi_c) / 2.0

        # Initial queue
        queue = [(root_lo, root_hi)]

        while len(evaluated_points) < budget and queue:
            cell_lo, cell_hi = queue.pop(0)
            centroid = get_centroid(cell_lo, cell_hi)

            # If we can't afford a full subdivision, just evaluate the centroid
            if budget - len(evaluated_points) < n_children:
                f_val = problem(centroid)
                evaluated_points.append(centroid)
                evaluated_fvals.append(f_val)
                break

            # Subdivide the cell
            mid = (cell_lo + cell_hi) / 2.0
            children = []
            for mask in range(n_children):
                child_lo = cell_lo.copy()
                child_hi = cell_hi.copy()
                for d in range(dim):
                    if (mask >> d) & 1:
                        child_lo[d] = mid[d]
                    else:
                        child_hi[d] = mid[d]
                children.append((child_lo, child_hi))

            # Evaluate all children centroids
            new_points = []
            new_fvals = []
            for c_lo, c_hi in children:
                c_centroid = get_centroid(c_lo, c_hi)
                f_val = problem(c_centroid)
                new_points.append(c_centroid)
                new_fvals.append(f_val)

            # Add to global evaluated
            evaluated_points.extend(new_points)
            evaluated_fvals.extend(new_fvals)

            # Update F matrix for dominance check
            F = np.array(evaluated_fvals)

            # Determine which children are non-dominated
            # Note: A child is non-dominated if its centroid is not dominated by any point in F
            # We only add non-dominated children to the queue for further refinement
            for i, (c_lo, c_hi) in enumerate(children):
                f_val = new_fvals[i]
                if not is_dominated(f_val, F):
                    queue.append((c_lo, c_hi))

        return evaluated_points
# EVOLVE-BLOCK-END
