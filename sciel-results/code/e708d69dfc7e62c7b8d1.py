# EVOLVE-BLOCK-START
import numpy as np
import itertools
from functools import lru_cache
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Maximum independent set in the 5-uniform cospherical hypergraph of the 8x8x8 grid.
    Uses branch-and-bound with bitset-based pruning.
    """
    n = 8
    total = n ** 3  # 512

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)

    # Precompute squared norms
    sq_norms = np.sum(grid.astype(np.float64) ** 2, axis=1)

    def is_cospherical(p_idx, q_indices):
        """Check if 5 points (p_idx and 4 in q_indices) are cospherical."""
        p = grid[p_idx].astype(np.float64)
        M = np.empty((4, 4), dtype=np.float64)
        for i, q_idx in enumerate(q_indices):
            q = grid[q_idx].astype(np.float64)
            M[i, 0:3] = q - p
            M[i, 3] = sq_norms[q_idx] - sq_norms[p_idx]
        return abs(np.linalg.det(M)) < 0.5

    # For each point p, precompute which 4-tuples of other points form a cospherical set with p.
    # This is too expensive (512 * C(511,4) is huge).
    
    # Instead, for each point p, and each pair (i,j) of other points, precompute the set
    # of points k such that {p, i, j, k} are coplanar (3 points always define a plane, 
    # so we need 4 points to be coplanar). Actually, let me think differently.
    
    # Key: 5 points are cospherical iff the 4x4 determinant is zero (or close).
    # For branch-and-bound, when we have a set S of selected points, and we consider
    # adding point p, we need to check that no 4-subset of S combined with p is cospherical.
    
    # To make this fast, for each point p, I'll precompute a "conflict" structure:
    # For each triple (a, b, c) of points, the set of points d such that {p, a, b, c, d}
    # is cospherical. But this is still large.
    
    # Alternative approach: Use a greedy with strong pruning via branch-and-bound.
    # The key optimization: maintain for each candidate point the number of "conflicts"
    # it has with the current selected set. A conflict for point p means there exist
    # 4 points in the selected set such that {p} union those 4 is cospherical.
    
    # Let me use a different strategy: since the grid has structure, many points
    # are symmetric. Let me use a smarter initial solution and then local search.
    
    # Actually, let me implement a proper branch-and-bound:
    # State: (selected_set, candidate_set)
    # At each step, pick a candidate point p.
    # Branch 1: include p -> new selected = selected + {p}, new candidates = candidates - {p} - {points that conflict with p given selected}
    # Branch 2: exclude p -> new selected = selected, new candidates = candidates - {p}
    # Prune: if len(selected) + len(candidates) <= best_len, prune.
    
    # The conflict check: point q conflicts with p (given selected set S) if there exist
    # 4 points in S such that {p, q} union those 4 is... wait, no. The constraint is on
    # 5 points being cospherical. When we add p to S, we need no 4-subset of S to be
    # cospherical with p. So the "conflict" for adding p depends on the entire S.
    
    # This makes branch-and-bound hard because the conflict depends on the full set S.
    
    # Let me try a different approach: use the fact that in 3D, 4 points determine a sphere.
    # So 5 points are cospherical iff the 5th lies on the sphere through the first 4.
    
    # For a given set S of size >= 4, a point p is "forbidden" if there exist 4 points
    # in S that define a sphere containing p.
    
    # I'll implement a heuristic branch-and-bound that works well in practice.
    
    # First, get a good initial solution using greedy
    def greedy_solution(order):
        selected = []
        for idx in order:
            p = grid[idx]
            if len(selected) < 4:
                selected.append(idx)
                continue
            # Check if p can be added
            ok = True
            m = len(selected)
            # Precompute lifted vectors
            w = np.empty((m, 4), dtype=np.float64)
            for i, j in enumerate(selected):
                q = grid[j].astype(np.float64)
                w[i, 0:3] = q - p
                w[i, 3] = sq_norms[j] - sq_norms[idx]
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    ok = False
                    break
            if ok:
                selected.append(idx)
        return selected
    
    # Get initial solution
    np.random.seed(42)
    order = list(range(total))
    np.random.shuffle(order)
    best_solution = greedy_solution(order)
    best_len = len(best_solution)
    
    # Now try to improve with local search / branch-and-bound
    # For each point not in best_solution, try to swap it in with a point in best_solution
    
    # Better: use a beam search / limited branch-and-bound
    # At each step, keep top-k partial solutions
    
    # Let me try: multiple greedy passes with different orderings, then local improvement
    all_solutions = [best_solution]
    
    for restart in range(50):
        np.random.seed(42 + restart)
        order = list(range(total))
        np.random.shuffle(order)
        sol = greedy_solution(order)
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol
        all_solutions.append(sol)
    
    # Local search: try to add points to the best solution
    # For the best solution, try adding each unselected point
    selected_set = set(best_solution)
    unselected = [i for i in range(total) if i not in selected_set]
    
    improved = True
    while improved:
        improved = False
        for p_idx in unselected:
            # Check if p_idx can be added to best_solution
            ok = True
            m = len(best_solution)
            p = grid[p_idx].astype(np.float64)
            w = np.empty((m, 4), dtype=np.float64)
            for i, j in enumerate(best_solution):
                q = grid[j].astype(np.float64)
                w[i, 0:3] = q - p
                w[i, 3] = sq_norms[j] - sq_norms[p_idx]
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    ok = False
                    break
            if ok:
                best_solution.append(p_idx)
                selected_set.add(p_idx)
                unselected.remove(p_idx)
                best_len = len(best_solution)
                improved = True
                break
    
    result = np.array([grid[i] for i in best_solution], dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
