import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute squared norms
    norms_sq = (grid ** 2).sum(axis=1)

    def are_cospherical(p1, p2, p3, p4, p5):
        # Check if 5 points are cospherical
        # Matrix rows: [x, y, z, x^2+y^2+z^2, 1]
        pts = np.array([p1, p2, p3, p4, p5])
        m = np.column_stack([pts, norms_sq[np.where((grid == pts).all(axis=1))[0]], np.ones(5)])
        # Wait, norms_sq lookup is slow. Just compute directly.
        m = np.empty((5, 5), dtype=np.float64)
        for i in range(5):
            m[i, 0:3] = pts[i]
            m[i, 3] = np.dot(pts[i], pts[i])
            m[i, 4] = 1.0
        det = np.linalg.det(m)
        return abs(det) < 0.5

    def can_add(p, S_indices):
        # S_indices is a list of indices in grid
        if len(S_indices) < 4:
            return True
        
        p = grid[p]
        # Form matrix W for all points in S
        # Row i: (q_i - p, |q_i|^2 - |p|^2)
        m = len(S_indices)
        w = np.empty((m, 4), dtype=np.float64)
        for i, idx in enumerate(S_indices):
            q = grid[idx]
            w[i, 0:3] = q - p
            w[i, 3] = np.dot(q, q) - np.dot(p, p)
        
        # Check all combinations of 4
        # Optimization: If rank(w) < 4, return False immediately?
        # No, rank(w) < 4 implies ANY 4 are dependent? 
        # If rank is 3, yes, any 4 vectors in a 3-dim space are dependent.
        # So if rank(w) < 4, return False.
        rank = np.linalg.matrix_rank(w)
        if rank < 4:
            return False
            
        # If rank is 4, we still need to check if there is a dependent 4-tuple.
        # This happens if any 4x4 sub-determinant is 0.
        # We can iterate through combinations.
        # To speed up, we can check random combinations first?
        # But we need to be sure.
        # For m=50, 230k checks.
        # We can vectorize the determinant calculation for all combinations?
        # itertools.combinations is slow.
        
        # Alternative: 
        # If we find one dependent, return False.
        # We can just loop.
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            det = np.linalg.det(sub)
            if abs(det) < 0.5:
                return False
        return True

    # Initialize
    S = list(range(N))
    E = []
    
    # Phase 1: Remove violations
    # We need to find a violation in S.
    # Since S is large, we can pick random 5-tuples.
    rng = np.random.default_rng(0)
    
    while True:
        violation_found = False
        # Try to find a violation
        for _ in range(1000): # Limit trials
            idxs = rng.choice(S, 5, replace=False)
            if are_cospherical(grid[idxs[0]], grid[idxs[1]], grid[idxs[2]], grid[idxs[3]], grid[idxs[4]]):
                # Found violation
                # Remove one point (e.g., the first one)
                remove_idx = idxs[0]
                S.remove(remove_idx)
                E.append(remove_idx)
                violation_found = True
                break
        
        if not violation_found:
            break

    # Phase 2: Add back points
    # Shuffle E for randomness
    rng.shuffle(E)
    
    improved = True
    while improved:
        improved = False
        new_E = []
        for p in E:
            if can_add(p, S):
                S.append(p)
                improved = True
            else:
                new_E.append(p)
        E = new_E
        if improved:
            # After adding, S is still valid.
            # We can try to add more? The loop handles it.
            pass

    # Convert S to array
    return np.array([grid[i] for i in S], dtype=np.int64)
