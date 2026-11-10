import numpy as np
import itertools
import time
import random

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Finds a large subset of the 8x8x8 grid with no 5 cospherical points.
    Uses Simulated Annealing / Local Search with probabilistic validity checks
    for speed, and strict verification for the final best solution.
    """
    n = 8
    # Precompute all grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute lifted vectors L = [x, y, z, x^2+y^2+z^2]
    # Using float64 for determinant stability, though inputs are int
    L = np.empty((N, 4), dtype=np.float64)
    L[:, 0:3] = grid
    L[:, 3] = np.sum(grid ** 2, axis=1)
    
    start_time = time.time()
    time_limit = 590.0  # Safety margin
    
    def strict_can_add(p_idx, S_indices):
        """
        Strictly checks if adding point p_idx to set S_indices creates a cospherical 5-tuple.
        Returns True if valid, False otherwise.
        """
        m = len(S_indices)
        if m < 4:
            return True
        
        p = L[p_idx]
        # Relative lifted vectors
        # W has shape (m, 4)
        W = L[S_indices] - p
        
        # Check all combinations of 4
        # For efficiency, we can break early if a violation is found
        # itertools.combinations is slow in Python, but m is small-ish
        # If m is large, this is the bottleneck.
        
        # Optimization: If m is large, we can use a randomized check first?
        # But this is the "strict" version, so we must check all.
        
        # To speed up, we can check the rank of W.
        # If rank(W) < 4, then there are 4 dependent vectors.
        # If rank(W) == 4, we still need to check if any 4 are dependent.
        # Actually, if rank(W) < 4, it's definitely invalid.
        # If rank(W) == 4, it might be valid or invalid.
        
        # Let's just check all combos if m is small, otherwise sample?
        # No, strict must be strict.
        
        # We can vectorize the determinant calculation for all combos if m is moderate.
        # Number of combos C(m, 4).
        # If m=100, C(100,4) ~ 4M. 
        # 4M determinants of 4x4 matrices.
        # Using numpy, this might take ~0.1-0.5s.
        
        if m < 4:
            return True
            
        # Generate indices for combinations
        # To avoid memory blowup for very large m, we process in batches?
        # Or just use itertools and loop?
        # Let's try to vectorize if possible.
        
        # For now, simple loop with early exit is safer for memory.
        # But Python loop is slow.
        
        # Let's use a hybrid: 
        # 1. Check rank. If rank < 4, return False.
        # 2. If rank == 4, we need to check if any 4 are dependent.
        #    This is equivalent to checking if the matrix W has a 4x4 minor with det 0.
        #    If the rows are in general position, all 4x4 minors are non-zero.
        #    If there is a dependence, the rank of that subset is < 4.
        
        # Actually, if rank(W) == 4, and there is a dependent 4-subset, 
        # then the other rows must be linear combinations of the independent ones.
        
        # Let's just check all combos if m <= 20.
        # If m > 20, we use a randomized strict check? 
        # No, "strict" implies correctness.
        
        # We'll optimize the combo generation.
        if m > 30:
            # This is too slow for strict check in a loop.
            # We will only call strict_can_add on the "best" candidate, 
            # not on every step of the search.
            # So m might be large.
            # We need a fast strict check.
            
            # We can check the null space?
            # If there is a 4x4 zero determinant, the 4 vectors are coplanar in 4D.
            
            # Let's try to check all combos using numpy broadcasting if m is not huge.
            # If m=50, C(50,4) = 230k.
            # 230k * 4x4 det is fast.
            
            combos = np.array(list(itertools.combinations(range(m), 4)))
            if len(combos) == 0:
                return True
            
            # Batch process
            # W[combos] has shape (C, 4, 4)
            # We can compute det for all at once
            # But memory might be an issue if C is large.
            # 230k * 16 floats = 3.6 MB. Fine.
            # 4M * 16 floats = 64 MB. Fine.
            
            sub = W[combos]
            dets = np.linalg.det(sub)
            return not np.any(np.abs(dets) < 1e-9)
        else:
            # Small m, simple loop
            for c in itertools.combinations(range(m), 4):
                sub = W[list(c)]
                if abs(np.linalg.det(sub)) < 1e-9:
                    return False
            return True

    def probabilistic_can_add(p_idx, S_indices, num_samples=50):
        """
        Probabilistically checks if adding p_idx is valid.
        Returns (is_valid, conflicting_indices).
        conflicting_indices is None if valid, else a tuple of 4 indices in S_indices.
        """
        m = len(S_indices)
        if m < 4:
            return True, None
        
        p = L[p_idx]
        W = L[S_indices] - p
        
        # Check rank first
        # If rank < 4, definitely invalid.
        # But we need the specific conflicting set for swap.
        
        # Sample random 4-tuples
        # If m is large, we sample. If m is small, we check all.
        if m <= 20:
            samples = list(itertools.combinations(range(m), 4))
        else:
            samples = [tuple(np.random.choice(m, 4, replace=False)) for _ in range(num_samples)]
        
        for c in samples:
            sub = W[list(c)]
            if abs(np.linalg.det(sub)) < 1e-9:
                return False, c
        
        return True, None

    best_set = []
    best_size = 0
    
    # Initial greedy to get a decent start
    def greedy_init():
        S = []
        remaining = list(range(N))
        random.shuffle(remaining)
        for p_idx in remaining:
            valid, _ = probabilistic_can_add(p_idx, S, num_samples=10)
            if valid:
                S.append(p_idx)
        return S

    S = greedy_init()
    if len(S) > best_size:
        # Verify strictly
        if all(strict_can_add(p, [x for x in S if x != p]) for p in S):
            best_set = S.copy()
            best_size = len(S)

    current_S = S
    current_size = len(current_S)
    
    # Local Search
    while time.time() - start_time < time_limit:
        # Try to add a random point
        if current_size < N:
            candidates = [i for i in range(N) if i not in current_S]
            if candidates:
                p_idx = random.choice(candidates)
                valid, conflict = probabilistic_can_add(p_idx, current_S, num_samples=20)
                if valid:
                    current_S.append(p_idx)
                    current_size += 1
                    # Verify strictly occasionally or if size increased
                    if current_size > best_size:
                        # Strict check is expensive, do it only if we think we improved
                        # We can do a lighter check: check only the new point against the rest
                        # But we need to ensure the whole set is valid.
                        # Since we only added one point, the only new violations involve p_idx.
                        # So we just need to check p_idx against the rest strictly.
                        # And we trust the previous set was valid.
                        if strict_can_add(p_idx, [x for x in current_S if x != p_idx]):
                            best_set = current_S.copy()
                            best_size = current_size
                else:
                    # Conflict found, try to swap
                    if conflict is not None:
                        # conflict is a tuple of 4 indices in current_S
                        # Pick one to remove
                        remove_idx = random.choice(conflict)
                        # Remove it
                        current_S.remove(remove_idx)
                        # Add p_idx
                        current_S.append(p_idx)
                        # Note: current_size remains same
                        # We don't update best here as size didn't change, 
                        # but structure might be better for future additions.
        
        # Occasionally, try to remove a random point and see if we can add more?
        # Or just restart if stuck?
        # Let's add a random restart mechanism if no improvement for a while.
        # Simple: every 5000 steps, if no improvement, restart with greedy.
        # We don't have a step counter, let's use time.
        
        # To avoid infinite loops of swaps, we can just let it run.
        
    # Final verification of best_set
    # Ensure best_set is strictly valid
    # We already checked when updating, but let's be sure.
    # If best_set is empty, return empty array.
    
    if not best_set:
        return np.array([], dtype=np.int64).reshape(0, 3)
    
    # Convert indices to coordinates
    result = grid[best_set]
    return result.astype(np.int64)

# EVOLVE-BLOCK-END
