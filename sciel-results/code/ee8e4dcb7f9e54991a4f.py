import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    # Generate all grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    
    # Precompute squared norms for grid points
    grid_sq_norms = (grid ** 2).sum(axis=1)

    def check_validity_fast(current_indices, candidate_indices):
        """
        Vectorized check: For each candidate in candidate_indices, check if adding it
        to current_indices violates the no-5-cospherical/plane condition.
        
        Returns: boolean array of same length as candidate_indices. True if valid.
        """
        if len(current_indices) < 4:
            return np.ones(len(candidate_indices), dtype=bool)
        
        # current_indices: array of indices into grid
        # candidate_indices: array of indices into grid
        
        p_pts = grid[candidate_indices] # (K, 3)
        p_sq = grid_sq_norms[candidate_indices] # (K,)
        
        q_pts = grid[current_indices] # (M, 3)
        q_sq = grid_sq_norms[current_indices] # (M,)
        
        M = len(current_indices)
        K = len(candidate_indices)
        
        # We need to check for each candidate p_k and each combination of 4 q's from current set
        # The condition is: |det([q_i - p, |q_i|^2 - |p|^2])| < 0.5 for any 4 q's.
        
        # This is hard to fully vectorize over combinations without exploding memory.
        # However, we can vectorize over candidates for a FIXED combination of 4 q's.
        # Or, we can iterate over combinations of 4 q's, and for each, check all K candidates at once.
        
        # Number of combinations of 4 from M.
        # If M is large, this is slow.
        # But we can limit the number of q's we consider? No, must be exact.
        
        # Alternative: Iterate over combinations of 4 from M.
        # For each combo (q1, q2, q3, q4):
        #   Build 4x4 matrix for each candidate p.
        #   Row i of matrix for p: [q_i - p, |q_i|^2 - |p|^2]
        #   Compute det for all K candidates at once.
        
        # Let's try this.
        # M is current size. K is candidate count.
        # Combinations of 4 from M.
        
        # Optimization: If M is small, just do it.
        # If M is large, this might be too slow.
        # Let's assume M grows up to ~30-50. C(50,4) ~ 230k.
        # For each combo, we build a (K, 4, 4) array and compute det.
        # np.linalg.det on (K, 4, 4) is fast.
        
        valid_mask = np.ones(K, dtype=bool)
        
        # To speed up, we can break early if a candidate becomes invalid.
        # But vectorized det returns all at once.
        
        # Iterate over combinations
        # Note: itertools.combinations is in Python, which is slow.
        # But we can generate indices and use them.
        
        # Let's use a different approach: 
        # Instead of iterating all C(M,4), we can just check a random subset if M is huge?
        # No, must be exact.
        
        # Let's stick to iterating combinations but optimize the inner loop.
        
        # If M < 10, C(M,4) is small.
        # If M is large, we might need a smarter check.
        # For now, let's try the direct combo iteration with vectorized det.
        
        for combo in itertools.combinations(range(M), 4):
            # q_pts[combo] is (4, 3)
            # q_sq[combo] is (4,)
            
            # For each candidate p_k:
            # Row i: [q_i - p_k, |q_i|^2 - |p_k|^2]
            
            # q_pts[combo] shape (4,3)
            # p_pts shape (K,3)
            
            # diff = q_pts[combo][None, :, :] - p_pts[:, None, :]  # (K, 4, 3)
            # sq_diff = q_sq[combo][None, :] - p_sq[:, None] # (K, 4)
            
            # mat = np.concatenate([diff, sq_diff[..., None]], axis=-1) # (K, 4, 4)
            # dets = np.linalg.det(mat) # (K,)
            
            # This creates large temporary arrays.
            # Let's see if we can do it more efficiently.
            
            q_c = q_pts[combo] # (4,3)
            q_sq_c = q_sq[combo] # (4,)
            
            # diff: (K, 4, 3)
            # q_c[None, :, :] is (1, 4, 3)
            # p_pts[:, None, :] is (K, 1, 3)
            diff = q_c[None, :, :] - p_pts[:, None, :]
            
            # sq_diff: (K, 4)
            sq_diff = q_sq_c[None, :] - p_sq[:, None]
            
            # Combine to (K, 4, 4)
            mat = np.empty((K, 4, 4), dtype=np.float64)
            mat[:, :, :3] = diff
            mat[:, :, 3] = sq_diff
            
            dets = np.linalg.det(mat)
            
            # If |det| < 0.5, invalid
            invalid_here = np.abs(dets) < 0.5
            
            # Update valid_mask
            valid_mask &= ~invalid_here
            
            # Optimization: If no candidates are valid anymore, break
            if not np.any(valid_mask):
                break
        
        return valid_mask

    def solve():
        start_time = time.time()
        deadline = start_time + 590 # Safety margin for 600s limit
        
        best_points = np.empty((0, 3), dtype=np.int64)
        best_size = 0
        
        # Deterministic seeds for reproducibility and diversity
        seeds = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768, 
                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 111, 222, 333, 444, 555, 666, 777, 888, 999, 1000]
        
        # Phase 1: Fast Greedy Construction with multiple seeds
        for seed in seeds:
            if time.time() > deadline:
                break
                
            rng = np.random.RandomState(seed)
            order = rng.permutation(num_points)
            
            current_indices = []
            
            # Greedy: iterate through order, add if valid
            # To speed up, we can batch the validity checks?
            # The check is sequential because adding a point changes the set.
            # So we must check one by one.
            
            for p_idx in order:
                # Check if p_idx is valid to add to current_indices
                # Use the fast check with a single candidate
                # But check_validity_fast expects arrays.
                # For single candidate, we can use a specialized check or just call it.
                
                # Actually, for a single candidate, the combo iteration is the bottleneck.
                # Let's write a specialized single-point check that is faster.
                
                if len(current_indices) < 4:
                    current_indices.append(p_idx)
                    continue
                
                # Single point check
                p = grid[p_idx]
                p_sq = grid_sq_norms[p_idx]
                
                q_pts = grid[current_indices]
                q_sq = grid_sq_norms[current_indices]
                M = len(current_indices)
                
                valid = True
                # Iterate combos
                for combo in itertools.combinations(range(M), 4):
                    q_c = q_pts[combo]
                    q_sq_c = q_sq[combo]
                    
                    # Build 4x4 matrix
                    # Rows: [q_i - p, |q_i|^2 - |p|^2]
                    mat = np.empty((4, 4), dtype=np.float64)
                    mat[:, :3] = q_c - p
                    mat[:, 3] = q_sq_c - p_sq
                    
                    det_val = np.linalg.det(mat)
                    if abs(det_val) < 0.5:
                        valid = False
                        break
                
                if valid:
                    current_indices.append(p_idx)
            
            if len(current_indices) > best_size:
                best_size = len(current_indices)
                best_points = grid[np.array(current_indices)]
                
                # Optional: Try to extend best_points further?
                # The greedy already did that.
        
        # Phase 2: Local Search / Swap on the best set found
        # Try to swap points to allow adding more?
        # This is complex. Let's just do more greedy restarts if time permits.
        
        # Add more seeds if time permits
        extra_seeds = list(range(100, 500))
        rng = np.random.RandomState(999)
        rng.shuffle(extra_seeds)
        
        for seed in extra_seeds:
            if time.time() > deadline:
                break
                
            rng_s = np.random.RandomState(seed)
            order = rng_s.permutation(num_points)
            
            current_indices = []
            
            for p_idx in order:
                if len(current_indices) < 4:
                    current_indices.append(p_idx)
                    continue
                
                p = grid[p_idx]
                p_sq = grid_sq_norms[p_idx]
                
                q_pts = grid[current_indices]
                q_sq = grid_sq_norms[current_indices]
                M = len(current_indices)
                
                valid = True
                for combo in itertools.combinations(range(M), 4):
                    q_c = q_pts[combo]
                    q_sq_c = q_sq[combo]
                    
                    mat = np.empty((4, 4), dtype=np.float64)
                    mat[:, :3] = q_c - p
                    mat[:, 3] = q_sq_c - p_sq
                    
                    det_val = np.linalg.det(mat)
                    if abs(det_val) < 0.5:
                        valid = False
                        break
                
                if valid:
                    current_indices.append(p_idx)
            
            if len(current_indices) > best_size:
                best_size = len(current_indices)
                best_points = grid[np.array(current_indices)]
        
        return best_points

    return solve()

# EVOLVE-BLOCK-START
# (The code above is the implementation)
# EVOLVE-BLOCK-END
