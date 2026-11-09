# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Finds a large set of points in the 8x8x8 grid such that no 5 points are co-spherical.
    Uses greedy construction with random restarts, followed by local search (add and swap).
    """
    n = 8
    deadline = time.time() + 600.0  # Safety margin before 630s kill

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1)  # |p|^2 for each grid point
    num_points_total = len(grid)
    
    # Precompute all 4-combinations of indices for a set of size m?
    # m varies, so we generate on the fly or cache for specific sizes if needed.
    # Given m is likely < 100, itertools.combinations is fine for index generation,
    # but the determinant calculation is the bottleneck.

    def can_add_vectorized(arr: np.ndarray, p: np.ndarray, arr_sq: np.ndarray) -> bool:
        """
        Check if point p can be added to arr without creating 5 co-spherical points.
        Uses vectorized determinant calculation.
        """
        m = len(arr)
        if m < 4:
            return True
        
        # Build lifted matrix relative to p
        # Rows: (q_i - p, |q_i|^2 - |p|^2)
        # Shape: (m, 4)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = arr_sq - (p ** 2).sum()
        
        # We need to check all combinations of 4 rows from w
        # If any 4x4 submatrix has |det| < 0.5, return False
        
        # Generate all combinations of 4 indices
        # For m ~ 50, C(50,4) ~ 230k. This might be too slow if done in pure python loop.
        # However, we can batch the determinant calculation.
        
        # Optimization: If m is small, just loop. If m is large, we might need a smarter check.
        # Let's try to vectorize as much as possible.
        
        # Since we can't easily vectorize "all combinations" without generating them,
        # we generate the indices, then batch the det calculation.
        
        combos = list(itertools.combinations(range(m), 4))
        if not combos:
            return True
            
        # Convert to numpy array of indices
        # Shape: (num_combos, 4)
        combo_arr = np.array(combos, dtype=np.int64)
        
        # Extract the submatrices
        # w[combo_arr] gives shape (num_combos, 4, 4)
        submats = w[combo_arr]
        
        # Calculate determinants
        # np.linalg.det handles (N, 4, 4) -> (N,)
        dets = np.linalg.det(submats)
        
        if np.any(np.abs(dets) < 0.5):
            return False
            
        return True

    def greedy_fill(order: np.ndarray) -> np.ndarray:
        """Greedy fill with given order."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        for idx in order:
            if time.time() > deadline - 10:
                break
            p = grid[idx]
            if can_add_vectorized(arr, p, arr_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([grid_sq[idx]], dtype=np.float64)
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, grid_sq[idx])
        return arr, arr_sq

    def local_search(arr: np.ndarray, arr_sq: np.ndarray) -> tuple:
        """
        Local search: Try to add points, then try swaps.
        Returns improved arr and arr_sq.
        """
        in_arr_set = set(map(tuple, arr))
        remaining_indices = [i for i in range(num_points_total) if tuple(grid[i]) not in in_arr_set]
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Phase 1: Try to add a point
            # Shuffle remaining to try different ones
            np.random.shuffle(remaining_indices)
            for idx in remaining_indices:
                p = grid[idx]
                if can_add_vectorized(arr, p, arr_sq):
                    # Add it
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, grid_sq[idx])
                    in_arr_set.add(tuple(p))
                    remaining_indices.remove(idx)
                    improved = True
                    break # Restart add phase
            
            # Phase 2: Try swaps if no add was possible
            if not improved and len(remaining_indices) > 0 and len(arr) > 0:
                # Try a few random swaps
                # Pick a random point from arr to remove, and a random point from remaining to add
                # Check if (arr - p_in + p_out) is valid.
                # This is equivalent to checking if p_out is valid against (arr - p_in).
                
                # To avoid expensive checks, we can just try a limited number of swaps.
                swap_attempts = 0
                max_swap_attempts = 50
                
                while swap_attempts < max_swap_attempts and time.time() < deadline - 5:
                    swap_attempts += 1
                    
                    # Random in-point
                    in_idx = np.random.randint(len(arr))
                    p_in = arr[in_idx]
                    
                    # Random out-point
                    out_idx = np.random.choice(remaining_indices)
                    p_out = grid[out_idx]
                    
                    # Create reduced set
                    # Remove in_idx from arr
                    if len(arr) == 1:
                        reduced_arr = np.empty((0, 3), dtype=np.int64)
                        reduced_arr_sq = np.empty(0, dtype=np.float64)
                    else:
                        mask = np.ones(len(arr), dtype=bool)
                        mask[in_idx] = False
                        reduced_arr = arr[mask]
                        reduced_arr_sq = arr_sq[mask]
                    
                    # Check if p_out can be added to reduced_arr
                    if can_add_vectorized(reduced_arr, p_out, reduced_arr_sq):
                        # Perform swap
                        # Update arr and arr_sq
                        if len(arr) == 1:
                            arr = p_out[None, :]
                            arr_sq = np.array([grid_sq[out_idx]], dtype=np.float64)
                        else:
                            mask = np.ones(len(arr), dtype=bool)
                            mask[in_idx] = False
                            arr = arr[mask]
                            arr_sq = arr_sq[mask]
                            arr = np.vstack([arr, p_out])
                            arr_sq = np.append(arr_sq, grid_sq[out_idx])
                        
                        # Update sets
                        in_arr_set.remove(tuple(p_in))
                        in_arr_set.add(tuple(p_out))
                        
                        # Update remaining_indices
                        remaining_indices.remove(out_idx)
                        # p_in is now available
                        in_idx_grid = np.where((grid == p_in).all(axis=1))[0][0]
                        if in_idx_grid not in remaining_indices:
                            remaining_indices.append(in_idx_grid)
                        
                        improved = True
                        break

        return arr, arr_sq

    np.random.seed(42)
    best_arr = np.empty((0, 3), dtype=np.int64)
    best_len = 0
    
    # Number of restarts depends on time.
    # Let's do a fixed number of restarts, but break if time is up.
    # Each restart takes some time.
    
    for restart in range(100):
        if time.time() > deadline - 60:
            break
        
        order = np.random.permutation(num_points_total)
        
        # Greedy phase
        curr_arr, curr_arr_sq = greedy_fill(order)
        
        # Local search phase
        curr_arr, curr_arr_sq = local_search(curr_arr, curr_arr_sq)
        
        if len(curr_arr) > best_len:
            best_arr = curr_arr.copy()
            best_len = len(curr_arr)
            
            # If we reach a very high count, we can stop early? 
            # Max possible is 512, but likely much lower.
            # Let's say if we get > 80, it's good.
            if best_len > 100:
                break

    return best_arr.astype(np.int64)

# EVOLVE-BLOCK-END
