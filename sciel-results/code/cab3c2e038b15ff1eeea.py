# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Find the maximum number of points from the 8x8x8 integer grid {0,...,7}^3
    such that no five points lie on a common sphere or plane.
    
    Uses simulated annealing with vectorized determinant checks to escape
    local minima and find larger valid configurations.
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute squared norms for each grid point
    sq_norms = (grid ** 2).sum(axis=1)  # shape (N,)
    
    # Precompute combination indices for 4-tuples of grid indices
    # We'll use these for vectorized determinant checks
    
    def can_add_vectorized(arr_indices: np.ndarray, p_idx: int) -> bool:
        """Check if adding grid point p_idx keeps the set sphere-free."""
        m = len(arr_indices)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sq = sq_norms[p_idx]
        
        # Get existing points and their squared norms
        existing = grid[arr_indices]  # (m, 3)
        existing_sq = sq_norms[arr_indices]  # (m,)
        
        # Build lifted matrix: rows are (q - p, |q|^2 - |p|^2)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = existing - p
        w[:, 3] = existing_sq - p_sq
        
        # Check all 4-tuples
        if m >= 4:
            combos = np.array(list(itertools.combinations(range(m), 4)))
            if len(combos) > 0:
                sub = w[combos]  # (C, 4, 4)
                dets = np.linalg.det(sub)
                if np.any(np.abs(dets) < 0.5):
                    return False
        return True
    
    def count_violations(arr_indices: np.ndarray) -> int:
        """Count the number of co-spherical 5-tuples in the set."""
        m = len(arr_indices)
        if m < 5:
            return 0
        
        p = grid[arr_indices]  # (m, 3)
        p_sq = sq_norms[arr_indices]  # (m,)
        
        violations = 0
        # Check all 5-tuples
        for combo in itertools.combinations(range(m), 5):
            pts = p[combo]
            sqs = p_sq[combo]
            # Build 5x5 matrix with rows [x, y, z, x^2+y^2+z^2, 1]
            mat = np.column_stack([pts, sqs, np.ones(5)])
            if abs(np.linalg.det(mat)) < 0.5:
                violations += 1
        return violations
    
    def energy(arr_indices: np.ndarray) -> float:
        """Energy = number of violations (0 if valid)."""
        return count_violations(arr_indices)
    
    # Initial greedy construction
    np.random.seed(42)
    order = np.random.permutation(N)
    arr = np.empty(0, dtype=np.int64)
    for p_idx in order:
        if can_add_vectorized(arr, p_idx):
            arr = np.append(arr, p_idx) if len(arr) > 0 else np.array([p_idx], dtype=np.int64)
    
    best_arr = arr.copy()
    best_energy = energy(arr)
    
    # Simulated annealing
    start_time = time.time()
    deadline = 580.0  # Leave 50s safety margin
    T = 10.0  # Initial temperature
    T_min = 0.001
    cooling = 0.9995
    
    current_arr = best_arr.copy()
    current_energy = best_energy
    
    while time.time() < deadline:
        if len(current_arr) < 5:
            break
        
        # Choose a random move: add, remove, or swap
        move_type = np.random.random()
        
        if move_type < 0.3 and len(current_arr) < N:
            # Add a random point
            candidates = np.setdiff1d(np.arange(N), current_arr)
            if len(candidates) == 0:
                continue
            new_idx = candidates[np.random.randint(len(candidates))]
            new_arr = np.append(current_arr, new_idx)
            new_energy = energy(new_arr)
        elif move_type < 0.6 and len(current_arr) > 0:
            # Remove a random point
            rm_pos = np.random.randint(len(current_arr))
            new_arr = np.delete(current_arr, rm_pos)
            new_energy = energy(new_arr)
        else:
            # Swap: remove one, add another
            if len(current_arr) == 0 or len(current_arr) == N:
                continue
            rm_pos = np.random.randint(len(current_arr))
            old_idx = current_arr[rm_pos]
            candidates = np.setdiff1d(np.arange(N), current_arr)
            if len(candidates) == 0:
                continue
            new_idx = candidates[np.random.randint(len(candidates))]
            new_arr = np.delete(current_arr, rm_pos)
            new_arr = np.append(new_arr, new_idx)
            new_energy = energy(new_arr)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / max(T, T_min)):
            current_arr = new_arr
            current_energy = new_energy
            
            # Update best if we found a valid (or better) solution
            if current_energy == 0:
                if len(current_arr) > len(best_arr):
                    best_arr = current_arr.copy()
                    best_energy = current_energy
            elif current_energy < best_energy:
                if len(current_arr) >= len(best_arr) or current_energy < best_energy:
                    best_arr = current_arr.copy()
                    best_energy = current_energy
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # If we've found a large valid set and temperature is low, try local search
        if current_energy == 0 and T < 0.1:
            # Try to add more points greedily
            improved = True
            while improved:
                improved = False
                candidates = np.setdiff1d(np.arange(N), current_arr)
                for cand in candidates:
                    if can_add_vectorized(current_arr, cand):
                        new_arr = np.append(current_arr, cand)
                        if energy(new_arr) == 0:
                            current_arr = new_arr
                            best_arr = current_arr.copy()
                            improved = True
                            break
    
    # Final local search: try to add more points to best_arr
    final_arr = best_arr.copy()
    improved = True
    while improved:
        improved = False
        candidates = np.setdiff1d(np.arange(N), final_arr)
        for cand in candidates:
            if can_add_vectorized(final_arr, cand):
                new_arr = np.append(final_arr, cand)
                if energy(new_arr) == 0:
                    final_arr = new_arr
                    improved = True
                    break
    
    # Convert indices to points
    result = grid[final_arr]
    return result.astype(np.int64)


# EVOLVE-BLOCK-END
