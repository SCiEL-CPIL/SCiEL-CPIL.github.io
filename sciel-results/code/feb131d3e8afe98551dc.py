import numpy as np
import itertools
import time
from scipy.optimize import linprog

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 620.0
    
    # Precompute grid
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    sq = (grid ** 2).sum(axis=1).astype(np.int64)
    
    # Precompute combination indices for various sizes
    combo_cache = {}
    def get_combo_indices(m):
        if m not in combo_cache:
            combo_cache[m] = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int64)
        return combo_cache[m]
    
    def can_add_point(set_indices, new_idx):
        """
        Check if new_idx can be added to set_indices without creating 5 co-spherical points.
        """
        m = len(set_indices)
        if m < 4:
            return True
        
        p = grid[new_idx]
        p_sq = sq[new_idx]
        
        pts = grid[set_indices]
        sqs = sq[set_indices]
        
        W = np.empty((m, 4), dtype=np.int64)
        W[:, 0:3] = pts - p
        W[:, 3] = sqs - p_sq
        
        combo_idx = get_combo_indices(m)
        # combo_idx shape: (C(m,4), 4)
        
        # Batch determinant computation
        # Extract submatrices: W[combo_idx] gives shape (C(m,4), 4, 4)
        submats = W[combo_idx]  # shape (k, 4, 4)
        
        # Compute determinants in batch
        dets = np.linalg.det(submats.astype(np.float64))
        
        if np.any(np.abs(dets) < 0.5):
            return False
        return True
    
    def greedy_build(order):
        current_set = []
        current_set_set = set()
        
        for idx in order:
            if idx in current_set_set:
                continue
            if can_add_point(np.array(current_set, dtype=np.int64), idx):
                current_set.append(idx)
                current_set_set.add(idx)
        
        return np.array(current_set, dtype=np.int64)
    
    def local_search(initial_set):
        current = initial_set.copy()
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        rng = np.random.RandomState(123)
        
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            
            # Phase 1: Try to add points
            for idx in all_indices:
                if idx in current_set:
                    continue
                if can_add_point(current, idx):
                    current = np.append(current, idx)
                    current_set.add(idx)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Phase 2: Try swaps
            order_of_removal = rng.permutation(len(current))
            for i_idx in order_of_removal:
                if time.time() > deadline - 10.0:
                    break
                
                removed_idx = current[i_idx]
                without = np.delete(current, i_idx)
                
                comp_indices = [i for i in all_indices if i not in current_set]
                rng.shuffle(comp_indices)
                
                for j_idx in comp_indices[:100]:
                    if time.time() > deadline - 10.0:
                        break
                    if can_add_point(without, j_idx):
                        current = np.append(without, j_idx)
                        current_set.remove(removed_idx)
                        current_set.add(j_idx)
                        improved = True
                        break
                if improved:
                    break
        
        return current
    
    best_set = None
    best_len = 0
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 15.0:
        order = rng.permutation(N)
        initial_set = greedy_build(order)
        
        if len(initial_set) > best_len:
            best_set = initial_set
            best_len = len(initial_set)
        
        if time.time() < deadline - 25.0:
            improved_set = local_search(initial_set)
            if len(improved_set) > best_len:
                best_set = improved_set
                best_len = len(improved_set)
    
    if best_set is None:
        best_set = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    
    return grid[best_set].astype(np.int64)
