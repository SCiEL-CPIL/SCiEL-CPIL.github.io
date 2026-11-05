"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 280  # safety margin before 330s kill
    
    n_pts = N * N
    all_points = [(x, y) for y in range(N) for x in range(N)]
    pts_arr = np.array(all_points, dtype=np.int32)
    
    # Precompute all pairwise squared distances: sq_dists[i, j]
    diff = pts_arr[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
    sq_dists = (diff ** 2).sum(axis=2).astype(np.int64)
    
    def can_add_fast(idx, S_indices, dist_sets):
        """
        Fast check if point idx can be added to S_indices.
        S_indices: list of indices in S
        dist_sets: dict idx -> set of squared distances from idx to points in S
        """
        if not S_indices:
            return True, set()
        
        S_arr = np.array(S_indices, dtype=np.int64)
        dists = sq_dists[idx, S_arr]
        
        # Check 1: all distances from idx to points in S must be distinct
        if len(set(dists.tolist())) != len(dists):
            return False, None
        
        # Check 2: for each point j in S, dist(idx, j) must not be in dist_sets[j]
        for i, j in enumerate(S_indices):
            d = int(dists[i])
            if d in dist_sets[j]:
                return False, None
        
        return True, set(dists.tolist())
    
    def add_point_fast(idx, S_indices, dist_sets, dists_to_idx):
        """Add point idx to S and update dist_sets."""
        S_indices.append(idx)
        dist_sets[idx] = dists_to_idx
        for i, j in enumerate(S_indices):
            if j == idx:
                continue
            d = int(sq_dists[idx, j])
            dist_sets[j].add(d)
    
    def greedy_run(order, start_S=None, start_dist_sets=None):
        """Greedily construct/expand a valid set using the given order."""
        if start_S is not None:
            S_indices = list(start_S)
            dist_sets = {k: set(v) for k, v in start_dist_sets.items()}
        else:
            S_indices = []
            dist_sets = {}
        
        for idx in order:
            if time.time() > deadline:
                break
            if idx in dist_sets:
                continue
            
            valid, dists_to_idx = can_add_fast(idx, S_indices, dist_sets)
            if valid:
                add_point_fast(idx, S_indices, dist_sets, dists_to_idx)
        
        return S_indices, dist_sets
    
    def local_search(S_indices, dist_sets, rng, time_budget):
        """
        Local search: try removing one point and greedily refilling.
        Returns improved (S_indices, dist_sets) or original if no improvement.
        """
        best_S = list(S_indices)
        best_ds = {k: set(v) for k, v in dist_sets.items()}
        best_size = len(best_S)
        
        if time.time() > deadline - 5:
            return best_S, best_ds
        
        # Try each point removal
        removal_order = list(range(len(S_indices)))
        rng.shuffle(removal_order)
        
        for pos in removal_order:
            if time.time() > deadline - 5:
                break
            
            removed_idx = S_indices[pos]
            
            # Build reduced set
            temp_S = [S_indices[i] for i in range(len(S_indices)) if i != pos]
            temp_ds = {}
            for idx in temp_S:
                # Recompute distances from idx to other points in temp_S
                ds = set()
                for j in temp_S:
                    if j != idx:
                        ds.add(int(sq_dists[idx, j]))
                temp_ds[idx] = ds
            
            # Try to expand: shuffle all points not in temp_S and greedily add
            temp_S_set = set(temp_S)
            candidates = [i for i in range(n_pts) if i not in temp_S_set]
            rng.shuffle(candidates)
            
            # Limit candidates to save time
            max_candidates = min(len(candidates), 500)
            candidates = candidates[:max_candidates]
            
            expanded_S, expanded_ds = greedy_run(candidates, temp_S, temp_ds)
            
            if len(expanded_S) > best_size:
                best_S = expanded_S
                best_ds = expanded_ds
                best_size = len(expanded_S)
        
        return best_S, best_ds
    
    # Generate index order for greedy
    all_indices = list(range(n_pts))
    
    best_S = []
    best_ds = {}
    best_size = 0
    
    rng = random.Random(42)
    
    # Phase 1: Multiple randomized greedy runs
    num_greedy_runs = 10
    for i in range(num_greedy_runs):
        if time.time() > deadline - 30:
            break
        
        order = all_indices.copy()
        rng.shuffle(order)
        
        S, ds = greedy_run(order)
        
        if len(S) > best_size:
            best_S = S
            best_ds = ds
            best_size = len(S)
    
    # Phase 2: Local search refinement
    if time.time() < deadline - 10 and best_size > 0:
        # Run local search a few times
        for ls_iter in range(3):
            if time.time() > deadline - 5:
                break
            
            new_S, new_ds = local_search(best_S, best_ds, rng, 10)
            
            if len(new_S) > best_size:
                best_S = new_S
                best_ds = new_ds
                best_size = len(new_S)
            else:
                break  # No improvement, stop
    
    # Convert indices back to (x, y) tuples
    result = [all_points[idx] for idx in best_S]
    
    return result
# EVOLVE-BLOCK-END
