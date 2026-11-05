"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 290  # safety margin before 330s kill
    
    n_pts = N * N
    all_points = [(x, y) for y in range(N) for x in range(N)]
    pts_arr = np.array(all_points, dtype=np.int32)
    
    # Precompute all pairwise squared distances
    diff = pts_arr[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
    sq_dists = (diff ** 2).sum(axis=2).astype(np.int64)
    
    # For each point i and distance d, store which points j have sq_dists[i,j] == d
    # This is expensive to build fully, so we'll use it on-demand with caching
    # Actually, let's precompute distance lists for each point
    # dist_lists[i] = list of (distance, point_j) for all j != i
    dist_lists = []
    for i in range(n_pts):
        d = sq_dists[i].copy()
        d[i] = -1  # mark self
        # Sort by distance
        order = np.argsort(d)
        dist_lists.append([(int(d[order[k]]), int(order[k])) for k in range(n_pts) if d[order[k]] > 0])
    
    def can_add(idx, S_set, dist_to_S):
        """
        Check if point idx can be added to S.
        S_set: set of indices in S
        dist_to_S: dict point_idx -> set of distances from that point to points in S
        Returns True if can add, and the distances from idx to points in S
        """
        if not S_set:
            return True, set()
        
        # Get distances from idx to all points in S
        S_arr = np.array(list(S_set), dtype=np.int64)
        dists = sq_dists[idx, S_arr]
        dist_list = [(int(dists[k]), int(S_arr[k])) for k in range(len(S_arr))]
        
        # Check 1: all distances from idx to points in S must be distinct
        # (otherwise idx is apex of isosceles)
        seen_dists = set()
        for d, j in dist_list:
            if d in seen_dists:
                return False, None
            seen_dists.add(d)
        
        # Check 2: for each point j in S, dist(idx, j) must not equal dist(j, k) for any k in S
        # i.e., j cannot be apex of isosceles with idx and some k in S
        for d, j in dist_list:
            if d in dist_to_S[j]:
                return False, None
        
        return True, seen_dists
    
    def add_point(idx, S_list, S_set, dist_to_S, dists_from_idx):
        """Add point idx to S and update structures."""
        S_list.append(idx)
        S_set.add(idx)
        dist_to_S[idx] = set(dists_from_idx)
        # Update dist_to_S for all existing points
        for j in S_list:
            if j == idx:
                continue
            d = int(sq_dists[idx, j])
            dist_to_S[j].add(d)
    
    def greedy_run(order, start_S_list=None, start_S_set=None, start_dist_to_S=None):
        """Greedily construct/expand a valid set using the given order."""
        if start_S_list is not None:
            S_list = list(start_S_list)
            S_set = set(start_S_set) if start_S_set is not None else set(start_S_list)
            dist_to_S = {k: set(v) for k, v in start_dist_to_S.items()}
        else:
            S_list = []
            S_set = set()
            dist_to_S = {}
        
        for idx in order:
            if time.time() > deadline:
                break
            if idx in S_set:
                continue
            
            valid, dists_from_idx = can_add(idx, S_set, dist_to_S)
            if valid:
                add_point(idx, S_list, S_set, dist_to_S, dists_from_idx)
        
        return S_list, S_set, dist_to_S
    
    def local_search(S_list, S_set, dist_to_S, rng, time_budget):
        """
        Local search: try removing one point and greedily refilling with better candidates.
        """
        best_S_list = list(S_list)
        best_S_set = set(S_set)
        best_dist_to_S = {k: set(v) for k, v in dist_to_S.items()}
        best_size = len(best_S_list)
        
        if time.time() > deadline - 5:
            return best_S_list, best_S_set, best_dist_to_S
        
        removal_order = list(range(len(S_list)))
        rng.shuffle(removal_order)
        
        for pos in removal_order:
            if time.time() > deadline - 5:
                break
            
            removed_idx = S_list[pos]
            
            # Build reduced set
            temp_S_list = [S_list[i] for i in range(len(S_list)) if i != pos]
            temp_S_set = set(temp_S_list)
            temp_dist_to_S = {}
            for idx in temp_S_list:
                ds = set()
                for j in temp_S_list:
                    if j != idx:
                        ds.add(int(sq_dists[idx, j]))
                temp_dist_to_S[idx] = ds
            
            # Try to expand: pick candidates not in temp_S_set
            candidates = [i for i in range(n_pts) if i not in temp_S_set]
            rng.shuffle(candidates)
            
            # Limit candidates to save time
            max_candidates = min(len(candidates), 300)
            candidates = candidates[:max_candidates]
            
            expanded_S_list, expanded_S_set, expanded_dist_to_S = greedy_run(
                candidates, temp_S_list, temp_S_set, temp_dist_to_S
            )
            
            if len(expanded_S_list) > best_size:
                best_S_list = expanded_S_list
                best_S_set = expanded_S_set
                best_dist_to_S = expanded_dist_to_S
                best_size = len(best_S_list)
        
        return best_S_list, best_S_set, best_dist_to_S
    
    # Generate index order for greedy
    all_indices = list(range(n_pts))
    
    best_S_list = []
    best_S_set = set()
    best_dist_to_S = {}
    best_size = 0
    
    rng = random.Random(42)
    
    # Phase 1: Multiple randomized greedy runs
    num_greedy_runs = 15
    for i in range(num_greedy_runs):
        if time.time() > deadline - 30:
            break
        
        order = all_indices.copy()
        rng.shuffle(order)
        
        S_list, S_set, dist_to_S = greedy_run(order)
        
        if len(S_list) > best_size:
            best_S_list = S_list
            best_S_set = S_set
            best_dist_to_S = dist_to_S
            best_size = len(S_list)
    
    # Phase 2: Local search refinement
    if time.time() < deadline - 10 and best_size > 0:
        for ls_iter in range(5):
            if time.time() > deadline - 5:
                break
            
            new_S_list, new_S_set, new_dist_to_S = local_search(
                best_S_list, best_S_set, best_dist_to_S, rng, 10
            )
            
            if len(new_S_list) > best_size:
                best_S_list = new_S_list
                best_S_set = new_S_set
                best_dist_to_S = new_dist_to_S
                best_size = len(new_S_list)
            else:
                break
    
    # Convert indices back to (x, y) tuples
    result = [all_points[idx] for idx in best_S_list]
    
    return result
# EVOLVE-BLOCK-END
