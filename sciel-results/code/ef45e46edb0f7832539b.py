"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _build_bitmasks():
    """Build bitmask adjacency for fast degree counting."""
    n = P['n']
    masks = [0] * n
    for u, v in P['edges']:
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks

def _degree_count_fast(subset_mask, masks):
    """Return number of distinct degrees using bitmask operations."""
    degrees = set()
    for v in range(17):
        if subset_mask & (1 << v):
            deg = bin(masks[v] & subset_mask).count('1')
            degrees.add(deg)
    return len(degrees)

def _objective(subset, adj):
    """Higher is better: number of distinct degrees."""
    if len(subset) < 8:
        return -1
    s = set(subset)
    degrees = []
    for v in s:
        deg = sum(1 for u in s if u in adj[v] and u != v)
        degrees.append(deg)
    return len(set(degrees))

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    adj = _build_adj()
    masks = _build_bitmasks()
    n = P['n']
    
    best_score = -1
    best_subset = list(range(8))
    best_mask = 0
    
    # Try all subsets of size 8 (24310 combos) - fast enough
    for subset in itertools.combinations(range(n), 8):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        score = _degree_count_fast(mask, masks)
        if score > best_score:
            best_score = score
            best_subset = list(subset)
            best_mask = mask
    
    # Now try larger sizes with randomized local search using bitmask operations
    rng = random.Random(42)
    
    for target_size in range(9, 18):
        # Check time
        if time.time() > deadline:
            break
        
        # Initialize with random subset of target_size
        for restart in range(500):
            if time.time() > deadline:
                break
            
            # Start from best_subset if size matches, else random
            if len(best_subset) == target_size:
                current = list(best_subset)
                current_mask = best_mask
            else:
                current = rng.sample(range(n), target_size)
                current_mask = 0
                for v in current:
                    current_mask |= (1 << v)
            current_score = _degree_count_fast(current_mask, masks)
            
            # Local search: try adding/removing/swapping vertices
            improved = True
            while improved and time.time() < deadline:
                improved = False
                
                # Generate candidates: swap one vertex in with one out
                in_set = set(current)
                out_set = set(range(n)) - in_set
                
                if not out_set:
                    break
                
                # Try all single swaps
                best_swap_score = current_score
                best_swap = None
                
                for v_in in out_set:
                    for v_out in in_set:
                        new_mask = current_mask ^ (1 << v_out) | (1 << v_in)
                        new_score = _degree_count_fast(new_mask, masks)
                        if new_score > best_swap_score:
                            best_swap_score = new_score
                            best_swap = (v_in, v_out, new_mask)
                
                if best_swap is not None:
                    v_in, v_out, new_mask = best_swap
                    current = [v if v != v_out else v_in for v in current]
                    current_mask = new_mask
                    current_score = best_swap_score
                    improved = True
                else:
                    # Try 2-swaps: remove two, add two
                    if len(out_set) >= 2 and len(in_set) >= 2:
                        out_list = list(out_set)
                        in_list = list(in_set)
                        rng.shuffle(out_list)
                        rng.shuffle(in_list)
                        
                        for i in range(min(len(out_list), 10)):
                            for j in range(min(len(in_list), 10)):
                                if i == j:
                                    continue
                                v_out1, v_out2 = out_list[i], out_list[j]
                                v_in1, v_in2 = in_list[i], in_list[j]
                                
                                new_mask = current_mask
                                new_mask &= ~(1 << v_out1)
                                new_mask &= ~(1 << v_out2)
                                new_mask |= (1 << v_in1)
                                new_mask |= (1 << v_in2)
                                
                                new_score = _degree_count_fast(new_mask, masks)
                                if new_score > current_score:
                                    # Apply the 2-swap
                                    current = []
                                    for v in current:
                                        if v == v_out1:
                                            current.append(v_in1)
                                        elif v == v_out2:
                                            current.append(v_in2)
                                        else:
                                            current.append(v)
                                    current_mask = new_mask
                                    current_score = new_score
                                    improved = True
                                    break
                            if improved:
                                break
                        if not improved:
                            break
            
            if current_score > best_score:
                best_score = current_score
                best_subset = list(current)
                best_mask = current_mask
    
    # Final polish: try to improve by adding vertices if possible
    if time.time() < deadline and len(best_subset) < n:
        best_mask = 0
        for v in best_subset:
            best_mask |= (1 << v)
        
        for _ in range(100):
            if time.time() > deadline:
                break
            out_set = set(range(n)) - set(best_subset)
            if not out_set:
                break
            best_add = None
            best_add_score = _degree_count_fast(best_mask, masks)
            for v in out_set:
                new_mask = best_mask | (1 << v)
                new_score = _degree_count_fast(new_mask, masks)
                if new_score > best_add_score:
                    best_add_score = new_score
                    best_add = (v, new_mask)
            if best_add:
                v, new_mask = best_add
                best_subset.append(v)
                best_mask = new_mask
            else:
                break
    
    # Convert to proper types
    result = [int(v) for v in best_subset]
    return result

# EVOLVE-BLOCK-END
