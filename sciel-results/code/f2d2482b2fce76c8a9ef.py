import time
import numpy as np
import itertools
import random

D = 11
DEADLINE = 140.0

def _generate_candidates_by_weight():
    """Generate all integer vectors in R^11 with squared norm 4."""
    candidates_w4 = []
    for i, j, k, l in itertools.combinations(range(D), 4):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    for sl in (1, -1):
                        v = [0] * D
                        v[i] = si
                        v[j] = sj
                        v[k] = sk
                        v[l] = sl
                        candidates_w4.append(tuple(v))
    return candidates_w4

_W4 = _generate_candidates_by_weight()
_N_W4 = len(_W4)

# Precompute dot products and conflict bitsets
# Conflict if dot > 2 (i.e., dot >= 3)
_MAT = np.array(_W4, dtype=np.int32)
_DOT = _MAT @ _MAT.T

# Build bitsets for neighbors
# neighbors[i] is a bitmask where bit j is 1 if i and j conflict
_NEIGHBORS = [0] * _N_W4
for i in range(_N_W4):
    mask = 0
    # Vectorized: find j where _DOT[i, j] > 2
    # But we need to set bits in a Python int
    # Loop is faster for setting bits in Python int than converting numpy array to int
    # However, 5280 * 5280 is ~28M ops.
    # Let's use numpy to find indices, then set bits
    idxs = np.where(_DOT[i] > 2)[0]
    for j in idxs:
        mask |= (1 << j)
    # Exclude self
    mask &= ~(1 << i)
    _NEIGHBORS[i] = mask

def _greedy_mis_bitset(n, neighbors, rng, time_budget=1.0):
    """
    Greedy Maximal Independent Set using bitsets.
    """
    start = time.time()
    available = (1 << n) - 1
    selected = []
    
    while available:
        if time.time() - start > time_budget:
            break
        
        # Find a node in available with minimum degree in the available subgraph
        # Degree of node i in available subgraph = popcount(neighbors[i] & available)
        # We want to minimize this.
        # Iterating over all available nodes is O(n).
        # n=5280.
        # We can sample or just iterate.
        
        # Optimization: If we just pick a random node from available, it's O(1) but quality is lower.
        # Min-degree is better.
        
        # To find min-degree efficiently:
        # We can iterate through available nodes.
        # available is a bitmask.
        
        min_deg = n + 1
        best_node = -1
        
        # Iterate through set bits in available
        # Using a while loop on the mask
        temp_avail = available
        while temp_avail:
            # Get lowest set bit
            lsb = temp_avail & -temp_avail
            i = lsb.bit_length() - 1
            
            # Calculate degree of i in available
            deg = (neighbors[i] & available).bit_count()
            
            if deg < min_deg:
                min_deg = deg
                best_node = i
                
            # If min_deg is 0, we can break early?
            # No, there might be another 0, but any 0 is fine.
            if min_deg == 0:
                break
                
            temp_avail ^= lsb
            
        if best_node == -1:
            break
            
        selected.append(best_node)
        
        # Remove best_node and its neighbors from available
        # available &= ~(neighbors[best_node] | (1 << best_node))
        available &= ~(neighbors[best_node] | (1 << best_node))
        
    return selected

def _local_search_bitset(selected, n, neighbors, rng, time_budget=5.0):
    """
    Local search: try to add nodes, then swap.
    """
    start = time.time()
    selected_set = set(selected)
    # Precompute union of neighbors of selected nodes
    # This represents nodes that CANNOT be added
    blocked = 0
    for s in selected_set:
        blocked |= neighbors[s]
    
    # Nodes that are not selected and not blocked are "addable"
    # addable = ~(blocked | selected_mask)
    
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        
        # Phase 1: Add
        selected_mask = 0
        for s in selected_set:
            selected_mask |= (1 << s)
            
        addable = ((1 << n) - 1) & ~(blocked | selected_mask)
        
        if addable:
            # Pick a random addable node
            # Get list of set bits
            temp = addable
            indices = []
            while temp:
                lsb = temp & -temp
                indices.append(lsb.bit_length() - 1)
                temp ^= lsb
            
            if indices:
                # Pick one with min degree in addable? Or just random.
                # Random is faster.
                node = indices[rng.randint(0, len(indices)-1)]
                selected_set.add(node)
                blocked |= neighbors[node]
                improved = True
                continue
        
        # Phase 2: Swap
        # Try to remove a node s and add multiple nodes
        # This is complex with bitsets.
        # Simple swap: remove s, see if we can add 1 node.
        # If we add 1, net change 0. Not useful unless we can chain.
        # Better: Remove s, recalculate blocked, try to add greedily.
        
        if selected_set:
            # Pick a random selected node to try removing
            s_list = list(selected_set)
            s = s_list[rng.randint(0, len(s_list)-1)]
            
            # Temporarily remove s
            selected_set.discard(s)
            
            # Recalculate blocked
            temp_blocked = 0
            for t in selected_set:
                temp_blocked |= neighbors[t]
            
            temp_selected_mask = 0
            for t in selected_set:
                temp_selected_mask |= (1 << t)
                
            temp_addable = ((1 << n) - 1) & ~(temp_blocked | temp_selected_mask)
            
            # Greedily add as many as possible from temp_addable
            new_selected = set(selected_set)
            new_blocked = temp_blocked
            added = 0
            current_addable = temp_addable
            
            while current_addable:
                # Pick min degree?
                # Just pick random for speed
                temp = current_addable
                indices = []
                while temp:
                    lsb = temp & -temp
                    indices.append(lsb.bit_length() - 1)
                    temp ^= lsb
                    
                if not indices:
                    break
                    
                node = indices[rng.randint(0, len(indices)-1)]
                new_selected.add(node)
                new_blocked |= neighbors[node]
                current_addable &= ~neighbors[node]
                current_addable &= ~(1 << node)
                added += 1
                
            if added > 1: # Must gain at least 1 net (added > 1 because we removed 1)
                # Actually, if added == 1, we replaced s with node. Size same.
                # We want to increase size. So added >= 2?
                # If we remove 1 and add 1, size is same.
                # If we remove 1 and add 2, size increases by 1.
                # Let's accept if added >= 1? No, we want to maximize size.
                # If added == 1, we might have improved "quality" but not size.
                # For now, only accept if size increases.
                if added > 1:
                    selected_set = new_selected
                    blocked = new_blocked
                    improved = True
                else:
                    # Rollback
                    selected_set.add(s)
                    # Recalculate blocked? Or just keep old?
                    # Old blocked was valid for old set.
                    # We modified selected_set to old state.
                    # But we didn't update blocked during the trial.
                    # So blocked is still correct for the old set.
                    pass
            else:
                # Rollback
                selected_set.add(s)

    return list(selected_set)

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_selected = []
    best_count = 0
    
    # Phase 1: Greedy MIS with restarts
    for restart in range(100):
        if time.time() - start_time > DEADLINE - 20:
            break
        selected = _greedy_mis_bitset(_N_W4, _NEIGHBORS, rng, time_budget=0.5)
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected
            
    # Phase 2: Local Search on best
    if best_selected:
        improved = _local_search_bitset(best_selected, _N_W4, _NEIGHBORS, rng, time_budget=20.0)
        if len(improved) > best_count:
            best_count = len(improved)
            best_selected = improved

    # Phase 3: More restarts
    for restart in range(100):
        if time.time() - start_time > DEADLINE - 5:
            break
        rng2 = random.Random(1000 + restart)
        selected = _greedy_mis_bitset(_N_W4, _NEIGHBORS, rng2, time_budget=0.5)
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = selected
        else:
            # Try local search on this one too if it's close?
            if len(selected) > best_count - 5 and time.time() - start_time < DEADLINE - 10:
                improved = _local_search_bitset(selected, _N_W4, _NEIGHBORS, rng2, time_budget=1.0)
                if len(improved) > best_count:
                    best_count = len(improved)
                    best_selected = improved

    # Build result
    if best_selected:
        result = np.array([_W4[i] for i in best_selected], dtype=np.int64)
    else:
        result = np.array([_W4[0:10]], dtype=np.int64)
    
    return result
