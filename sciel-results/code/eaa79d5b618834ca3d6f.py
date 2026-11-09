import time
import random
import numpy as np

def construct():
    N = 512
    start_time = time.time()
    time_limit = 58.0  # Safety margin before 60s kill

    # Precompute all 4-APs as tuples of 0-indexed positions
    # For each element x, we need to know which 4-APs contain x
    # And for each 4-AP, the other 3 elements

    # Strategy: Use a different representation.
    # Represent S as a numpy boolean array or use integer bitmasks.
    # The key bottleneck is the can_add check.
    
    # Let's precompute for each x a list of "forbidden triples" - 
    # but that's what we had. The issue was speed.
    
    # Alternative: Use numpy for the conflict checking.
    # For each 4-AP, store the 4 positions.
    # Maintain a count array: for each 4-AP, how many of its elements are in S.
    # x can be added iff for all 4-APs containing x, the count of other elements in S is < 3.
    # Equivalently: x can be added iff for all 4-APs containing x, count[ap] + 1 < 4,
    # i.e., count[ap] < 3 for all APs containing x.
    
    # This is a constraint satisfaction approach with incremental updates.
    
    # Precompute:
    # aps: list of tuples (a, b, c, d) 0-indexed
    # ap_index: dict mapping (a,b,c,d) -> index
    # element_aps: for each x, list of ap indices containing x
    # ap_elements: for each ap index, the 4 elements
    
    aps = []
    element_aps = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        for a in range(N - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= N:
                break
            ap_idx = len(aps)
            aps.append((a, b, c, e))
            element_aps[a].append(ap_idx)
            element_aps[b].append(ap_idx)
            element_aps[c].append(ap_idx)
            element_aps[e].append(ap_idx)
    
    num_aps = len(aps)
    
    # Convert to numpy arrays for speed
    ap_array = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # For each element, we have a list of ap indices. Convert to numpy.
    element_aps_arrays = []
    for x in range(N):
        element_aps_arrays.append(np.array(element_aps[x], dtype=np.int32))
    
    def can_add_fast(x, in_set, ap_counts):
        """Check if x can be added. in_set is a bool numpy array, ap_counts is int array."""
        if in_set[x]:
            return False
        # For each AP containing x, check if the other 3 are all in S
        # i.e., ap_counts[ap] == 3 for some ap containing x
        # Because if 3 of the 4 are in S, and x is the 4th, adding x completes the AP.
        # Wait: ap_counts[ap] is the number of elements of that AP currently in S.
        # If ap_counts[ap] == 3, then the 4th element (which is x) would complete it.
        # But we need to check: is x the missing one? 
        # Actually, if x is in the AP and ap_counts[ap] == 3, then the other 3 are in S,
        # so adding x would create a 4-AP. So x cannot be added.
        # If ap_counts[ap] < 3 for all ap containing x, then x can be added.
        if len(element_aps_arrays[x]) == 0:
            return True
        counts_for_x = ap_counts[element_aps_arrays[x]]
        return np.all(counts_for_x < 3)
    
    def greedy_construct(seed):
        rng = random.Random(seed)
        in_set = np.zeros(N, dtype=bool)
        ap_counts = np.zeros(num_aps, dtype=np.int32)
        
        elements = list(range(N))
        rng.shuffle(elements)
        
        for x in elements:
            if can_add_fast(x, in_set, ap_counts):
                in_set[x] = True
                # Update ap_counts for all APs containing x
                for ap_idx in element_aps[x]:
                    ap_counts[ap_idx] += 1
        
        return in_set
    
    def greedy_construct_ordered(seed):
        """Process elements in order of fewest APs first."""
        rng = random.Random(seed)
        in_set = np.zeros(N, dtype=bool)
        ap_counts = np.zeros(num_aps, dtype=np.int32)
        
        # Sort by number of APs, with random tie-breaking
        elements = sorted(range(N), key=lambda x: (len(element_aps[x]), rng.random()))
        
        for x in elements:
            if can_add_fast(x, in_set, ap_counts):
                in_set[x] = True
                for ap_idx in element_aps[x]:
                    ap_counts[ap_idx] += 1
        
        return in_set
    
    def local_search(in_set, rng, time_budget):
        """Local search: remove random elements, then greedy fill."""
        ap_counts = np.zeros(num_aps, dtype=np.int32)
        for x in range(N):
            if in_set[x]:
                for ap_idx in element_aps[x]:
                    ap_counts[ap_idx] += 1
        
        current_size = np.sum(in_set)
        best_size = current_size
        best_in_set = in_set.copy()
        
        all_elements = list(range(N))
        
        while time.time() - start_time < time_budget:
            # Remove 1-3 random elements from the set
            num_remove = rng.randint(1, 3)
            removed = []
            for _ in range(num_remove):
                if current_size == 0:
                    break
                # Pick a random element that's in the set
                x_remove = rng.choice(all_elements)
                if not in_set[x_remove]:
                    continue
                in_set[x_remove] = False
                removed.append(x_remove)
                current_size -= 1
                for ap_idx in element_aps[x_remove]:
                    ap_counts[ap_idx] -= 1
            
            if not removed:
                continue
            
            # Greedy fill
            candidates = all_elements[:]
            rng.shuffle(candidates)
            
            for x in candidates:
                if can_add_fast(x, in_set, ap_counts):
                    in_set[x] = True
                    current_size += 1
                    for ap_idx in element_aps[x]:
                        ap_counts[ap_idx] += 1
            
            if current_size > best_size:
                best_size = current_size
                best_in_set = in_set.copy()
        
        # Final fill pass
        for x in all_elements:
            if can_add_fast(x, best_in_set, ap_counts):
                best_in_set[x] = True
                best_size += 1
                for ap_idx in element_aps[x]:
                    ap_counts[ap_idx] += 1
        
        return best_in_set, best_size
    
    def final_fill(in_set, ap_counts):
        """Try to add any remaining valid elements."""
        size = np.sum(in_set)
        for x in range(N):
            if not in_set[x] and can_add_fast(x, in_set, ap_counts):
                in_set[x] = True
                size += 1
                for ap_idx in element_aps[x]:
                    ap_counts[ap_idx] += 1
        return in_set, size
    
    # Main
    best_in_set = None
    best_size = 0
    
    rng = random.Random(42)
    
    # Initial greedy (random order)
    in_set = greedy_construct(0)
    size = np.sum(in_set)
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
    
    # Initial greedy (conflict-degree ordered)
    if time.time() - start_time < time_limit - 10:
        in_set = greedy_construct_ordered(0)
        size = np.sum(in_set)
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
    
    # Local search on best so far
    if time.time() - start_time < time_limit - 5:
        in_set, size = local_search(best_in_set, rng, time_limit - 5)
        if size > best_size:
            best_size = size
            best_in_set = in_set
    
    # More restarts
    seed = 1
    while time.time() - start_time < time_limit - 2:
        if seed % 2 == 0:
            in_set = greedy_construct(seed)
        else:
            in_set = greedy_construct_ordered(seed)
        size = np.sum(in_set)
        
        if size >= best_size - 5:
            in_set, size = local_search(in_set, rng, time_limit - 2)
        
        if size > best_size:
            best_size = size
            best_in_set = in_set
        
        seed += 1
    
    # Convert to list of 1-indexed integers
    result = [i + 1 for i in range(N) if best_in_set[i]]
    return result
