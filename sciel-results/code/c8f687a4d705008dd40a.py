import numpy as np
import time
import random
import sys

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin

    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        r = (x * x) % m
        if r != 0:
            forbidden.add(r)
    
    # 2. Build Adjacency Bitsets
    # neighbors[i] is a bitset where bit j is 1 if i and j are adjacent (conflict)
    neighbors = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j1 = (i + d) % m
            j2 = (i - d) % m
            mask |= (1 << j1)
            mask |= (1 << j2)
        neighbors[i] = mask

    # 3. Branch and Bound Logic
    best_set = []
    best_size = 0
    
    def update_best(current_set):
        nonlocal best_set, best_size
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)

    # Greedy Coloring Bound
    # Returns an upper bound on the size of the MIS in the induced subgraph of candidates_mask
    def greedy_color_bound(candidates_mask):
        uncolored = candidates_mask
        bound = 0
        while uncolored:
            bound += 1
            # Greedily pick an independent set from uncolored
            # To make this fast, we pick vertices with low degree in the uncolored subgraph?
            # Or just pick one, remove it and its neighbors from the "available for this color" set?
            # Standard greedy coloring:
            # We want to assign color 'bound' to as many vertices as possible.
            # This is equivalent to finding a maximal independent set in the subgraph induced by uncolored.
            # Greedy MIS:
            # 1. Order vertices (e.g., by degree in uncolored, ascending is often good for packing, descending for covering? 
            #    For coloring, we want to color as many as possible. 
            #    Let's just pick a vertex, add to IS, remove it and neighbors from uncolored for this color layer?
            #    No, that's not coloring.
            #    Coloring: Assign color 1 to a maximal independent set. Remove them. Assign color 2 to next maximal IS.
            #    To find a maximal IS greedily:
            #    - Pick a vertex v from uncolored.
            #    - Add v to IS.
            #    - Remove v and its neighbors from the set of vertices *available for this color*.
            #    - Continue until no vertices available for this color.
            #    - Then start next color.
            
            available_for_color = uncolored
            while available_for_color:
                # Pick a vertex. Heuristic: pick one with min degree in available_for_color? 
                # Or just the first one?
                # Let's pick the one with min degree in the *uncolored* set to help future colors?
                # Actually, for the bound, we just need *a* valid coloring. 
                # A greedy approach: pick any vertex, say the first bit.
                # To improve the bound (make it tighter/smaller), we should try to pack more vertices into each color class.
                # This means finding a large independent set.
                # Greedy IS: pick vertex with min degree in available_for_color?
                
                # Simple: pick lowest index?
                # Better: pick vertex with min degree in available_for_color to leave more options?
                # Let's try min degree in available_for_color.
                
                # Get list of bits
                bits = []
                temp = available_for_color
                while temp:
                    lsb = temp & -temp
                    idx = lsb.bit_length() - 1
                    bits.append(idx)
                    temp ^= lsb
                    
                if not bits:
                    break
                    
                # Find min degree
                min_deg = 1000
                pivot = bits[0]
                for v in bits:
                    deg = (neighbors[v] & available_for_color).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        pivot = v
                        
                # Add pivot to this color class
                # Remove pivot and its neighbors from available_for_color
                available_for_color &= ~(1 << pivot)
                available_for_color &= ~neighbors[pivot]
                
            # Remove all vertices colored with this color from uncolored
            # The vertices colored are those removed from uncolored during the inner loop?
            # No, we tracked available_for_color. 
            # The vertices assigned color 'bound' are (uncolored ^ available_for_color)?
            # Yes, because we only removed from available_for_color.
            uncolored = available_for_color
            
        return bound

    def bnb(candidates_mask, current_set, depth):
        nonlocal best_size, best_set
        
        # Pruning: if current size + bound <= best_size, return
        if candidates_mask == 0:
            if len(current_set) > best_size:
                update_best(current_set)
            return

        if time.time() > deadline:
            return

        # Calculate Upper Bound
        ub = greedy_color_bound(candidates_mask)
        if len(current_set) + ub <= best_size:
            return

        # Get list of candidates
        cands = []
        temp = candidates_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            cands.append(idx)
            temp ^= lsb
            
        if not cands:
            if len(current_set) > best_size:
                update_best(current_set)
            return

        # Pivot Selection: Max Degree in candidates
        max_deg = -1
        pivots = []
        for i in cands:
            deg = (neighbors[i] & candidates_mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivots = [i]
            elif deg == max_deg:
                pivots.append(i)
        
        pivot = random.choice(pivots)

        # Branch 1: Include pivot
        new_cands_1 = candidates_mask & ~(1 << pivot) & ~neighbors[pivot]
        current_set.append(pivot)
        
        # Check bound before recursing?
        # The recursive call will check it. But we can check cheaply here.
        # If len(current_set) + 1 + (new_cands_1.bit_count()) <= best_size: 
        # But bit_count is weak. Let's rely on the recursive call's greedy_color_bound.
        
        bnb(new_cands_1, current_set, depth + 1)
        
        current_set.pop()

        # Branch 2: Exclude pivot
        new_cands_2 = candidates_mask & ~(1 << pivot)
        
        bnb(new_cands_2, current_set, depth + 1)

    # Phase 1: Greedy to get a good initial lower bound
    def greedy_random():
        remaining_mask = (1 << m) - 1
        selected = []
        while remaining_mask:
            cands = []
            temp = remaining_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                cands.append(idx)
                temp ^= lsb
            if not cands:
                break
            pivot = random.choice(cands)
            selected.append(pivot)
            remaining_mask &= ~(1 << pivot)
            remaining_mask &= ~neighbors[pivot]
        return selected

    for _ in range(100):
        if time.time() > deadline - 20:
            break
        gs = greedy_random()
        if len(gs) > best_size:
            best_size = len(gs)
            best_set = gs

    # Phase 2: Branch and Bound
    random.seed(42)
    
    full_mask = (1 << m) - 1
    
    # Run B&B
    # We can try multiple random seeds or just let the random pivot tie-breaking do the work.
    # Since the graph is fixed, the structure is fixed.
    # The random tie-breaking allows exploring different parts of the tree.
    
    bnb(full_mask, [], 0)

    # Final verification
    result = sorted(set(best_set))
    
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            d = (result[i] - result[j]) % m
            if d in forbidden:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        result = greedy_random()

    return np.array(result, dtype=np.int64)

if __name__ == "__main__":
    res = furstenberg_sarkozy_205()
    print(f"Size: {len(res)}")
    print(res)
