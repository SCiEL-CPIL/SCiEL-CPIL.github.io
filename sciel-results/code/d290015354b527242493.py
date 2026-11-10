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

    # Precompute complement masks? Not directly, but we use bit operations.
    
    # 3. Branch and Bound Logic
    best_set = []
    best_size = 0
    
    def update_best(current_set):
        nonlocal best_set, best_size
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)

    # Upper bound using greedy matching
    def greedy_maximal_matching(candidates_mask):
        """Find a maximal matching in the conflict graph induced by candidates_mask."""
        remaining = candidates_mask
        matching_size = 0
        while remaining:
            # Pick a vertex with minimum degree in remaining subgraph
            temp = remaining
            min_deg = m + 1
            pivot = -1
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                deg = (neighbors[idx] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    pivot = idx
                    if deg == 0:
                        break
                temp ^= lsb
            
            if pivot == -1:
                break
            
            if min_deg == 0:
                # Isolated vertex, remove it
                remaining &= ~(1 << pivot)
                continue

            # Find a neighbor of pivot in remaining
            neigh_mask = neighbors[pivot] & remaining
            lsb = neigh_mask & (-neigh_mask)
            neighbor_idx = lsb.bit_length() - 1
            
            matching_size += 1
            remaining &= ~(1 << pivot)
            remaining &= ~(1 << neighbor_idx)
        
        return matching_size

    def upper_bound(candidates_mask):
        """Upper bound on the size of an independent set in the subgraph induced by candidates_mask."""
        num_v = candidates_mask.bit_count()
        if num_v == 0:
            return 0
        
        # Bound based on maximal matching: |IS| <= |V| - |M|
        match_size = greedy_maximal_matching(candidates_mask)
        return num_v - match_size

    # Improved pivot selection
    def select_pivot(candidates_mask, strategy="mixed"):
        """Select a pivot vertex from candidates_mask using the given strategy."""
        cands = [i for i in range(m) if (candidates_mask >> i) & 1]
        if not cands:
            return -1
        
        if len(cands) == 1:
            return cands[0]
        
        if strategy == "degree":
            # Pick vertex with maximum degree in the induced subgraph
            max_deg = -1
            pivot = -1
            for i in cands:
                deg = (neighbors[i] & candidates_mask).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pivot = i
            return pivot
        
        elif strategy == "impact":
            # Pick vertex that removes the most candidates when included
            max_impact = -1
            pivot = -1
            for i in cands:
                # If we include i, we remove i and all its neighbors from candidates
                removed = (1 << i) | (neighbors[i] & candidates_mask)
                impact = removed.bit_count()
                if impact > max_impact:
                    max_impact = impact
                    pivot = i
            return pivot
        
        elif strategy == "mixed":
            # Combine degree and impact
            max_score = -1
            pivot = -1
            for i in cands:
                deg = (neighbors[i] & candidates_mask).bit_count()
                removed = (1 << i) | (neighbors[i] & candidates_mask)
                impact = removed.bit_count()
                score = deg * 2 + impact  # Weight degree more
                if score > max_score:
                    max_score = score
                    pivot = i
            return pivot
        
        return cands[0]

    def bnb(candidates_mask, current_set, depth, strategy="mixed"):
        nonlocal best_size, best_set
        
        # Pruning: if current size + upper bound <= best_size, return
        ub = upper_bound(candidates_mask)
        if len(current_set) + ub <= best_size:
            return
        
        if candidates_mask == 0:
            update_best(current_set)
            return

        # Check time
        if time.time() > deadline:
            return

        # Get list of candidates
        cands = [i for i in range(m) if (candidates_mask >> i) & 1]
        if not cands:
            update_best(current_set)
            return

        # Select pivot
        pivot = select_pivot(candidates_mask, strategy)
        if pivot == -1:
            update_best(current_set)
            return

        # Branch 1: Include pivot
        # New candidates = candidates - {pivot} - neighbors(pivot)
        new_cands_1 = candidates_mask & ~(1 << pivot) & ~neighbors[pivot]
        current_set.append(pivot)
        
        # Check if worth recursing
        ub1 = upper_bound(new_cands_1)
        if len(current_set) + ub1 > best_size:
            bnb(new_cands_1, current_set, depth + 1, strategy)
        
        current_set.pop()

        # Branch 2: Exclude pivot
        new_cands_2 = candidates_mask & ~(1 << pivot)
        
        # Check if worth recursing
        ub2 = upper_bound(new_cands_2)
        if len(current_set) + ub2 > best_size:
            bnb(new_cands_2, current_set, depth + 1, strategy)

    # Phase 1: Greedy to get a good initial lower bound
    def greedy_with_strategy(strategy="random"):
        remaining_mask = (1 << m) - 1
        selected = []
        while remaining_mask:
            cands = [i for i in range(m) if (remaining_mask >> i) & 1]
            if not cands:
                break
            
            if strategy == "random":
                pivot = random.choice(cands)
            elif strategy == "min_degree":
                # Pick vertex with minimum degree in remaining subgraph
                min_deg = m + 1
                pivot = -1
                for i in cands:
                    deg = (neighbors[i] & remaining_mask).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        pivot = i
            elif strategy == "max_degree":
                # Pick vertex with maximum degree
                max_deg = -1
                pivot = -1
                for i in cands:
                    deg = (neighbors[i] & remaining_mask).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        pivot = i
            else:
                pivot = random.choice(cands)
            
            selected.append(pivot)
            remaining_mask &= ~(1 << pivot)
            remaining_mask &= ~neighbors[pivot]
        return selected

    # Run several greedy iterations with different strategies
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        gs = greedy_with_strategy("random")
        if len(gs) > best_size:
            best_size = len(gs)
            best_set = gs
    
    for _ in range(30):
        if time.time() > deadline - 20:
            break
        gs = greedy_with_strategy("min_degree")
        if len(gs) > best_size:
            best_size = len(gs)
            best_set = gs
    
    for _ in range(30):
        if time.time() > deadline - 20:
            break
        gs = greedy_with_strategy("max_degree")
        if len(gs) > best_size:
            best_size = len(gs)
            best_set = gs

    # Phase 2: Branch and Bound with multiple strategies
    random.seed(42)
    
    full_mask = (1 << m) - 1
    
    # Try different pivot strategies
    strategies = ["mixed", "impact", "degree"]
    
    for strategy in strategies:
        if time.time() > deadline - 10:
            break
        
        # Run B&B with this strategy
        bnb(full_mask, [], 0, strategy)
        
        # If we found a better solution, continue
        # Otherwise, try next strategy

    # Phase 3: Local search / hill climbing on the best solution
    # Try to add elements or swap elements to improve
    
    def try_improve(current_set):
        """Try to improve the current set by adding or swapping elements."""
        current_set = list(current_set)
        current_mask = 0
        for v in current_set:
            current_mask |= (1 << v)
        
        # Try adding each non-conflicting element
        for i in range(m):
            if (current_mask >> i) & 1:
                continue
            # Check if i conflicts with any element in current_set
            if neighbors[i] & current_mask:
                continue
            # i can be added
            new_set = current_set + [i]
            if len(new_set) > len(current_set):
                return new_set
        
        # Try swapping: remove one element, add another
        for remove_idx in range(len(current_set)):
            removed = current_set[remove_idx]
            temp_set = current_set[:remove_idx] + current_set[remove_idx+1:]
            temp_mask = 0
            for v in temp_set:
                temp_mask |= (1 << v)
            
            for i in range(m):
                if (temp_mask >> i) & 1:
                    continue
                # Check if i conflicts with any element in temp_set
                if neighbors[i] & temp_mask:
                    continue
                # Check if i was the removed one (skip)
                if i == removed:
                    continue
                # i can be added
                new_set = temp_set + [i]
                if len(new_set) > len(current_set):
                    return new_set
        
        return current_set

    # Run local search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            new_set = try_improve(best_set)
            if len(new_set) > len(best_set):
                best_set = new_set
                best_size = len(best_set)
                improved = True

    # Final verification
    result = sorted(set(best_set))
    
    # Double check validity
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
        # Fallback to a simple greedy if something went wrong
        result = greedy_with_strategy("min_degree")

    return np.array(result, dtype=np.int64)

if __name__ == "__main__":
    res = furstenberg_sarkozy_205()
    print(f"Size: {len(res)}")
    print(res)
