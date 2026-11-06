import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute conflicts for each x
    # conflicts[x] is a list of pairs (u, v) such that {u, v, x} is a bad triple.
    # We also precompute for each x and u, the bitmask of v that conflict with x given u is present.
    # conflict_mask[x][u] is a bitmask of v such that lcm(u,v)==lcm(u,x)==lcm(v,x)
    
    # First, find all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    bad_triples.append((a, b, c))
                    
    # Build conflict structures
    # For each x, we want to know which pairs (u, v) in the rest of the set would block x.
    # conflict_pairs[x] = set of (u, v) with u < v, u != x, v != x, such that {u, v, x} is bad.
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
        
    # Precompute conflict bitmasks for faster checking
    # conflict_mask[x] = list of (u, v_mask) where v_mask is bitmask of elements that conflict with x when u is present
    # Actually, let's precompute: for each x, a dict mapping u -> bitmask of v such that (u,v) in conflict_pairs[x]
    conflict_dict = [dict() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v in conflict_pairs[x]:
            if u not in conflict_dict[x]:
                conflict_dict[x][u] = 0
            conflict_dict[x][u] |= (1 << (v - 1))
            if v not in conflict_dict[x]:
                conflict_dict[x][v] = 0
            conflict_dict[x][v] |= (1 << (u - 1))
    
    # Precompute flat conflict lists for faster can_add
    # conflict_flat[x] = list of (u, v) pairs
    conflict_flat = [list() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v_mask in conflict_dict[x].items():
            # Convert v_mask to list of v's
            vm = v_mask
            while vm:
                lsb = vm & (-vm)
                v = lsb.bit_length()
                conflict_flat[x].append((u, v))
                vm &= vm - 1

    def can_add(x, mask):
        # Check if x can be added to the set represented by mask
        # x conflicts with pair (u,v) if both u and v are in mask
        for u, v in conflict_flat[x]:
            if (mask & (1 << (u - 1))) and (mask & (1 << (v - 1))):
                return False
        return True

    def greedy(order, mask=0):
        selected = mask
        for x in order:
            if not (selected & (1 << (x - 1))):
                if can_add(x, selected):
                    selected |= (1 << (x - 1))
        return selected

    def popcount(x):
        return bin(x).count('1')
    
    def local_search(mask, max_iters=50):
        improved = True
        iters = 0
        
        while improved and time.time() < deadline - 2 and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try to add any missing element
            for x in range(1, n + 1):
                if not (mask & (1 << (x - 1))):
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
                        break
            if improved:
                continue

            # 2. Try 1-for-2 swap: remove x, add y, z
            # Try ALL possible removals, track best
            best_new_mask = mask
            best_size = popcount(mask)
            s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]

            for x in s_list:
                if time.time() > deadline - 2:
                    break
                mask_no_x = mask & ~(1 << (x - 1))
                # Try to add two elements greedily
                new_mask = mask_no_x
                added_count = 0
                for y in range(1, n + 1):
                    if y == x:
                        continue
                    if not (new_mask & (1 << (y - 1))):
                        if can_add(y, new_mask):
                            new_mask |= (1 << (y - 1))
                            added_count += 1
                            if added_count == 2:
                                break
                if added_count >= 2:
                    sz = popcount(new_mask)
                    if sz > best_size:
                        best_size = sz
                        best_new_mask = new_mask

            if best_size > popcount(mask):
                mask = best_new_mask
                improved = True
                continue

            # 3. Try 2-for-3 swap: remove x, y, add z, w, v
            # This is more expensive but can escape deeper local minima
            if time.time() < deadline - 3:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                # Try a few random pairs to remove, track best
                random.shuffle(s_list)
                best_new_mask = mask
                best_size = popcount(mask)
                for i in range(min(len(s_list), 20)):
                    if time.time() > deadline - 3:
                        break
                    x = s_list[i]
                    for j in range(i + 1, min(len(s_list), i + 5)):
                        y = s_list[j]
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        # Try to add three elements
                        added_count = 0
                        new_mask = mask_no_xy
                        for z in range(1, n + 1):
                            if z == x or z == y:
                                continue
                            if not (new_mask & (1 << (z - 1))):
                                if can_add(z, new_mask):
                                    new_mask |= (1 << (z - 1))
                                    added_count += 1
                                    if added_count == 3:
                                        break
                        if added_count >= 3:
                            sz = popcount(new_mask)
                            if sz > best_size:
                                best_size = sz
                                best_new_mask = new_mask
                if best_size > popcount(mask):
                    mask = best_new_mask
                    improved = True
                    continue

            # 4. Random perturbation if stuck
            if not improved and iters > 5:
                # Remove 2 random elements and try to rebuild
                if len(s_list) > 4:
                    remove_idx = random.sample(range(len(s_list)), 2)
                    perturb_mask = mask
                    for idx in remove_idx:
                        perturb_mask &= ~(1 << (s_list[idx] - 1))
                    # Greedily add back
                    for x in range(1, n + 1):
                        if not (perturb_mask & (1 << (x - 1))):
                            if can_add(x, perturb_mask):
                                perturb_mask |= (1 << (x - 1))
                    if popcount(perturb_mask) > popcount(mask):
                        mask = perturb_mask
                        improved = True
                        continue

            if not improved:
                break
                
        return mask

    # Branch and bound for maximum independent set in 3-uniform hypergraph
    # State: (selected_mask, candidate_mask, depth)
    # We want to maximize popcount(selected_mask)
    
    def branch_and_bound(selected_mask, candidate_mask, best_size, depth=0, max_depth=15):
        """
        Find a good independent set by branching on candidates.
        Returns (best_mask, best_size)
        """
        if time.time() > deadline - 5:
            return selected_mask, best_size
            
        # Upper bound: current selected + all candidates
        upper_bound = popcount(selected_mask) + popcount(candidate_mask)
        if upper_bound <= best_size:
            return selected_mask, best_size
            
        if candidate_mask == 0:
            if popcount(selected_mask) > best_size:
                best_size = popcount(selected_mask)
                return selected_mask, best_size
            return selected_mask, best_size
            
        if depth >= max_depth:
            # Do a greedy completion
            greedy_mask = selected_mask
            remaining = candidate_mask
            while remaining:
                # Pick the element with fewest conflicts
                best_x = -1
                best_conflicts = float('inf')
                temp = remaining
                while temp:
                    lsb = temp & (-temp)
                    x = lsb.bit_length()
                    # Count how many pairs in remaining would conflict with x
                    cd = conflict_dict[x]
                    conf = 0
                    for u, v_mask in cd.items():
                        if remaining & (1 << (u - 1)):
                            conf += popcount(remaining & v_mask)
                    if conf < best_conflicts:
                        best_conflicts = conf
                        best_x = x
                    temp &= temp - 1
                if best_x == -1:
                    break
                if can_add(best_x, greedy_mask):
                    greedy_mask |= (1 << (best_x - 1))
                    remaining &= ~(1 << (best_x - 1))
                else:
                    remaining &= ~(1 << (best_x - 1))
            
            if popcount(greedy_mask) > best_size:
                best_size = popcount(greedy_mask)
                return greedy_mask, best_size
            return selected_mask, best_size
        
        # Choose the next candidate to branch on
        # Heuristic: pick element with most conflicts (most constrained)
        best_x = -1
        best_score = -1
        temp = candidate_mask
        while temp:
            lsb = temp & (-temp)
            x = lsb.bit_length()
            cd = conflict_dict[x]
            score = 0
            for u, v_mask in cd.items():
                if candidate_mask & (1 << (u - 1)):
                    score += popcount(candidate_mask & v_mask)
            if score > best_score:
                best_score = score
                best_x = x
            temp &= temp - 1
        
        if best_x == -1:
            return selected_mask, best_size
        
        # Branch 1: Include best_x
        if can_add(best_x, selected_mask):
            new_selected = selected_mask | (1 << (best_x - 1))
            # Update candidates: remove elements that conflict with best_x
            new_candidates = candidate_mask & ~(1 << (best_x - 1))
            cd = conflict_dict[best_x]
            for u, v_mask in cd.items():
                if new_candidates & (1 << (u - 1)):
                    # If u is in candidates, remove v's that are also in candidates
                    new_candidates &= ~v_mask
            mask1, size1 = branch_and_bound(new_selected, new_candidates, best_size, depth + 1, max_depth)
            if size1 > best_size:
                best_size = size1
                selected_mask = mask1
        
        # Branch 2: Exclude best_x
        new_candidates = candidate_mask & ~(1 << (best_x - 1))
        mask2, size2 = branch_and_bound(selected_mask, new_candidates, best_size, depth + 1, max_depth)
        if size2 > best_size:
            best_size = size2
            selected_mask = mask2
            
        return selected_mask, best_size

    best_mask = 0
    best_size = 0
    
    # Precompute degrees for sorting
    degrees = [len(conflict_pairs[k]) for k in range(n + 1)]

    # Precompute a "density" score: number of conflict pairs per element
    # Elements with high density are more constrained
    density = [0.0] * (n + 1)
    for x in range(1, n + 1):
        density[x] = len(conflict_pairs[x]) / max(1, n - 1)

    # Run multiple randomized greedy passes
    num_restarts = 500  # Reduced to leave time for branch-and-bound

    for i in range(num_restarts):
        if time.time() > deadline - 20:  # Leave time for branch-and-bound and final polish
            break

        # Different ordering strategies
        strategy = i % 6
        if strategy == 0:
            # Low degree first
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] + random.random() * 0.1)
        elif strategy == 1:
            # High degree first
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] + random.random() * 0.1)
        elif strategy == 2:
            # Random
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif strategy == 3:
            # Reverse order
            order = list(range(n, 0, -1))
            random.shuffle(order)
        elif strategy == 4:
            # Low density first
            order = sorted(range(1, n + 1), key=lambda x: density[x] + random.random() * 0.01)
        else:
            # High density first
            order = sorted(range(1, n + 1), key=lambda x: -density[x] + random.random() * 0.01)

        mask = greedy(order)
        mask = local_search(mask)

        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Try branch-and-bound from the best solution found
    if time.time() < deadline - 10:
        # Start B&B from empty set with all candidates
        bb_mask, bb_size = branch_and_bound(0, (1 << n) - 1, best_size, max_depth=12)
        if bb_size > best_size:
            best_size = bb_size
            best_mask = bb_mask

    # Also try B&B starting from the current best solution
    if time.time() < deadline - 5:
        # Candidates are elements not in best_mask
        candidates = ((1 << n) - 1) & ~best_mask
        bb_mask2, bb_size2 = branch_and_bound(best_mask, candidates, best_size, max_depth=8)
        if bb_size2 > best_size:
            best_size = bb_size2
            best_mask = bb_mask2

    # Additional B&B with different max_depth
    if time.time() < deadline - 3:
        candidates = ((1 << n) - 1) & ~best_mask
        bb_mask3, bb_size3 = branch_and_bound(best_mask, candidates, best_size, max_depth=10)
        if bb_size3 > best_size:
            best_size = bb_size3
            best_mask = bb_mask3
    
    # Final polish with more iterations
    if time.time() < deadline - 2:
        best_mask = local_search(best_mask, max_iters=100)
    
    result = [i for i in range(1, n + 1) if (best_mask & (1 << (i - 1)))]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result

# Helper for popcount if not using bin (bin is fast enough for 128 bits)
