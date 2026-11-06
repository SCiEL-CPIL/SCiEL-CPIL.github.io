import time
import random
from math import gcd

def construct():
    deadline = time.time() + 85  # Safety margin
    
    def _lcm(a, b):
        return a // gcd(a, b) * b
    
    n = 128
    
    # Precompute all bad triples
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    bad_triples.append((a, b, c))
                    
    # Build conflict structures
    conflict_pairs = [set() for _ in range(n + 1)]
    for a, b, c in bad_triples:
        conflict_pairs[a].add((b, c))
        conflict_pairs[b].add((a, c))
        conflict_pairs[c].add((a, b))
        
    # Precompute conflict bitmasks for faster checking
    conflict_dict = [dict() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for u, v in conflict_pairs[x]:
            if u not in conflict_dict[x]:
                conflict_dict[x][u] = 0
            conflict_dict[x][u] |= (1 << (v - 1))
            if v not in conflict_dict[x]:
                conflict_dict[x][v] = 0
            conflict_dict[x][v] |= (1 << (u - 1))
    
    # Precompute for each x, the list of (u, v_mask) sorted by frequency
    conflict_list = [list(conflict_dict[x].items()) for x in range(n + 1)]
    
    def can_add(x, mask):
        # Check if x can be added to the set represented by mask
        for u, v_mask in conflict_list[x]:
            if mask & (1 << (u - 1)):
                if mask & v_mask:
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
    
    def local_search(mask, max_iters=100):
        improved = True
        iters = 0
        last_improve = time.time()
        
        while improved and time.time() < deadline - 2 and iters < max_iters:
            improved = False
            iters += 1
            start_time = time.time()
            
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
            found_swap = False
            s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
            random.shuffle(s_list)
            for x in s_list:
                if time.time() > deadline - 2: break
                mask_no_x = mask & ~(1 << (x - 1))
                # Try to add two elements
                added_count = 0
                new_mask = mask_no_x
                for y in range(1, n + 1):
                    if y == x: continue
                    if not (new_mask & (1 << (y - 1))):
                        if can_add(y, new_mask):
                            new_mask |= (1 << (y - 1))
                            added_count += 1
                            if added_count == 2:
                                break
                if added_count >= 2:
                    if popcount(new_mask) > popcount(mask):
                        mask = new_mask
                        improved = True
                        found_swap = True
                        break
            if found_swap:
                continue

            # 3. Try 2-for-3 swap: remove x, y, add z, w, v
            if time.time() < deadline - 3:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                random.shuffle(s_list)
                for i in range(min(len(s_list), 30)):
                    if time.time() > deadline - 3: break
                    x = s_list[i]
                    for j in range(i+1, min(len(s_list), i+8)):
                        y = s_list[j]
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        # Try to add three elements
                        added_count = 0
                        new_mask = mask_no_xy
                        for z in range(1, n + 1):
                            if z == x or z == y: continue
                            if not (new_mask & (1 << (z - 1))):
                                if can_add(z, new_mask):
                                    new_mask |= (1 << (z - 1))
                                    added_count += 1
                                    if added_count == 3:
                                        break
                        if added_count >= 3:
                            if popcount(new_mask) > popcount(mask):
                                mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    continue

            # 4. Random perturbation with simulated annealing-like acceptance
            if not improved:
                # Remove k random elements and try to rebuild
                k = random.choice([2, 3, 4])
                if len(s_list) > k + 1:
                    remove_idx = random.sample(range(len(s_list)), k)
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
    degrees = [len(conflict_pairs[k]) for k in range(n+1)]
    
    # Run multiple randomized greedy passes
    num_restarts = 500
    
    for i in range(num_restarts):
        if time.time() > deadline - 25: # Leave more time for B&B and final polish
            break
            
        # Different ordering strategies
        strategy = i % 6
        if strategy == 0:
            # Low degree first
            order = sorted(range(1, n + 1), key=lambda x: degrees[x] + random.random()*0.1)
        elif strategy == 1:
            # High degree first
            order = sorted(range(1, n + 1), key=lambda x: -degrees[x] + random.random()*0.1)
        elif strategy == 2:
            # Random
            order = list(range(1, n + 1))
            random.shuffle(order)
        elif strategy == 3:
            # Reverse order
            order = list(range(n, 0, -1))
            random.shuffle(order)
        elif strategy == 4:
            # Weighted random by inverse degree
            weights = [1.0 / (degrees[x] + 1) for x in range(1, n + 1)]
            order = list(range(1, n + 1))
            for i in range(n):
                total_w = sum(weights[i:])
                r = random.random() * total_w
                cum = 0
                for j in range(i, n):
                    cum += weights[j]
                    if r <= cum:
                        order[i], order[j] = order[j], order[i]
                        break
        else:
            # Degree-based with random tie-breaking
            order = sorted(range(1, n + 1), key=lambda x: (degrees[x] // 10, random.random()))
            
        mask = greedy(order)
        mask = local_search(mask)
        
        size = popcount(mask)
        if size > best_size:
            best_size = size
            best_mask = mask
            
    # Try branch-and-bound from the best solution found
    if time.time() < deadline - 15:
        # Start B&B from empty set with all candidates
        bb_mask, bb_size = branch_and_bound(0, (1 << n) - 1, best_size, max_depth=14)
        if bb_size > best_size:
            best_size = bb_size
            best_mask = bb_mask
            
    # Also try B&B starting from the current best solution
    if time.time() < deadline - 8:
        # Candidates are elements not in best_mask
        candidates = ((1 << n) - 1) & ~best_mask
        bb_mask2, bb_size2 = branch_and_bound(best_mask, candidates, best_size, max_depth=10)
        if bb_size2 > best_size:
            best_size = bb_size2
            best_mask = bb_mask2
    
    # Final polish with more iterations
    if time.time() < deadline - 2:
        best_mask = local_search(best_mask, max_iters=150)
    
    result = [i for i in range(1, n + 1) if (best_mask & (1 << (i - 1)))]
    
    # Ensure non-empty
    if not result:
        result = [1]
        
    return result
