import time
from fractions import Fraction

def construct():
    deadline = time.time() + 85  # Safety margin before 90s kill
    n = 256
    weights = [Fraction(1, x) for x in range(1, n + 1)]
    
    # Precompute AP pairs for each x
    # pairs_with_x[x] = list of (y, z) such that {x, y, z} is a 3-AP
    # We store x as 0-indexed internally for array access, but values are 1-indexed
    pairs_with_x = [[] for _ in range(n + 1)]
    
    # Generate all 3-APs
    # a, a+d, a+2d
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # For each element in the AP, record the other two
            pairs_with_x[a].append((b, c))
            pairs_with_x[b].append((a, c))
            pairs_with_x[c].append((a, b))
            
    # Convert to sets for faster membership? No, list of tuples is fine for iteration.
    # But we need to check if y in S and z in S.
    
    def can_add(x, S):
        """Check if x can be added to set S without forming a 3-AP."""
        for y, z in pairs_with_x[x]:
            if y in S and z in S:
                return False
        return True

    # Greedy initialization
    # Sort elements by weight (descending), i.e., x ascending
    initial_set = set()
    for x in range(1, n + 1):
        if can_add(x, initial_set):
            initial_set.add(x)
            
    best_set = initial_set.copy()
    best_weight = sum(weights[x - 1] for x in best_set)
    
    # Local Search
    # We will perform multiple rounds of improvement.
    # Round 1: Add any valid elements (should be none if greedy was maximal, but check just in case)
    # Round 2: 1-Swap
    # Round 3: 2-Swap
    # Round 4: 3-Swap (if time permits)
    
    improved = True
    max_iterations = 50
    iteration = 0
    
    while improved and time.time() < deadline - 5 and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Phase 1: Add
        for x in range(1, n + 1):
            if x not in best_set and can_add(x, best_set):
                best_set.add(x)
                best_weight += weights[x - 1]
                improved = True
                
        if time.time() > deadline - 5:
            break
            
        # Phase 2: 1-Swap
        # Try removing u and adding v
        # To be efficient, we only care if weight improves: 1/v > 1/u => u > v
        # So we want to replace large u with small v
        for u in sorted(best_set, reverse=True):  # Remove large ones first (low weight)
            if u not in best_set:
                continue
            temp_set = best_set - {u}
            # Try adding small v
            for v in range(1, u): # v < u implies 1/v > 1/u
                if v in best_set:
                    continue
                if can_add(v, temp_set):
                    # Check if it's actually an improvement (it is, since v < u)
                    # But we must ensure v is not already in temp_set (it isn't)
                    best_set = temp_set
                    best_set.add(v)
                    best_weight += weights[v - 1] - weights[u - 1]
                    improved = True
                    break
            if improved:
                # Restart 1-swap phase or continue?
                # Usually restarting local search after a swap is better
                break
                
        if improved:
            continue
            
        # Phase 3: 2-Swap
        # Remove u, v and add w (or w, x)
        # This is more complex.
        # Try removing two elements u, v from best_set
        # And adding one or two elements w, x not in best_set
        
        # To limit complexity, we can try specific strategies:
        # 1. Remove two largest elements, try to add two smallest available
        # 2. Iterate over pairs
        
        # Let's do a limited 2-swap:
        # Try removing each pair (u, v) where u, v are among the largest 20 elements
        # And try adding each w among the smallest 20 elements not in set
        
        largest_elements = sorted(best_set, reverse=True)[:20]
        smallest_candidates = [x for x in range(1, 21) if x not in best_set]
        
        if not smallest_candidates:
            # Try slightly larger candidates
            smallest_candidates = [x for x in range(1, 41) if x not in best_set]
            
        swap_found = False
        for i in range(len(largest_elements)):
            for j in range(i + 1, len(largest_elements)):
                u = largest_elements[i]
                v = largest_elements[j]
                temp_set = best_set - {u, v}
                removed_weight = weights[u - 1] + weights[v - 1]
                
                # Try adding one candidate
                for w in smallest_candidates:
                    if w in temp_set:
                        continue
                    if can_add(w, temp_set):
                        new_weight = best_weight - removed_weight + weights[w - 1]
                        if new_weight > best_weight:
                            best_set = temp_set
                            best_set.add(w)
                            best_weight = new_weight
                            improved = True
                            swap_found = True
                            break
                if swap_found:
                    break
                    
                # Try adding two candidates
                for k in range(len(smallest_candidates)):
                    w1 = smallest_candidates[k]
                    if w1 in temp_set:
                        continue
                    if not can_add(w1, temp_set):
                        continue
                    temp_set2 = temp_set | {w1}
                    for l in range(k + 1, len(smallest_candidates)):
                        w2 = smallest_candidates[l]
                        if w2 in temp_set2:
                            continue
                        if can_add(w2, temp_set2):
                            new_weight = best_weight - removed_weight + weights[w1 - 1] + weights[w2 - 1]
                            if new_weight > best_weight:
                                best_set = temp_set2
                                best_set.add(w2)
                                best_weight = new_weight
                                improved = True
                                swap_found = True
                                break
                    if swap_found:
                        break
            if swap_found:
                break
                
        if improved:
            continue
            
        # If no improvement in 1-swap or 2-swap, try a random restart or 3-swap
        # For now, if we get stuck, we might just break or try a simple perturbation
        # Given time constraints, breaking is safer than complex 3-swap which might not converge
        # But let's try one more thing: random removal of 3 elements and greedy refill
        if time.time() < deadline - 10:
             # Perturb: remove 3 random elements from the "middle" of the set
             # and greedily refill
             pass # Skipping complex perturbation for now to ensure stability
            
    return sorted(best_set)
