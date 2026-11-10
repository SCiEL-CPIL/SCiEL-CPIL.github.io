import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 12
    m = 40
    num_masks = 1 << n
    full_mask = num_masks - 1
    
    # Precompute popcount using bit_count
    # We need comp_bits[i] to be an integer where bit j is 1 if i and j are comparable.
    # Comparability: (i & j) == i or (i & j) == j.
    # Note: i != j is assumed for the count of pairs, but for the bitset we can include self.
    # If we include self, popcount(comp_bits[i] & S_mask) will include i if i is in S.
    # So deg[i] = popcount(comp_bits[S[i]] & S_mask) - 1.
    
    # Precomputation
    # This might take a bit of time but 4096^2 is 16M operations.
    # 16M bitwise ops in Python might be slow (~1-2s).
    # Let's optimize.
    
    # Alternative: Don't precompute full matrix if it's too slow.
    # But 16M simple ops is usually under 1s in PyPy or optimized CPython?
    # In standard CPython, 16M loops might take 5-10s.
    # We have 55s. It's worth it.
    
    # Actually, we can compute comparability on the fly if we are careful.
    # But precomputing `comp_bits` makes the inner loop of local search very fast.
    
    # Let's check if we can speed up precomputation.
    # For each i, we want to find all j such that i subset j or j subset i.
    # j subset i: j is a submask of i.
    # i subset j: j is a supermask of i.
    
    # We can generate submasks and supermasks.
    # Number of submasks of i is 2^popcount(i).
    # Number of supermasks of i is 2^(n - popcount(i)).
    # Sum over all i of (2^popcount(i) + 2^(n-popcount(i))) is roughly 3^n = 531441.
    # This is much faster than N^2!
    
    comp_bits = [0] * num_masks
    
    # Generate submasks
    for i in range(num_masks):
        # i is a supermask of all its submasks
        # So for all submasks j of i, j is comparable to i.
        # We set bit j in comp_bits[i] and bit i in comp_bits[j].
        
        # Iterate submasks of i
        sub = i
        while True:
            # sub is a submask of i
            # Set bit sub in comp_bits[i]
            comp_bits[i] |= (1 << sub)
            # Set bit i in comp_bits[sub]
            comp_bits[sub] |= (1 << i)
            
            if sub == 0:
                break
            sub = (sub - 1) & i
            
    # Wait, this double counts and is a bit messy with the bit shifts if num_masks > 64.
    # Python integers have arbitrary precision, so 1 << 4095 is fine.
    # However, the loop `while True` iterates 2^popcount(i) times.
    # Total iterations is sum_{i=0}^{2^n-1} 2^{popcount(i)} = 3^n.
    # 3^12 = 531,441. This is very fast.
    
    # Let's verify the logic.
    # If j is a submask of i, then j subset i. They are comparable.
    # We set comp_bits[i] bit j and comp_bits[j] bit i.
    # This covers all comparable pairs.
    
    # Initialization
    # Start with all subsets of {0, 1, 2, 3, 4} (size 5)
    # There are 2^5 = 32 such sets.
    initial_masks = []
    for i in range(1 << 5):
        initial_masks.append(i)
        
    current_masks = initial_masks[:]
    used_masks = set(current_masks)
    
    # Greedily add remaining m - 32 sets
    # To speed up, we can just pick the best from the rest.
    # We need to calculate comparability with existing sets.
    
    # Build S_mask
    S_mask = 0
    for mask in current_masks:
        S_mask |= (1 << mask)
        
    # Calculate initial degrees
    def get_degrees(masks, S_mask_val):
        degs = []
        for mask in masks:
            # Count comparable masks in S_mask_val
            # This includes self, so subtract 1
            deg = comp_bits[mask].bit_count() & 0 # bit_count() returns int
            # Wait, comp_bits[mask] is a huge int.
            # We need to AND with S_mask_val first.
            deg = (comp_bits[mask] & S_mask_val).bit_count() - 1
            degs.append(deg)
        return degs
        
    degs = get_degrees(current_masks, S_mask)
    current_score = sum(degs) // 2
    
    # Greedy addition
    all_masks_list = list(range(num_masks))
    random.seed(42)
    random.shuffle(all_masks_list)
    
    for _ in range(m - len(current_masks)):
        best_mask = -1
        best_score = -1
        for v in all_masks_list:
            if v in used_masks:
                continue
            # Score is number of comparable sets in current S
            # This is (comp_bits[v] & S_mask).bit_count()
            score = (comp_bits[v] & S_mask).bit_count()
            if score > best_score:
                best_score = score
                best_mask = v
        if best_mask != -1:
            current_masks.append(best_mask)
            used_masks.add(best_mask)
            S_mask |= (1 << best_mask)
            
    # Recalculate degrees and score
    degs = get_degrees(current_masks, S_mask)
    current_score = sum(degs) // 2
    best_score = current_score
    best_solution = list(current_masks)
    
    # Local Search
    # We maintain:
    # current_masks: list of m masks
    # degs: list of m degrees
    # S_mask: int representing the set of masks
    # used_masks: set of masks for O(1) lookup
    
    # To find best swap (u_idx, v):
    # Delta = (comparables of v in S\{u}) - (comparables of u in S\{u})
    # comparables of u in S\{u} is degs[u_idx].
    # comparables of v in S\{u} is (comp_bits[v] & (S_mask ^ (1 << current_masks[u_idx]))).bit_count()
    
    # We can precompute `S_mask_without_u` for each u? No, just do the XOR.
    
    # Optimization:
    # Iterate u_idx from 0 to m-1.
    # Iterate v from all_masks_list (or just range(num_masks) if we don't care about order).
    # We should iterate v not in used_masks.
    # Checking `if v in used_masks` is O(1).
    
    # Total ops per iteration: m * (num_masks - m)
    # 40 * 4056 = 162,240 checks.
    # Each check: 1 XOR, 1 AND, 1 bit_count.
    # bit_count is C-optimized in Python 3.10+.
    
    iterations = 0
    no_improvement_count = 0
    
    while time.time() < deadline:
        iterations += 1
        
        max_delta = 0
        best_u_idx = -1
        best_v = -1
        
        # To speed up, we can skip v if its potential is low?
        # For now, full scan.
        
        for u_idx in range(m):
            u_mask = current_masks[u_idx]
            u_deg = degs[u_idx]
            S_mask_without_u = S_mask ^ (1 << u_mask)
            
            # Iterate over all v not in used_masks
            # We can iterate over all_masks_list and check if not in used_masks
            # Or maintain a list of unused masks? Updating that list is O(m) per swap.
            # Checking set membership is O(1) average.
            
            for v in range(num_masks):
                if v in used_masks:
                    continue
                    
                v_deg_in_S_minus_u = (comp_bits[v] & S_mask_without_u).bit_count()
                delta = v_deg_in_S_minus_u - u_deg
                
                if delta > max_delta:
                    max_delta = delta
                    best_u_idx = u_idx
                    best_v = v
        
        if max_delta > 0:
            # Perform swap
            old_u = current_masks[best_u_idx]
            new_v = best_v
            
            # Update S_mask
            S_mask ^= (1 << old_u)
            S_mask |= (1 << new_v)
            
            # Update used_masks
            used_masks.remove(old_u)
            used_masks.add(new_v)
            
            # Update current_masks
            current_masks[best_u_idx] = new_v
            
            # Update degs
            # For all j != best_u_idx:
            # If old_u was comparable to S[j], degs[j] -= 1
            # If new_v is comparable to S[j], degs[j] += 1
            # We can recompute degs for all j? O(m * bit_count)
            # Or update incrementally.
            # Incremental:
            # For each j in range(m):
            #   if j == best_u_idx: continue
            #   j_mask = current_masks[j]
            #   if comp_bits[old_u] has bit j_mask: degs[j] -= 1
            #   if comp_bits[new_v] has bit j_mask: degs[j] += 1
            
            # Checking bit: (comp_bits[old_u] >> j_mask) & 1
            # This is O(1) but bit shift on large ints might be slow?
            # Alternative: (comp_bits[old_u] & (1 << j_mask)) != 0
            
            for j in range(m):
                if j == best_u_idx:
                    continue
                j_mask = current_masks[j]
                if (comp_bits[old_u] >> j_mask) & 1:
                    degs[j] -= 1
                if (comp_bits[new_v] >> j_mask) & 1:
                    degs[j] += 1
            
            # Update degs[best_u_idx]
            # It is the number of comparables of new_v in S (which now includes new_v)
            # So (comp_bits[new_v] & S_mask).bit_count() - 1
            degs[best_u_idx] = (comp_bits[new_v] & S_mask).bit_count() - 1
            
            current_score += max_delta
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_masks)
            no_improvement_count = 0
        else:
            # No improvement found
            no_improvement_count += 1
            if no_improvement_count > 100:
                # Simulated Annealing step or Random Restart
                # Let's do a random perturbation
                # Pick random u_idx and random v not in used
                u_idx = random.randint(0, m - 1)
                unused = [v for v in range(num_masks) if v not in used_masks]
                if not unused:
                    break
                v = random.choice(unused)
                
                # Accept with probability
                # Calculate delta
                old_u = current_masks[u_idx]
                S_mask_without_u = S_mask ^ (1 << old_u)
                v_deg_in_S_minus_u = (comp_bits[v] & S_mask_without_u).bit_count()
                delta = v_deg_in_S_minus_u - degs[u_idx]
                
                # Simple SA: accept if delta >= 0 or random < exp(delta/T)
                # T should decrease over time?
                # For simplicity, use fixed T or random restart.
                # Given time constraints, random restart might be safer.
                # But let's try SA with T=5.
                if delta >= 0 or random.random() < math.exp(delta / 5.0):
                    # Perform swap (same logic as above)
                    S_mask ^= (1 << old_u)
                    S_mask |= (1 << v)
                    used_masks.remove(old_u)
                    used_masks.add(v)
                    current_masks[u_idx] = v
                    
                    for j in range(m):
                        if j == u_idx:
                            continue
                        j_mask = current_masks[j]
                        if (comp_bits[old_u] >> j_mask) & 1:
                            degs[j] -= 1
                        if (comp_bits[v] >> j_mask) & 1:
                            degs[j] += 1
                            
                    degs[u_idx] = (comp_bits[v] & S_mask).bit_count() - 1
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_solution = list(current_masks)
                
                # Reset no_improvement_count to allow more SA steps
                no_improvement_count = 0
                if random.random() < 0.1:
                    # Occasionally restart greedy? No, too expensive.
                    pass

    # Convert best_solution to list of lists
    result = []
    for mask in best_solution:
        s = []
        for i in range(n):
            if (mask >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result
