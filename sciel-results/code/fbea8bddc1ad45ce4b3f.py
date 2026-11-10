"""Optimized program for large reciprocal weight under lcm separation."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm as math_lcm

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # 90s wall-clock, 5s safety margin
    MAX_VAL = 256
    RANGE = list(range(2, MAX_VAL + 1))
    N = len(RANGE)
    
    # Precompute incompatibility adjacency as bitsets
    # adj[i] is a bitmask where bit j is set if RANGE[i] and RANGE[j] are incompatible
    adj = [0] * N
    
    for i in range(N):
        a = RANGE[i]
        for j in range(i + 1, N):
            b = RANGE[j]
            if math_lcm(a, b) <= MAX_VAL:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Weights: reciprocal
    w = [1.0 / RANGE[i] for i in range(N)]
    
    def weight_of(mask):
        """Calculate total weight of a bitmask."""
        total = 0.0
        m = mask
        while m:
            lsb = m & (-m)
            i = lsb.bit_length() - 1
            total += w[i]
            m ^= lsb
        return total
    
    def is_compatible_with_mask(i, mask):
        """Check if vertex i is compatible with all vertices in mask."""
        return (adj[i] & mask) == 0
    
    def greedy(order):
        """Greedy construction based on a specific order of indices."""
        mask = 0
        for i in order:
            if is_compatible_with_mask(i, mask):
                mask |= (1 << i)
        return mask
    
    # Step 1: Generate diverse initial solutions using different heuristics
    starts = []
    seen_masks = set()
    
    # Order 1: Ascending value (smallest numbers first -> highest reciprocal)
    order1 = list(range(N))
    m = greedy(order1)
    if m and m not in seen_masks:
        starts.append(m)
        seen_masks.add(m)
    
    # Order 2: Descending value
    order2 = list(range(N, -1, -1))
    m = greedy(order2)
    if m and m not in seen_masks:
        starts.append(m)
        seen_masks.add(m)
    
    # Order 3: Sorted by number of incompatible neighbors (least constrained first)
    # This often yields larger independent sets
    order3 = sorted(range(N), key=lambda i: bin(adj[i]).count('1'))
    m = greedy(order3)
    if m and m not in seen_masks:
        starts.append(m)
        seen_masks.add(m)
    
    # Order 4: Sorted by reciprocal (largest first), tie-break by degree
    order4 = sorted(range(N), key=lambda i: (-w[i], bin(adj[i]).count('1')))
    m = greedy(order4)
    if m and m not in seen_masks:
        starts.append(m)
        seen_masks.add(m)
    
    # Random shuffles
    rnd = random.Random(42)
    for _ in range(10):
        order_r = list(range(N))
        rnd.shuffle(order_r)
        m = greedy(order_r)
        if m and m not in seen_masks:
            starts.append(m)
            seen_masks.add(m)
            if len(starts) > 20:
                break
    
    if not starts:
        return [256]
    
    best_mask = max(starts, key=weight_of)
    best_w = weight_of(best_mask)
    
    # Step 2: Local Search
    # Moves: Add, Remove, Swap (1-in, 1-out), 2-flip (2-in, 1-out or 1-in, 2-out)
    
    def get_candidates(mask):
        """Get list of indices not in mask that are compatible with mask."""
        cands = []
        for i in range(N):
            if not (mask & (1 << i)):
                if is_compatible_with_mask(i, mask):
                    cands.append(i)
        return cands
    
    def local_search(mask, w_val, t_limit):
        cur = mask
        cur_w = w_val
        improved = True
        iterations = 0
        
        while improved and iterations < 100 and time.time() < t_limit:
            improved = False
            iterations += 1
            
            # 1. Try ADD moves
            cands = get_candidates(cur)
            # Sort candidates by weight descending to find best add quickly
            cands.sort(key=lambda i: -w[i])
            for i in cands:
                new_mask = cur | (1 << i)
                new_w = cur_w + w[i]
                if new_w > cur_w + 1e-12:
                    cur = new_mask
                    cur_w = new_w
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try SWAP moves (remove one, add one)
            # Iterate over elements in cur
            elems = [i for i in range(N) if (cur >> i) & 1]
            # Sort elements in cur by weight ascending (remove low weight first)
            elems.sort(key=lambda i: w[i])
            
            for r in elems:
                base_mask = cur ^ (1 << r)
                base_w = cur_w - w[r]
                
                # Find best compatible addition to base_mask
                cands = get_candidates(base_mask)
                cands.sort(key=lambda i: -w[i])
                
                for a in cands:
                    new_mask = base_mask | (1 << a)
                    new_w = base_w + w[a]
                    if new_w > cur_w + 1e-12:
                        cur = new_mask
                        cur_w = new_w
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # 3. Try 2-FLIP moves (remove one, add two)
            # This can escape local optima where no single swap helps but two additions help
            # We limit the search space to keep it fast
            
            for r in elems:
                base_mask = cur ^ (1 << r)
                base_w = cur_w - w[r]
                
                cands = get_candidates(base_mask)
                if len(cands) < 2:
                    continue
                
                # Sort candidates by weight
                cands.sort(key=lambda i: -w[i])
                
                # Try pairs of top candidates
                # Limit pairs to top 20 to keep complexity manageable
                top_cands = cands[:20]
                best_pair_gain = -1.0
                best_pair = None
                
                for i1 in range(len(top_cands)):
                    a1 = top_cands[i1]
                    mask1 = base_mask | (1 << a1)
                    w1 = base_w + w[a1]
                    
                    # Check if a2 is compatible with mask1
                    for i2 in range(i1 + 1, len(top_cands)):
                        a2 = top_cands[i2]
                        if is_compatible_with_mask(a2, mask1):
                            new_w = w1 + w[a2]
                            gain = new_w - cur_w
                            if gain > best_pair_gain:
                                best_pair_gain = gain
                                best_pair = (a1, a2, new_w)
                
                if best_pair and best_pair_gain > 1e-12:
                    a1, a2, new_w = best_pair
                    new_mask = base_mask | (1 << a1) | (1 << a2)
                    cur = new_mask
                    cur_w = new_w
                    improved = True
                    break
            
            if improved:
                continue
                
            # 4. Try 2-FLIP moves (remove two, add one) - usually less useful for maximizing sum
            # but might help if two small numbers are blocking one huge reciprocal? 
            # Actually, removing two and adding one reduces cardinality. 
            # Since we want max sum, adding is generally better unless the added number is tiny.
            # Let's skip this for now to save time, or implement a simple version.
            
            # 5. Try 2-FLIP moves (remove two, add two)
            # This is expensive O(n^4). Let's do a limited version.
            # Only try if we have time.
            if time.time() < t_limit - 2:
                # Try removing the two smallest weight elements in cur
                if len(elems) >= 2:
                    r1, r2 = elems[0], elems[1]
                    base_mask = cur ^ (1 << r1) ^ (1 << r2)
                    base_w = cur_w - w[r1] - w[r2]
                    
                    cands = get_candidates(base_mask)
                    if len(cands) >= 2:
                        cands.sort(key=lambda i: -w[i])
                        top_cands = cands[:10]
                        best_pair_gain = -1.0
                        best_pair = None
                        
                        for i1 in range(len(top_cands)):
                            a1 = top_cands[i1]
                            mask1 = base_mask | (1 << a1)
                            w1 = base_w + w[a1]
                            
                            for i2 in range(i1 + 1, len(top_cands)):
                                a2 = top_cands[i2]
                                if is_compatible_with_mask(a2, mask1):
                                    new_w = w1 + w[a2]
                                    gain = new_w - cur_w
                                    if gain > best_pair_gain:
                                        best_pair_gain = gain
                                        best_pair = (a1, a2, new_w)
                        
                        if best_pair and best_pair_gain > 1e-12:
                            a1, a2, new_w = best_pair
                            new_mask = base_mask | (1 << a1) | (1 << a2)
                            cur = new_mask
                            cur_w = new_w
                            improved = True
        
        return cur, cur_w
    
    # Run local search from the best start
    t_limit = DEADLINE - 5
    best_mask, best_w = local_search(best_mask, best_w, t_limit)
    
    # If time permits, run local search from other promising starts
    remaining_time = t_limit - time.time()
    if remaining_time > 5:
        # Sort other starts by weight
        other_starts = [m for m in starts if m != best_mask]
        other_starts.sort(key=weight_of, reverse=True)
        
        for m in other_starts[:5]:
            if time.time() > t_limit - 2:
                break
            lm, lw = local_search(m, weight_of(m), t_limit)
            if lw > best_w:
                best_w = lw
                best_mask = lm
    
    # Final polish: try a few more swaps if time allows
    if time.time() < DEADLINE - 1:
        best_mask, best_w = local_search(best_mask, best_w, DEADLINE - 0.5)
        
    # Convert mask to list of values
    result = [RANGE[i] for i in range(N) if (best_mask >> i) & 1]
    
    if not result:
        result = [256]
        
    # Verify validity (safety check)
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math_lcm(result[i], result[j]) <= MAX_VAL:
                return [256]
                
    return result
# EVOLVE-BLOCK-END
