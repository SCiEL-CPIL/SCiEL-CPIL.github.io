import time
from math import lcm
import numpy as np
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    # 1. Define candidates
    candidates = list(range(2, 257))
    n = len(candidates)
    
    # 2. Precompute compatibility bitsets
    # compat[i] is bitmask of j such that lcm(candidates[i], candidates[j]) > 256
    # Note: i and j are distinct. We don't need self-compatibility.
    compat = [0] * n
    for i in range(n):
        a = candidates[i]
        for j in range(i + 1, n):
            b = candidates[j]
            if lcm(a, b) > 256:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
                
    # Weights
    weights = [1.0 / x for x in candidates]
    
    # 3. Initial Lower Bound via Greedy Heuristic
    # Sort by weight descending (smallest x first)
    order = sorted(range(n), key=lambda i: -weights[i])
    
    def greedy_heuristic(order):
        selected = 0
        w = 0.0
        for i in order:
            if not (compat[i] & selected) == 0: # Wait, if conflict, skip.
                # We want NO conflict. So if (compat[i] & selected) is NOT 0, it's compatible?
                # No, compat[i] has 1 where it IS compatible.
                # So if (compat[i] & selected) != 0, it is compatible with *some* selected?
                # No, we need it to be compatible with *ALL* selected.
                # So we need (selected & ~compat[i]) == 0.
                # i.e., selected is a subset of compat[i].
                if (selected & ~compat[i]) == 0:
                    selected |= (1 << i)
                    w += weights[i]
        return w, selected

    best_weight, best_set = greedy_heuristic(order)
    
    # Try a few different orderings for better LB?
    # Maybe randomize? Or just stick with weight-based.
    # Let's try a second greedy: random shuffle
    import random
    random.seed(42)
    for _ in range(10):
        rand_order = order[:]
        random.shuffle(rand_order)
        w, s = greedy_heuristic(rand_order)
        if w > best_weight:
            best_weight = w
            best_set = s
            
    # 4. Branch and Bound
    # We need to pass current_clique and candidates_bitmask
    # To speed up, we pre-sort nodes globally by weight? 
    # The B&B usually iterates candidates in a fixed order.
    
    # Let's implement the B&B.
    # To handle recursion depth and speed, we use a list for the stack or recursion.
    # N=255, recursion depth max 255. Python default limit is 1000, so it's fine.
    
    global_best_weight = best_weight
    global_best_set = best_set
    
    def get_colors_and_bound(candidates_mask):
        """
        Greedy coloring of the subgraph induced by candidates_mask.
        Returns upper bound on clique weight.
        """
        # Get list of nodes in candidates_mask
        # This is O(N)
        nodes = []
        m = candidates_mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nodes.append(idx)
            m ^= lsb
            
        if not nodes:
            return 0.0
            
        # Sort nodes by weight descending
        nodes.sort(key=lambda x: -weights[x])
        
        # colors: dict node -> color_idx
        # color_max_weights: list of max weight for each color
        color_max_weights = []
        # For fast lookup of "is color c available for node u", 
        # we can maintain a list of nodes for each color.
        # Or simply iterate.
        
        # Optimization: Since N is small (<=255), simple loop is fine.
        # But we can be smarter.
        
        for u in nodes:
            # Find smallest color c such that u is compatible with all nodes of color c
            # i.e., for all v in color_c, (compat[u] >> v) & 1 == 1
            # Or: (compat[u] & color_c_mask) == color_c_mask
            
            # We maintain color_masks: list of bitmasks for each color
            # and color_max_weights
            
            assigned = False
            for c_idx, c_mask in enumerate(color_masks):
                # Check if u is compatible with all in c_mask
                if (compat[u] & c_mask) == c_mask:
                    # Assign u to color c_idx
                    color_masks[c_idx] |= (1 << u)
                    if weights[u] > color_max_weights[c_idx]:
                        color_max_weights[c_idx] = weights[u]
                    assigned = True
                    break
                    
            if not assigned:
                # New color
                color_masks.append(1 << u)
                color_max_weights.append(weights[u])
                
        return sum(color_max_weights)

    # We need color_masks and color_max_weights to be accessible in the recursion
    # But they are local to the bound function.
    # This is inefficient to recreate.
    # However, for N=255, it might be okay.
    
    # Better: Pass the state down?
    # The standard Tomita algorithm modifies the coloring incrementally.
    # That is complex to implement correctly in Python quickly.
    # Given the time limit, a simpler recursive B&B with recomputed bounds might suffice
    # if the initial LB is strong and the graph is sparse/dense appropriately.
    
    # Let's try a slightly optimized recursion.
    
    def bnb(candidates_mask, current_clique_mask, current_weight):
        nonlocal global_best_weight, global_best_set
        
        if time.time() > deadline:
            return
            
        # Upper bound
        # If candidates is empty, update best
        if candidates_mask == 0:
            if current_weight > global_best_weight:
                global_best_weight = current_weight
                global_best_set = current_clique_mask
            return
            
        # Compute bound
        # To make it faster, we can use a simpler bound: sum of weights of all candidates?
        # No, that's too loose.
        # Let's use the greedy coloring bound.
        
        # To avoid overhead of creating lists every time, we can just do it.
        # N is small.
        
        # Get nodes
        nodes = []
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            nodes.append(idx)
            m ^= lsb
            
        if not nodes:
             if current_weight > global_best_weight:
                 global_best_weight = current_weight
                 global_best_set = current_clique_mask
             return

        # Sort by weight desc
        nodes.sort(key=lambda x: -weights[x])
        
        # Greedy Coloring
        color_masks = []
        color_max_weights = []
        
        for u in nodes:
            assigned = False
            for c_idx, c_mask in enumerate(color_masks):
                if (compat[u] & c_mask) == c_mask:
                    color_masks[c_idx] |= (1 << u)
                    if weights[u] > color_max_weights[c_idx]:
                        color_max_weights[c_idx] = weights[u]
                    assigned = True
                    break
            if not assigned:
                color_masks.append(1 << u)
                color_max_weights.append(weights[u])
                
        bound = sum(color_max_weights)
        
        if current_weight + bound <= global_best_weight:
            return
            
        # Branch
        # Pick a node to branch on.
        # Usually, we branch on the last node in the sorted list?
        # Or we iterate through nodes in reverse order of the sorted list?
        # Standard Tomita: 
        #   Order candidates P by weight.
        #   For i from len(P)-1 down to 0:
        #       u = P[i]
        #       if weight(u) + sum(weights of P[0..i-1]) <= best: break
        #       Add u to clique
        #       New candidates = P[0..i-1] intersect compat[u]
        #       Recurse
        #       Remove u from P
        
        # Let's implement this specific order.
        # P is the sorted list of nodes in candidates_mask.
        P = nodes
        
        for i in range(len(P) - 1, -1, -1):
            if time.time() > deadline:
                return
                
            u = P[i]
            
            # Prune: If current_weight + weight[u] + sum(weights of P[0..i-1]) <= best
            # sum(weights of P[0..i-1]) is an upper bound for the rest.
            # We can precompute prefix sums?
            # Or just use the bound calculated above?
            # The bound calculated above is for the whole set P.
            # A tighter bound for the subset P[0..i] is needed.
            # For simplicity, let's just check if current_weight + weight[u] <= global_best_weight
            # If so, skip? No, we can add u and others.
            
            # Better prune:
            # max_possible = current_weight + weight[u] + sum(weights[v] for v in P[:i])
            # If max_possible <= global_best_weight, break.
            
            # Calculate sum of weights of P[:i]
            # This is O(i).
            # We can optimize by maintaining a running sum from the end?
            # Let's just do it for now.
            
            # Optimization: Precompute suffix sums of weights in P?
            # P is sorted by weight desc.
            # So P[0] is max weight.
            # sum(P[0..i-1]) is the sum of the top i weights.
            
            # Let's compute prefix sums of P
            prefix_sums = [0.0] * (len(P) + 1)
            for k in range(len(P)):
                prefix_sums[k+1] = prefix_sums[k] + weights[P[k]]
                
            # In the loop:
            # upper_bound_for_this_branch = current_weight + weights[u] + prefix_sums[i]
            if current_weight + weights[u] + prefix_sums[i] <= global_best_weight:
                # Since P is sorted desc, for any j < i, weight[P[j]] >= weight[u]? No, P is sorted desc, so P[0] is largest.
                # Wait, if P is sorted desc, P[0] > P[1] ...
                # Then prefix_sums[i] is sum of P[0]...P[i-1].
                # If this bound fails, it will fail for all j < i?
                # Yes, because weight[P[j]] <= weight[u] is FALSE. P[j] >= P[i].
                # Wait, P is sorted descending. P[0] is max.
                # So P[i] is smaller than P[0].
                # If we iterate i from len(P)-1 down to 0:
                # u = P[len(P)-1] is the SMALLEST weight.
                # prefix_sums[i] is sum of P[0]...P[i-1].
                # As i decreases, u gets larger, and prefix_sums[i] gets smaller.
                # The bound `current_weight + weights[u] + prefix_sums[i]` is not monotonic in a simple way to allow early break.
                # However, if `current_weight + weights[u] + prefix_sums[i] <= best`,
                # does it imply for j < i?
                # weights[P[j]] > weights[u]. prefix_sums[j] < prefix_sums[i].
                # It's not strictly monotonic. So we can't just break. We must continue.
                # But we can skip this specific u.
                continue
                
            # Branch on u
            new_clique_mask = current_clique_mask | (1 << u)
            new_weight = current_weight + weights[u]
            
            # New candidates: P[0..i-1] that are compatible with u
            # P[0..i-1] is the set of nodes before u in the sorted order.
            # We need to filter them by compatibility with u.
            
            # Construct bitmask of P[0..i-1]
            # This is O(i).
            # Then AND with compat[u].
            
            # To speed up, we can just iterate P[0..i-1] and check compat.
            # But we need to pass a bitmask to the next recursion.
            
            # Let's build the new candidates mask.
            new_cand_mask = 0
            for j in range(i):
                v = P[j]
                if (compat[u] >> v) & 1:
                    new_cand_mask |= (1 << v)
                    
            # Recurse
            bnb(new_cand_mask, new_clique_mask, new_weight)
            
            # Remove u from P? 
            # In Tomita, P is modified in place.
            # Here P is a list. We don't modify it because we iterate downwards.
            # The next iteration will use P[i-1] as u.
            # The candidates for the next iteration should NOT include u.
            # But my `new_cand_mask` construction only takes P[0..i-1], so u is excluded.
            # This is correct.

    # Start B&B
    # Initial candidates: all nodes
    # Initial clique: empty
    # Initial weight: 0
    
    # To improve initial bound, we can start with the greedy solution's set?
    # No, B&B explores all possibilities.
    # But we can order the initial P by weight.
    
    # One issue: The recursion might be deep and slow in Python.
    # N=255.
    # If the graph is dense, the clique is small.
    # If the graph is sparse, the clique is large.
    # LCM > 256 condition:
    # Primes > 16 are compatible with almost everything (except multiples).
    # There are many primes.
    # The graph is likely fairly dense.
    # So the maximum clique might be smallish?
    # Actually, if we pick all primes > 16, their LCM is product > 256.
    # Primes in [2, 256]: 19, 23, ..., 251.
    # Count: ~50.
    # Sum of reciprocals of these primes is significant.
    # Can we add composite numbers?
    # E.g., 17 and 19. lcm=323 > 256.
    # 17 and 20. lcm=340 > 256.
    # 17 and 18. lcm=306 > 256.
    # 17 and 16. lcm=272 > 256.
    # 17 and 15. lcm=255 <= 256. Conflict.
    # So 17 conflicts with 15, 10, 5, 3, 2?
    # lcm(17, 2)=34 <= 256? No, 34 > 256 is FALSE. 34 <= 256 is TRUE.
    # So 17 conflicts with 2.
    # 17 conflicts with any divisor of 256/17? No.
    # lcm(17, x) = 17x / gcd(17,x).
    # If x is multiple of 17, lcm=x. If x<=256, lcm<=256.
    # So 17 conflicts with 34, 51, ..., 255.
    # Also conflicts with x where 17x/g <= 256.
    
    # The B&B should handle this.
    
    try:
        # Set recursion limit just in case
        sys.setrecursionlimit(3000)
        bnb((1 << n) - 1, 0, 0.0)
    except Exception as e:
        # If B&B fails or times out, use the best found so far (which includes the greedy LB)
        pass
        
    # Extract result from global_best_set
    result = []
    m = global_best_set
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(candidates[idx])
        m ^= lsb
        
    if not result:
        # Fallback to greedy if something went wrong
        result = [candidates[i] for i in range(n) if (best_set >> i) & 1]
        
    return result
