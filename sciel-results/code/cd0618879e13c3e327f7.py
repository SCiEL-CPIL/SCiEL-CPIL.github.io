import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    N = 256
    # Vertices are 2..256
    vertices = list(range(2, 257))
    n = len(vertices)
    
    # Map value to index in vertices list
    # We will work with indices 0..n-1 corresponding to vertices[0..n-1]
    # Sort vertices by value ascending (so weights descending) to help B&B
    # Actually, the order in the list determines the index.
    # Let's keep vertices sorted ascending: 2, 3, ..., 256.
    # Index i corresponds to value vertices[i] = i+2.
    
    weights = [1.0 / v for v in vertices]
    
    # Precompute conflict masks
    # conflict[i] is a bitmask of j such that lcm(vertices[i], vertices[j]) <= 256
    conflict = [0] * n
    
    for i in range(n):
        vi = vertices[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            vj = vertices[j]
            # lcm(vi, vj) = vi * vj // gcd(vi, vj)
            # To avoid overflow or just for speed, check vi * vj <= 256 * gcd(vi, vj)
            # Or just compute lcm directly, it's fast enough for 255^2
            g = gcd(vi, vj)
            l = (vi // g) * vj
            if l <= 256:
                mask |= (1 << j)
        conflict[i] = mask

    # B&B Solver
    best_sum = 0.0
    best_mask = 0
    
    # Helper to sum weights of available mask
    # Optimized: precompute prefix sums? No, mask is arbitrary.
    # Just iterate bits.
    def sum_weights(mask):
        s = 0.0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    # To speed up sum_weights, we can use a lookup table for 16-bit chunks?
    # Or just accept the O(popcount) cost. Max popcount is 255.
    # 255 operations per node. If we visit 10^5 nodes, 2.5 * 10^7 ops. OK in Python?
    # Maybe. Let's try.
    
    # Another optimization: 
    # If we always process vertices in a fixed order (e.g., index 0, then 1, etc.),
    # we can use a recursive function `dfs(idx, available_mask, current_sum)`
    # where `idx` is the next vertex to consider.
    # But `available_mask` might have bits set for indices < idx? No, if we remove them.
    # If we enforce that we only consider vertices >= idx, then `available_mask` only has bits >= idx.
    # This allows us to use prefix sums of weights for the upper bound!
    # UB = current_sum + prefix_sum_weights[idx] (sum of weights from idx to n-1)
    # Wait, `available_mask` might not contain all vertices from idx to n-1.
    # So the bound `sum(weights[idx:])` is an upper bound on `sum(weights[available_mask])`.
    # This is a valid upper bound.
    
    # Let's implement this ordered B&B.
    # It is more efficient because we don't need to calculate sum_weights(mask) dynamically,
    # we can just use the precomputed suffix sum of weights.
    
    suffix_weights = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_weights[i] = suffix_weights[i + 1] + weights[i]
        
    # dfs(idx, available_mask, current_sum)
    # idx: next vertex index to consider
    # available_mask: bitmask of vertices available (only bits >= idx should be set? 
    #                Actually, if we skip a vertex, we clear its bit. 
    #                If we include a vertex, we clear it and its neighbors.
    #                Neighbors can be < idx? No, because we process in order.
    #                But wait, if we include vertex i, we remove neighbors j.
    #                If j < i, it was already processed. 
    #                If j > i, it is removed from available.
    #                So available_mask only contains bits >= idx?
    #                Yes, if we start with all bits set, and only clear bits.
    #                And we iterate idx from 0 to n-1.
    #                So at step idx, all bits < idx are 0 in available_mask.
    
    import sys
    sys.setrecursionlimit(10000)

    def dfs(idx, available_mask, current_sum):
        nonlocal best_sum, best_mask
        
        if time.time() > deadline:
            return
            
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_mask = available_mask # This is 0 at the end? 
                # We need to track the chosen set.
                # It's easier to track chosen_mask.
            return
            
        # Upper Bound
        # Max possible sum is current_sum + sum of weights of all remaining available vertices.
        # Since available_mask only has bits >= idx, and suffix_weights[idx] is sum of weights[idx..n-1],
        # sum(weights[available_mask]) <= suffix_weights[idx].
        ub = current_sum + suffix_weights[idx]
        if ub <= best_sum:
            return
            
        # If no vertices available from idx onwards
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                # Need to recover set from best_mask? 
                # Let's track chosen_mask in state.
            return

        # Pick a vertex to branch on.
        # We must pick a vertex from available_mask.
        # To maintain the "idx" logic, we should pick the smallest index in available_mask?
        # Or just advance idx to the next available vertex?
        
        # Let's find the next available vertex k >= idx.
        # If none, return.
        # Actually, if available_mask has bits, let k be the lowest set bit >= idx.
        # But available_mask might have gaps.
        
        # Efficient way:
        # If bit idx is NOT in available_mask, we can just skip to next.
        # But we need to branch on "Exclude idx" if idx is available?
        # If idx is NOT available, we don't branch, we just move to idx+1.
        # If idx IS available, we branch: Include idx or Exclude idx.
        
        if not (available_mask & (1 << idx)):
            # idx is not available, skip it
            dfs(idx + 1, available_mask, current_sum)
        else:
            # idx is available
            
            # Branch 1: Include idx
            # New sum: current_sum + weights[idx]
            # New available: available_mask with idx and its neighbors removed.
            # Neighbors are in conflict[idx].
            # We need to remove bits from available_mask.
            # new_avail = available_mask & ~(1 << idx) & ~conflict[idx]
            # Note: conflict[idx] might have bits < idx? 
            # Since we process in order, bits < idx are already 0 in available_mask.
            # So `~conflict[idx]` might have 0s at < idx, but AND with available_mask (0s at < idx) is fine.
            
            new_avail = available_mask & ~(1 << idx) & ~conflict[idx]
            dfs(idx + 1, new_avail, current_sum + weights[idx])
            
            # Branch 2: Exclude idx
            new_avail = available_mask & ~(1 << idx)
            dfs(idx + 1, new_avail, current_sum)

    # We need to track the best set.
    # Modifying dfs to track chosen_mask is better.
    # Or, we can just run the B&B to find best_sum, then reconstruct?
    # No, reconstruction is hard without storing path.
    # Let's add chosen_mask to state.
    
    best_sum = 0.0
    best_chosen_mask = 0

    def dfs2(idx, available_mask, current_sum, chosen_mask):
        nonlocal best_sum, best_chosen_mask
        
        if time.time() > deadline:
            return
            
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_chosen_mask = chosen_mask
            return
            
        ub = current_sum + suffix_weights[idx]
        if ub <= best_sum:
            return
            
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_chosen_mask = chosen_mask
            return

        if not (available_mask & (1 << idx)):
            dfs2(idx + 1, available_mask, current_sum, chosen_mask)
        else:
            # Include idx
            new_avail = available_mask & ~(1 << idx) & ~conflict[idx]
            dfs2(idx + 1, new_avail, current_sum + weights[idx], chosen_mask | (1 << idx))
            
            # Exclude idx
            new_avail = available_mask & ~(1 << idx)
            dfs2(idx + 1, new_avail, current_sum, chosen_mask)

    initial_mask = (1 << n) - 1
    dfs2(0, initial_mask, 0.0, 0)

    # Extract result
    result_set = []
    for i in range(n):
        if (best_chosen_mask >> i) & 1:
            result_set.append(vertices[i])
            
    return result_set
