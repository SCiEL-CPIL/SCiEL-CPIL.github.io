import time
from math import gcd
from fractions import Fraction
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 256
    # Candidates are 2 to 256
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # Weights: 1/x
    # We want to maximize sum(1/x)
    # To make B&B effective, we should process vertices with high weight first.
    # High weight means small x.
    # So we sort candidates in ascending order of value.
    # candidates are already 2, 3, ..., 256.
    # So index 0 is 2 (weight 1/2), index 1 is 3 (weight 1/3), etc.
    # This is already sorted by weight descending.
    
    weights = [1.0 / candidates[i] for i in range(n)]
    
    # Precompute conflict masks
    # conflict[i] is a bitmask of j > i such that lcm(candidates[i], candidates[j]) <= 256
    # Note: We only need to store conflicts with higher indices to keep masks small and directional for B&B
    # However, standard MWIS B&B often uses full adjacency or careful directional logic.
    # Let's use full adjacency for simplicity but ensure we only consider j > i in the branching step.
    
    adj = [0] * n
    for i in range(n):
        a = candidates[i]
        mask = 0
        for j in range(i + 1, n):
            b = candidates[j]
            g = gcd(a, b)
            # Check lcm(a, b) <= 256
            # lcm(a, b) = a * b // g
            # To avoid large numbers, check: a // g * b <= 256
            if (a // g) * b <= 256:
                mask |= (1 << j)
        adj[i] = mask
        
    # Suffix sums of weights for upper bound
    # suffix[i] = sum(weights[i:n])
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + weights[i]
        
    best_sum = 0.0
    best_set_mask = 0
    
    # Optimization: 
    # The graph is quite sparse for large numbers. 
    # Most conflicts involve small numbers.
    # Numbers > 128 have no multiples in [2, 256] except themselves.
    # If x > 128, lcm(x, y) <= 256 implies y must be a divisor of x or share factors such that product/gcd <= 256.
    # Actually, if x > 128, the only y such that lcm(x,y) <= 256 is y=x (not allowed as distinct) or y is a divisor of x?
    # No, lcm(x, y) is a multiple of x. If x > 128, the smallest multiple of x is x itself (if y divides x, lcm=x).
    # If y does not divide x, lcm >= 2x > 256.
    # So for x > 128, y must divide x.
    # This structure is very specific.
    
    # Standard B&B for MWIS
    # We maintain a set of available vertices.
    # State: (current_index, current_sum, current_mask, available_mask)
    # available_mask contains bits for vertices j >= current_index that are compatible with current_mask.
    
    # To speed up bound calculation, we can precompute weight sums for chunks? 
    # Or just use suffix sums as a loose bound initially, then tighten.
    # A tight bound is sum of weights in available_mask.
    # Calculating this every step is O(N/word_size) or O(k).
    
    def calc_bound(available_mask, start_idx):
        # Upper bound is sum of weights of all available vertices from start_idx onwards
        # Since we process in order, available_mask only has bits >= start_idx.
        # We can sum weights of set bits.
        # To make this faster, we can use a precomputed table for 16-bit chunks?
        # Or just iterate. N=255, so at most 255 bits.
        # In practice, the mask becomes sparse quickly.
        s = 0.0
        m = available_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    # Recursive B&B
    # To avoid recursion depth issues and overhead, we can use a stack or increase recursion limit.
    # Depth is at most 255. Recursion is fine.
    sys.setrecursionlimit(5000)
    
    def bnb(idx, current_sum, current_mask, available_mask):
        nonlocal best_sum, best_set_mask
        
        if time.time() > deadline:
            return
            
        # Prune 1: If no more vertices
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set_mask = current_mask
            return
            
        # Prune 2: Upper bound
        # The maximum possible additional sum is the sum of weights of all available vertices.
        # We can approximate this with suffix[idx] if available_mask is "full", but it's rarely full.
        # Let's compute the exact sum of available weights.
        # Optimization: If current_sum + suffix[idx] <= best_sum, prune.
        # This is a very fast check.
        if current_sum + suffix[idx] <= best_sum:
            return
            
        # Tighter bound: sum of weights in available_mask
        # If current_sum + calc_bound(available_mask, idx) <= best_sum, prune.
        # calc_bound is O(k). Let's try it.
        bound = current_sum
        m = available_mask
        # Only sum weights for bits >= idx (which is true by construction of available_mask in this scheme)
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            bound += weights[j]
            m ^= lsb
            
        if bound <= best_sum:
            return
            
        # Branching
        # We must pick a vertex to branch on.
        # Standard strategy: Pick the first available vertex in order (smallest index, highest weight).
        # Find the lowest set bit in available_mask that is >= idx.
        # Since available_mask only has bits >= idx, the lowest set bit is the candidate.
        
        # Wait, is it always optimal to branch on the lowest index?
        # In MWIS, branching on a vertex with high degree or high weight is often better.
        # Here, weights are sorted descending, so lowest index = highest weight.
        # Let's stick with the lowest set bit.
        
        # Find lowest set bit
        lsb = available_mask & -available_mask
        v = lsb.bit_length() - 1
        
        # Option 1: Include v
        # New available = (available_mask & ~lsb) & ~adj[v]
        # Note: adj[v] contains conflicts with j > v.
        # We must remove v from available.
        # We must remove all neighbors of v from available.
        
        new_available = (available_mask & ~lsb) & ~adj[v]
        
        # Note: adj[v] might have bits < v? No, we built adj[i] only with j > i.
        # So adj[v] only has bits > v.
        # available_mask has bits >= idx. Since v is the lowest set bit in available, v >= idx.
        # The intersection handles the removal correctly.
        
        bnb(v + 1, current_sum + weights[v], current_mask | lsb, new_available)
        
        # Option 2: Exclude v
        # New available = available_mask & ~lsb
        # The next vertex to consider is the next lowest set bit in the new available mask?
        # Or do we just increment idx?
        # If we exclude v, the next vertex to consider is the next one in the sequence.
        # However, our "idx" parameter in the recursive call above was v+1.
        # For the exclude branch, we should consider the next vertex after v.
        # But what if there are vertices between idx and v that are not in available?
        # They are already excluded.
        # So the next vertex to branch on is the lowest set bit in (available_mask & ~lsb).
        # Let's call this next_v.
        # If we pass idx = v+1, we might skip some vertices? No, because available_mask only contains valid candidates.
        # The lowest set bit in (available_mask & ~lsb) is the next candidate.
        
        new_available_ex = available_mask & ~lsb
        if new_available_ex == 0:
             if current_sum > best_sum:
                 best_sum = current_sum
                 best_set_mask = current_mask
        else:
             # We need to find the next index to branch on.
             # It is the lowest set bit in new_available_ex.
             # We can just pass idx = v + 1? 
             # No, because the next available vertex might be v+1, or v+2, etc.
             # The bound calculation uses the mask, so it's accurate regardless of idx.
             # But the "suffix[idx]" check uses idx.
             # If we pass v+1, suffix[v+1] is the sum of weights from v+1 to end.
             # This is a valid upper bound for the remaining vertices.
             # So passing v+1 is correct.
             bnb(v + 1, current_sum, current_mask, new_available_ex)

    # Initial call
    # All vertices are available.
    initial_available = (1 << n) - 1
    bnb(0, 0.0, 0, initial_available)
    
    # Reconstruct best set
    if best_set_mask == 0:
        # Fallback: Just pick the smallest number 2? 
        # Or a greedy heuristic?
        # If B&B failed or found nothing (shouldn't happen as 2 is always valid alone), 
        # let's do a simple greedy.
        result = []
        avail = set(range(n))
        while avail:
            # Pick highest weight (lowest index)
            v = min(avail)
            result.append(candidates[v])
            avail.remove(v)
            # Remove conflicts
            to_remove = set()
            for u in avail:
                if (adj[v] >> u) & 1 or (adj[u] >> v) & 1:
                    to_remove.add(u)
            avail -= to_remove
        return result

    result = []
    m = best_set_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(candidates[idx])
        m ^= lsb
        
    return result

if __name__ == "__main__":
    res = construct()
    print(f"Result size: {len(res)}")
    # Exact sum for verification
    exact_sum = sum(Fraction(1, x) for x in res)
    print(f"Exact Sum: {exact_sum}")
    print(f"Float Sum: {sum(1/x for x in res)}")
    print(res[:20])
