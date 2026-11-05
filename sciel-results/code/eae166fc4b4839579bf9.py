import time
import sys
from fractions import Fraction

def construct():
    start_time = time.time()
    # Conservative deadline to ensure we return before the 90s hard limit
    deadline = start_time + 85.0
    n = 256
    
    # Weights are 1/i
    # For B&B, we work with integers to avoid floating point issues in bounds
    # Sum of 1/i = (Sum of (L/i)) / L
    # Let L = LCM(1..256). This is huge, but we only need it for final serialization.
    # For B&B, we can use a common denominator that is large enough to preserve order?
    # Actually, for B&B, we just need to maximize the sum.
    # We can use floating point for the bound if we are careful, or use exact rationals.
    # Given n=256, the number of vertices is 256.
    # The number of 3-APs is O(n^2) ~ 16k.
    
    # Conflict Graph:
    # Vertex i conflicts with j if there exists k such that {i, j, k} is a 3-AP.
    # i and j form a 3-AP with k if:
    # 1. i + j = 2k (i, k, j or j, k, i) -> k = (i+j)/2. Requires i+j even.
    # 2. i + k = 2j (i, j, k) -> k = 2j - i.
    # 3. j + k = 2i (j, i, k) -> k = 2i - j.
    # Basically, i and j cannot both be in the set if they are part of ANY 3-AP.
    # Two numbers i, j are incompatible if they form a 3-AP with some third number in [1, n].
    # This happens if:
    # - (i+j) is even and (i+j)//2 is in [1, n] (which is always true for i,j in [1,n] if i!=j? No, if i=1, j=255, k=128. Yes. If i=1, j=3, k=2. Yes.)
    #   Wait, if i and j are in [1, n], their average is in [1, n] iff i+j <= 2n. Since i,j <= n, i+j <= 2n.
    #   So any pair i, j with i+j even is a conflict?
    #   Let's check: 1, 3 -> 2. Yes.
    #   1, 5 -> 3. Yes.
    #   So if i+j is even, they conflict.
    # - Or if 2j - i is in [1, n] and 2j - i != i, 2j - i != j.
    #   This corresponds to i, j, k with i < j < k or k < j < i.
    #   If i < j, k = 2j - i. If k <= n and k != j (i!=j), then conflict.
    #   If i > j, k = 2j - i. If k >= 1 and k != j, then conflict.
    
    # Let's build the adjacency list.
    # adj[i] is a set of j such that i and j conflict.
    adj = [set() for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            # Check if i, j form a 3-AP with some k in [1, n]
            # Case 1: k is between i and j. i + j = 2k.
            if (i + j) % 2 == 0:
                k = (i + j) // 2
                if 1 <= k <= n:
                    adj[i].add(j)
                    adj[j].add(i)
            
            # Case 2: j is between i and k. i, j, k.
            # 2j = i + k => k = 2j - i
            k = 2 * j - i
            if k > j and k <= n:
                adj[i].add(j)
                adj[j].add(i)
            
            # Case 3: i is between k and j. k, i, j.
            # 2i = k + j => k = 2i - j
            k = 2 * i - j
            if k < i and k >= 1:
                adj[i].add(j)
                adj[j].add(i)

    # Precompute bitmasks for neighbors?
    # 256 bits. Python integers can handle this.
    # mask[i] is an integer where bit j is 1 if i and j conflict.
    masks = [0] * (n + 1)
    for i in range(1, n + 1):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m

    # Weights
    # To do B&B with exact arithmetic, we can use Fractions, but it's slow.
    # Let's use a large common denominator for the weights to keep them as integers?
    # LCM(1..256) is approx 10^100. Too big for standard int ops efficiently? Python handles big ints, but it might be slow.
    # Alternative: Use floating point for the bound, but exact Fraction for the incumbent?
    # Or use a scaled integer weight: w[i] = 10^18 // i ?
    # 1/i approx 10^18/i. The error is small.
    # Let's try using Fractions for the final result, but for B&B, use a high-precision float or integer scaling.
    # Given the "exact numerator/denominator" requirement in the prompt's score description,
    # the evaluator likely checks the exact sum.
    # I will compute the best set using B&B with integer weights scaled by a large factor,
    # then verify and compute the exact sum for the returned set.
    
    SCALE = 10**18
    weights = [0] * (n + 1)
    for i in range(1, n + 1):
        weights[i] = SCALE // i

    # Sort vertices by weight descending to find good incumbents early
    # And to order the branching
    vertices = list(range(1, n + 1))
    vertices.sort(key=lambda x: weights[x], reverse=True)
    
    # Remap indices to 0..N-1 for bit manipulation convenience?
    # Or just use 1..n indices in the bitmask.
    # Let's stick to 1..n indices.
    
    best_set = set()
    best_weight = 0
    
    # Initial heuristic: Greedy
    # Pick vertices in order of weight, if no conflict with picked.
    current_set = set()
    current_weight = 0
    for v in vertices:
        # Check if v conflicts with any in current_set
        conflict = False
        for u in current_set:
            if u in adj[v]:
                conflict = True
                break
        if not conflict:
            current_set.add(v)
            current_weight += weights[v]
    
    best_set = current_set
    best_weight = current_weight
    
    # B&B State
    # We process vertices in `vertices` order.
    # At step k, we decide for vertices[0..k-1].
    # We need to track which vertices are "available" (not conflicting with chosen ones).
    # Actually, standard MWIS B&B:
    # State: (index k, available_mask, current_weight)
    # available_mask contains bits for vertices from k to N that are not conflicting with chosen vertices.
    # When we choose vertices[k], we remove it and its neighbors from available_mask.
    # When we skip vertices[k], we just remove it from available_mask.
    
    # To make this work with 1..n indices:
    # We need to map the sorted order to the original indices.
    # Let order[k] = original index of the k-th vertex in sorted order.
    order = vertices
    
    # Precompute masks for "neighbors in the remaining set"?
    # It's easier to just maintain the `available` mask as a Python int.
    # Initially, all vertices 1..n are available.
    # But we process in `order`.
    # If we pick v, we must remove v and all neighbors of v from `available`.
    
    # Optimization: 
    # We can use a recursive function.
    # To avoid deep recursion limits, we might need to be careful, but N=256 is small enough for recursion depth if we prune well?
    # Actually, depth is 256. Python default recursion limit is 1000. So it's fine.
    
    sys.setrecursionlimit(3000)
    
    def branch_and_bound(k, available, current_weight):
        nonlocal best_set, best_weight
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if k == len(order):
            if current_weight > best_weight:
                best_weight = current_weight
                # Reconstruct set
                # This is tricky in B&B because we don't store the path.
                # We can store the path, or just return the weight and reconstruct later?
                # Storing the path is safer.
                pass
            return
        
        v = order[k]
        
        # Upper bound: current_weight + sum of weights of all available vertices from k onwards
        # To make this fast, we can precompute suffix sums?
        # But `available` changes.
        # A simple bound: current_weight + sum(weights[u] for u in range(1, n+1) if (available >> u) & 1)
        # This is O(n) per node. With many nodes, this might be slow.
        # Better bound: 
        # We can precompute a list of weights in `order`.
        # But `available` is a subset.
        # Let's use a greedy coloring bound (Tomita's algorithm) or just a simple sum bound first.
        # Given n=256, sum bound might be loose.
        # Let's try a simple sum bound first and see if it's fast enough.
        
        # Calculate sum of weights of available vertices
        avail_sum = 0
        temp_avail = available
        while temp_avail:
            # Get lowest set bit
            lsb = temp_avail & -temp_avail
            idx = lsb.bit_length() - 1
            if idx > 0: # index 0 is unused
                avail_sum += weights[idx]
            temp_avail ^= lsb
            
        if current_weight + avail_sum <= best_weight:
            return
        
        # Branch 1: Include v
        # New available: remove v and its neighbors
        new_avail = available & ~(masks[v] | (1 << v))
        
        # We need to update the set. 
        # To avoid passing a set around, we can just track the weight and reconstruct at the end?
        # Or we can store the chosen vertices in a global list and backtrack.
        
        # Let's use a global list `current_path`
        
        # Branch 2: Exclude v
        # New available: remove v
        new_avail_excl = available & ~(1 << v)
        
        # Heuristic: Include first if it looks promising?
        # Usually, include high weight vertices first.
        
        # We need to track the actual set to return it.
        # Let's modify the function to take the current set as a frozenset or just pass a list?
        # Passing a set is expensive to copy.
        # Let's use a global `current_set` and add/remove.
        
        # However, B&B explores many branches.
        # Let's store the best set by copying when we update best_weight.
        
        # Recurse Include
        current_set.add(v)
        branch_and_bound(k + 1, new_avail, current_weight + weights[v])
        current_set.remove(v)
        
        # Recurse Exclude
        branch_and_bound(k + 1, new_avail_excl, current_weight)
        
    # The simple sum bound O(n) might be too slow if the tree is large.
    # Let's optimize the bound.
    # We can precompute weights as a list indexed by bit position?
    # weights_list[bit] = weight.
    
    # Let's try to run it. If it's too slow, we'll fall back to the greedy result.
    
    # To make the bound faster, we can approximate the sum of available weights.
    # Or we can use a tighter bound based on matching or coloring, but that's complex.
    # Let's stick to the simple sum bound but optimize the calculation.
    
    # Actually, we can precompute the weight of each bit.
    # weight_of_bit = [0]*(n+1)
    # for i in range(1, n+1): weight_of_bit[i] = weights[i]
    
    # Let's refine the bound calculation:
    # Instead of iterating through bits, we can just sum weights of available vertices.
    # Since n=256, iterating 256 bits is fast in Python if done carefully?
    # `while temp_avail:` loop is fast.
    
    # One issue: `current_set` is global.
    current_set = set()
    
    # Initial available mask: all bits 1..n set
    full_mask = 0
    for i in range(1, n + 1):
        full_mask |= (1 << i)
        
    # Start B&B
    # We start with k=0, available=full_mask, current_weight=0
    # But we already have a good incumbent from Greedy.
    # We can also initialize `current_set` to empty for the B&B.
    
    # To speed up B&B, we can order the vertices.
    # We already sorted `order` by weight desc.
    
    # Let's run B&B
    branch_and_bound(0, full_mask, 0)
    
    # The B&B might have updated `best_set`? 
    # In my draft, I didn't update `best_set` inside `branch_and_bound`.
    # I need to fix that.
    
    # Let's rewrite the B&B to properly track the set.
    
    best_set_final = set(best_set) # Keep greedy result as fallback
    best_weight_final = best_weight
    
    # Reset for B&B
    current_set_bnb = set()
    
    def bnb(k, available, current_weight):
        nonlocal best_set_final, best_weight_final
        nonlocal deadline
        
        if time.time() > deadline:
            return
            
        if k == len(order):
            if current_weight > best_weight_final:
                best_weight_final = current_weight
                best_set_final = set(current_set_bnb)
            return
            
        # Bound
        # Sum of weights of available vertices
        avail_sum = 0
        temp_avail = available
        while temp_avail:
            lsb = temp_avail & -temp_avail
            idx = lsb.bit_length() - 1
            if idx > 0:
                avail_sum += weights[idx]
            temp_avail ^= lsb
            
        if current_weight + avail_sum <= best_weight_final:
            return
            
        v = order[k]
        
        # Branch Include
        if (available >> v) & 1:
            current_set_bnb.add(v)
            new_avail = available & ~(masks[v] | (1 << v))
            bnb(k + 1, new_avail, current_weight + weights[v])
            current_set_bnb.remove(v)
            
        # Branch Exclude
        new_avail_excl = available & ~(1 << v)
        bnb(k + 1, new_avail_excl, current_weight)
        
    bnb(0, full_mask, 0)
    
    # Use the best set found
    S = best_set_final
    
    # If B&B didn't find anything better (or didn't run), S is the greedy set.
    
    # Final Validation
    # Ensure S is a valid 3-AP free set.
    # And compute the exact sum.
    
    # The problem asks for a subset.
    # Return sorted list.
    
    return sorted(S)

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
    # Calculate exact sum for verification
    from fractions import Fraction
    total = Fraction(0, 1)
    for x in res:
        total += Fraction(1, x)
    print(total)
