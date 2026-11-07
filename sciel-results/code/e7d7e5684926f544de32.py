import math
import time
from itertools import product

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
    
    best_span = float('inf')
    best_values = None
    
    start_time = time.time()
    deadline = 85.0  # Leave 5s margin
    
    # Strategy: For each candidate minimum value M (the smallest representative),
    # we want to find the smallest possible maximum value such that all 10 values
    # are distinct, positive, <= 1000000, and each is a multiple of its divisor.
    
    # Key insight: If we fix the minimum to be some value m, then for each divisor d_i,
    # the representative x_i must satisfy x_i >= m and x_i % d_i == 0.
    # The smallest such x_i is m if m % d_i == 0, else the next multiple.
    # But we also need all x_i to be distinct.
    
    # Better approach: enumerate possible minimum values m, and for each m,
    # compute the minimal feasible set where min is exactly m.
    
    # For a given m, for each i, the smallest multiple of d_i that is >= m is:
    #   x_i = m + ((d_i - m % d_i) % d_i)
    # But we need x_i >= m and x_i % d_i == 0.
    # The smallest such value is: if m % d_i == 0 then m, else m + (d_i - m % d_i)
    
    # However, we also need all x_i to be distinct. If two x_i collide, we need to 
    # bump one of them up to the next multiple.
    
    # Let's try a different angle: iterate over candidate spans S from 1 upward.
    # For a given span S and candidate minimum m, check if there exist distinct multiples
    # x_i of d_i such that m <= x_i <= m + S for all i.
    
    # But this is a constraint satisfaction problem. Let's be smarter.
    
    # Observation: The LCM is 30030. The structure repeats every 30030.
    # So we only need to consider m in [1, 30030] (or [0, 30029] for the offset).
    
    # For each offset W in [0, L-1], consider the "base" assignment:
    #   x_i(W) = W + ((d_i - W % d_i) % d_i)
    # This gives the smallest multiple of d_i that is >= W.
    # If W % d_i == 0, then x_i(W) = W.
    # Otherwise, x_i(W) is the next multiple.
    
    # The span of this base assignment is max(x_i) - min(x_i).
    # But min(x_i) might not be W if W is not a multiple of any d_i.
    
    # Let's refine: for each W, compute x_i(W) as above. Then:
    #   actual_min = min(x_i(W))
    #   actual_max = max(x_i(W))
    #   span = actual_max - actual_min
    #   Check distinctness.
    
    # Then try to reduce the span by adjusting individual x_i upward (to next multiple)
    # if there are collisions, or by shifting the whole window.
    
    # Actually, let's just do the brute force over W in [0, L) and compute the 
    # minimal span achievable with min value being the minimum of the base assignment.
    
    # For each W:
    #   base_vals = [W + ((d_i - W % d_i) % d_i) for i]
    #   If all distinct, span = max - min.
    #   If not distinct, we need to resolve collisions by bumping up.
    
    # Collision resolution: if x_i == x_j for some i != j, we must bump one of them.
    # Greedy: sort by value, and for each value that appears multiple times, bump 
    # the later ones (in some order) to the next multiple.
    
    # Let's implement a greedy collision resolver:
    #   Start with base_vals.
    #   Sort indices by (base_val, index) to break ties consistently.
    #   Process in order. Keep track of used values.
    #   For each x_i, if x_i is already used, increment to next multiple of d_i.
    #   This gives a valid distinct set with min = min(base_vals) (since we only go up).
    
    # Wait, but bumping up might increase the max. We want to minimize the span.
    # The greedy approach of bumping the "later" ones (in sorted order) is reasonable.
    
    # Let's try this for all W in [0, L).
    
    for W in range(L):
        # Compute base values
        base_vals = [W + ((d - W % d) % d) for d in divisors]
        
        # Greedy collision resolution
        # Sort indices by base value, then by index
        indices = sorted(range(n), key=lambda i: (base_vals[i], i))
        used = set()
        resolved = [0] * n
        
        for idx in indices:
            v = base_vals[idx]
            d = divisors[idx]
            while v in used:
                v += d
            used.add(v)
            resolved[idx] = v
        
        # Check bounds
        if any(v > 1000000 for v in resolved):
            continue
        
        span = max(resolved) - min(resolved)
        if span < best_span:
            best_span = span
            best_values = resolved[:]
        
        if time.time() - start_time > deadline:
            break
    
    # Now, local polish: for the best W found, try shifting each element by -1, 0, or +1 multiples
    # to see if we can reduce the span further.
    
    # Find the W that gave us best_values
    # Actually, let's just try to polish best_values directly.
    # For each element, try shifting by -d_i, 0, or +d_i.
    # But we need to maintain distinctness and positivity.
    
    # Let's do a more targeted polish: for the current best, try all combinations
    # of shifting each element by -1, 0, or +1 (in terms of multiples of d_i).
    
    if best_values is not None:
        for shifts in product([-1, 0, 1], repeat=n):
            test_vals = [best_values[i] + shifts[i] * divisors[i] for i in range(n)]
            
            if any(v <= 0 or v > 1000000 for v in test_vals):
                continue
            
            if len(set(test_vals)) != n:
                continue
            
            span = max(test_vals) - min(test_vals)
            if span < best_span:
                best_span = span
                best_values = test_vals[:]
        
        if time.time() - start_time > deadline:
            break
    
    # Fallback
    if best_values is None:
        best_values = [d * (i + 1) for i, d in enumerate(divisors)]
    
    return [int(v) for v in best_values]

if __name__ == '__main__':
    res = construct()
    print(res)
    print("Span:", max(res) - min(res))
