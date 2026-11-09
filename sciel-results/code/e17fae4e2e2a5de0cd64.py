import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s limit
    
    p = 1009
    n = p - 1  # 1008, the upper bound of our interval
    
    # Precompute modular inverses for all residues 1..p-1
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # A value x is a "blocker" if it cannot be in a valid interval.
    # x is a blocker if:
    #   1. x is self-inverse: x^2 % p == 1, i.e., x == 1 or x == p-1 (1008)
    #   2. x's inverse is "smaller" in the sense that we'd want to exclude
    #      the smaller of the pair to break the pair. Actually, let's think differently.
    #
    # For an interval [s, e] to be valid, NO pair (x, y) in [s,e] should satisfy x*y % p == 1.
    # This includes x=y (so 1 and 1008 are always excluded).
    # For x != y: if x is in [s,e], then inv[x] must NOT be in [s,e].
    #
    # Strategy: For each possible start s, find the maximum valid end e.
    # An element e is invalid as an end if:
    #   - e is self-inverse (e==1 or e==1008)
    #   - inv[e] is in [s, e-1] (i.e., s <= inv[e] < e)
    #
    # But we need ALL elements in [s,e] to be mutually non-inverting.
    # A cleaner approach: build a boolean array "forbidden" where forbidden[i] = True
    # means i cannot be in ANY valid interval... no, that's not right either.
    #
    # Better approach: For a given interval [s, e], it's valid iff for every x in [s, e],
    # inv[x] is NOT in [s, e] (unless inv[x] == x, which only happens for 1 and 1008).
    #
    # Let's use a different strategy: 
    # Mark each position i (1..1008) as "bad" if i is a self-inverse (i==1 or i==1008).
    # For pairs: if we include both x and inv[x] where x != inv[x], that's invalid.
    # So for a valid interval, we can include at most one from each inverse pair {x, inv[x]}.
    #
    # Key insight: If we define for each i in 1..1008, the "partner" inv[i], then
    # an interval [s,e] is valid iff for all i in [s,e], inv[i] is not in [s,e].
    #
    # Efficient search: For each start s, extend e as far as possible.
    # e is the max such that for all x in [s, e], inv[x] < s or inv[x] > e.
    # Equivalently, for all x in [s, e], if inv[x] >= s then inv[x] > e.
    #
    # Let's precompute for each x, whether inv[x] < x or inv[x] > x.
    # If inv[x] < x, then x is the "larger" of the pair.
    # If inv[x] > x, then x is the "smaller" of the pair.
    #
    # For a valid interval [s, e]:
    #   For each x in [s, e] with inv[x] < x: we need inv[x] < s (i.e., inv[x] not in [s,e])
    #   For each x in [s, e] with inv[x] > x: we need inv[x] > e (i.e., inv[x] not in [s,e])
    #   For x with inv[x] == x (x=1 or x=1008): x cannot be in [s,e] at all.
    
    # Precompute: for each x, is it self-inverse?
    self_inv = [False] * p
    self_inv[1] = True
    self_inv[p-1] = True
    
    # For each x, store inv[x]
    # We want to find the longest interval [s, e] subset of [1, 1008] such that:
    #   1. No x in [s,e] is self-inverse
    #   2. For all x in [s,e], inv[x] not in [s,e]
    
    # Approach: iterate over all possible starts s, and for each, find max valid e.
    # Use the fact that if [s, e] is valid, then [s, e-1] is also valid (monotonicity).
    # So for each s, we can binary search or just extend greedily.
    
    # Greedy extension: for start s, start with e = s, and extend while valid.
    # To check if adding e+1 is valid:
    #   - e+1 must not be self-inverse
    #   - inv[e+1] must not be in [s, e+1]
    #   - Also, for all existing x in [s, e], we already ensured inv[x] not in [s, e].
    #     Adding e+1: we need inv[x] != e+1 for all x in [s, e], i.e., e+1's inverse not in [s,e].
    #     But inv[e+1] not in [s, e+1] covers this since if inv[e+1] is in [s, e], that's a problem.
    #     Wait, we also need: for each x in [s, e], inv[x] != e+1. This is equivalent to 
    #     e+1 != inv[x] for all x in [s,e], which means inv[e+1] != x for all x in [s,e],
    #     i.e., inv[e+1] not in [s, e]. Combined with inv[e+1] != e+1 (self-inverse check),
    #     we need inv[e+1] not in [s, e+1].
    
    best_len = 0
    best_start = 2  # default safe start
    
    # Precompute a "next_bad" or use direct checking
    # For efficiency, let's precompute for each position i, the constraint it imposes.
    
    # Let's just do the O(n^2) scan with early termination, n=1008 so it's fine.
    for s in range(1, n + 1):
        # Max possible length from s is n - s + 1
        if n - s + 1 <= best_len:
            break
        
        if self_inv[s]:
            continue
        
        # Check if s is valid as a single-element interval
        # s is valid alone if not self-inverse (already checked)
        e = s
        current_len = 1
        
        # Try to extend
        for e_candidate in range(s + 1, n + 1):
            if time.time() > deadline:
                break
            
            if self_inv[e_candidate]:
                break
            
            # Check if inv[e_candidate] is in [s, e_candidate]
            ie = inv[e_candidate]
            if s <= ie <= e_candidate:
                break
            
            # Also need to check: for all x in [s, e_candidate-1], inv[x] != e_candidate
            # This is equivalent to: e_candidate != inv[x] for all x in [s, e_candidate-1]
            # Which is: inv[e_candidate] != x for all x in [s, e_candidate-1]
            # i.e., inv[e_candidate] not in [s, e_candidate-1]
            # We already checked inv[e_candidate] not in [s, e_candidate], which is stronger.
            # So we're good.
            
            e = e_candidate
            current_len = e - s + 1
        
        if current_len > best_len:
            best_len = current_len
            best_start = s
    
    return [best_start, best_len]
