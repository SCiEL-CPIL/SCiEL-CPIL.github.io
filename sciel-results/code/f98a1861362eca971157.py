"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations

def construct():
    deadline = time.time() + 50  # 50s safety margin out of 60s
    
    N = 256
    # Vertices: integers 2..256, indexed 0..253
    # vertex i corresponds to integer (i+2)
    
    # Precompute: for each pair (b,c) with b != c, the set of a's that divide b*c
    # conflict[a_idx] = bitmask of pairs (b_idx, c_idx) where a divides b*c
    # Actually, let's think differently:
    # For a candidate set S, it's valid iff for all distinct a,b,c in S, a does not divide b*c
    # 
    # For branch-and-bound, I'll use the following:
    # - We build the set incrementally
    # - When adding a new element x, we need to check: for all pairs (b,c) already in the set, x does not divide b*c
    # - Also: for all pairs (x, b) where b is already in the set, no existing a divides x*b
    # - And: for all pairs (b, x), no existing a divides b*x
    # - And: for all existing a, a does not divide b*c for any new pair involving x
    #
    # Simpler: precompute for each a, the set of (b,c) pairs where a | b*c
    # Then when building set S, a is compatible iff no pair from conflict[a] is fully in S
    #
    # For branch and bound, let's use a different approach:
    # Represent the set as a bitmask. For each a, precompute conflict_mask[a] as a set of 
    # "forbidden pair indices". But pairs are 255*254/2 ≈ 32k, too many.
    #
    # Better approach: For each a, precompute which elements b are "dangerous" with a.
    # a is incompatible with {b,c} if a | b*c.
    # 
    # Let's precompute: for each a (2..256), for each b (2..256, b!=a), 
    #   the set of c values such that a | b*c
    # Then a is forbidden if there exist b,c both in the current set with c in that set.
    #
    # Even better for branch-and-bound:
    # Precompute for each a: divisors_info[a] = set of b values where gcd(a,b) > 1 
    #   (these are the b's that could potentially form a bad triple with some c)
    # Actually, a | b*c means: for each prime power p^k || a, we need p^k | b*c.
    #
    # Let me use a simpler but effective approach:
    # Precompute for each a (2..256): bad_pairs[a] = set of frozenset({b,c}) where a|b*c and b!=c and b,c != a
    # Then in branch and bound, when considering adding a to current set S,
    # check if any bad_pair of a is a subset of S.
    #
    # Number of bad pairs per a: for a=2, b*c must be even, so at least one of b,c is even.
    # That's roughly half of all pairs, so ~16k pairs. Too many to store per a.
    #
    # Alternative: For each a, store the set of b values such that there EXISTS c with a|b*c.
    # Actually, for a given a and b, a|b*c iff for each prime p^k || a, p^k | b*c.
    # Let g = gcd(a, b). Then a/g must divide c. So c must be a multiple of a/g.
    # So for given a, b: the forbidden c values are multiples of a/gcd(a,b) that are in 2..256 and != a,b.
    #
    # For branch and bound with current set S:
    # a is compatible with S iff for all b in S (b != a), there is no c in S (c != a, c != b) 
    #   such that a | b*c.
    # Equivalently: for all b in S, (a/gcd(a,b)) does not divide any c in S\{a,b}.
    #
    # This is O(|S|) checks per candidate, each check O(1) if we precompute.
    #
    # Let me precompute: for each a, b: threshold[a][b] = a // gcd(a, b)
    # Then a is incompatible with S if exists b in S such that exists c in S\{a,b} with c % threshold[a][b] == 0.
    #
    # For efficiency, for each a and b, precompute the set of c values (in 2..256) that are multiples of threshold[a][b].
    # Then check if any such c is in S (and c != a, c != b).
    
    # Let's use a bitmask approach for the current set
    # Bit i represents whether (i+2) is in the set
    
    # Precomputation
    n = N - 1  # 255 vertices
    # idx: integer x -> index (x-2)
    # For each a_idx, b_idx: threshold = a // gcd(a,b) where a = a_idx+2, b = b_idx+2
    # multiples_mask[a_idx][b_idx] = bitmask of c_idx where (c_idx+2) % threshold == 0
    
    # This is 255*255 = 65025 entries, each a 255-bit mask. Manageable.
    import math
    
    gcd_table = [[0]*n for _ in range(n)]
    thresh_table = [[0]*n for _ in range(n)]
    mult_mask = [[0]*n for _ in range(n)]
    
    for ai in range(n):
        a = ai + 2
        for bi in range(n):
            b = bi + 2
            g = math.gcd(a, b)
            t = a // g
            thresh_table[ai][bi] = t
            # Build mask of c values that are multiples of t
            mask = 0
            for ci in range(n):
                c = ci + 2
                if c % t == 0:
                    mask |= (1 << ci)
            mult_mask[ai][bi] = mask
    
    # For checking if a_idx can be added to set S (bitmask):
    # For each b_idx in S (b_idx != a_idx):
    #   Check if (mult_mask[a_idx][b_idx] & S & ~(1 << a_idx) & ~(1 << b_idx)) != 0
    # If any such b exists, a_idx cannot be added.
    
    def can_add(a_idx, S):
        """Check if a_idx can be added to set S."""
        S_without_a = S & ~(1 << a_idx)
        for bi in range(n):
            if bi == a_idx:
                continue
            if not (S_without_a & (1 << bi)):
                continue
            # b is in S, check if any c in S\{a,b} has a|b*c
            c_mask = mult_mask[a_idx][bi] & S_without_a & ~(1 << bi)
            if c_mask != 0:
                return False
        return True
    
    # Upper bound: greedy coloring on the "conflict graph" 
    # Two vertices conflict if they cannot both be in a valid set.
    # But the constraint is 3-uniform, not pairwise. 
    # 
    # Simpler upper bound: |remaining| + |current_set|
    # Tighter: use the fact that if a set has many elements, many triples are formed.
    #
    # For a practical upper bound, I'll use: current_size + number of remaining elements
    # that are individually compatible with the current set.
    
    def count_compatible(remaining_mask, S):
        """Count how many elements in remaining_mask can be added to S."""
        count = 0
        m = remaining_mask
        while m:
            # Get lowest set bit
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            if can_add(idx, S):
                count += 1
            m ^= lsb
        return count
    
    # Branch and bound
    best = [0]
    best_set = [0]
    
    # Order vertices by some heuristic - start with elements that are "less constraining"
    # An element a constrains the set if it has many bad pairs.
    # Number of bad pairs for a: for each b, count c's where a|b*c
    # Let's just use a simple ordering: try to include small numbers first? 
    # Or use degree in a derived graph.
    
    # Let's compute for each a: how many pairs (b,c) does it conflict with?
    conflict_count = [0]*n
    for ai in range(n):
        a = ai + 2
        for bi in range(n):
            if bi == ai:
                continue
            b = bi + 2
            g = math.gcd(a, b)
            t = a // g
            # Count c's (c != a, c != b) where c % t == 0
            cnt = 0
            for ci in range(n):
                if ci == ai or ci == bi:
                    continue
                c = ci + 2
                if c % t == 0:
                    cnt += 1
            conflict_count[ai] += cnt
    
    # Order: try elements with fewer conflicts first (more likely to be included)
    order = sorted(range(n), key=lambda i: conflict_count[i])
    
    # Branch and bound with the order
    # We process vertices in order, deciding include/exclude for each
    
    import sys
    sys.setrecursionlimit(10000)
    
    def bb(pos, S, current_size):
        if time.time() > deadline:
            return
        
        if current_size > best[0]:
            best[0] = current_size
            best_set[0] = S
        
        if pos >= n:
            return
        
        # Upper bound
        if current_size + (n - pos) <= best[0]:
            return
        
        # Try including order[pos]
        ai = order[pos]
        if can_add(ai, S):
            bb(pos + 1, S | (1 << ai), current_size + 1)
        
        # Try excluding order[pos]
        bb(pos + 1, S, current_size)
    
    # Initial: try to build a good starting set
    # Greedy: add elements in order of increasing conflict_count
    S = 0
    for ai in order:
        if time.time() > deadline:
            break
        if can_add(ai, S):
            S |= (1 << ai)
    
    best[0] = bin(S).count('1')
    best_set[0] = S
    
    # Now run branch and bound from the full problem
    # But this might be too slow. Let's use a smarter approach.
    
    # Actually, let's use a local search approach combined with the greedy:
    # 1. Start with greedy solution
    # 2. Try to add excluded elements
    # 3. Try 2-for-1 swaps: remove one, add two
    
    # First, let's get a good initial solution
    S = 0
    for ai in order:
        if can_add(ai, S):
            S |= (1 << ai)
    
    best_size = bin(S).count('1')
    best_S = S
    
    # Local search: try to add any excluded element
    improved = True
    while improved:
        improved = False
        for ai in order:
            if time.time() > deadline:
                break
            if not (S & (1 << ai)):
                if can_add(ai, S):
                    S |= (1 << ai)
                    best_size += 1
                    best_S = S
                    improved = True
    
    # Try 2-for-1 swaps: remove one element, add two
    for remove_idx in order:
        if time.time() > deadline:
            break
        if not (S & (1 << remove_idx)):
            continue
        S2 = S & ~(1 << remove_idx)
        # Try adding pairs
        for add1 in order:
            if time.time() > deadline:
                break
            if S2 & (1 << add1):
                continue
            if add1 == remove_idx:
                continue
            if not can_add(add1, S2):
                continue
            S3 = S2 | (1 << add1)
            for add2 in order:
                if time.time() > deadline:
                    break
                if S3 & (1 << add2):
                    continue
                if add2 == remove_idx or add2 == add1:
                    continue
                if not can_add(add2, S3):
                    continue
                # Valid swap
                new_S = S3 | (1 << add2)
                if bin(new_S).count('1') > best_size:
                    S = new_S
                    best_size = bin(S).count('1')
                    best_S = S
                    improved = True
                    break
            if improved:
                break
        if improved:
            break
    
    # Convert best_S back to list of integers
    result = []
    for i in range(n):
        if best_S & (1 << i):
            result.append(i + 2)
    
    return result

# EVOLVE-BLOCK-END
