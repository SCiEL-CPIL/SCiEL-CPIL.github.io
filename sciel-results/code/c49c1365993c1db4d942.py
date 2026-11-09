<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 128}

def construct():
    return list(range(P['n']//2+1,P['n']+1))
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55.0
    n = 128
    # Precompute conflict masks using bit-parallel approach
    # For each pair (i,j) with i<j, compute mask of k values that would form forbidden triple
    # Forbidden: distinct a,b,c in S such that a+b, a+c, b+c all in S
    
    # Strategy: Use numpy bit operations for fast validity
    # Represent set as a 128-bit integer (or numpy uint64 pair)
    
    # Precompute: for each pair (a,b), the set of c values such that if a,b,c in S,
    # then a+b, a+c, b+c must all be in S for it to be forbidden.
    # Actually, let's think of it as: given current set S, can we add element x?
    # Adding x is forbidden if there exist distinct a,b in S\{x} such that:
    #   x+a in S, x+b in S, and a+b in S
    # OR there exist distinct a,b in S\{x} such that:
    #   a+b, a+x, b+x all in S (but x is being added, so we need a+b in S and a+x in S and b+x in S)
    
    # Let me reframe: S is valid iff for no distinct a,b,c in S do we have a+b, a+c, b+c all in S.
    
    # For local search, use swaps: remove one element, add another.
    
    # Start with all of {65,...,128} (size 64)
    # Try to add elements from {1,...,64} and remove some from {65,...,128} if needed
    
    # Bit-parallel representation: use Python int as bitmask (128 bits)
    # S_mask: bit i is 1 if (i+1) is in S (0-indexed)
    
    def mask_to_set(mask):
        return [i+1 for i in range(128) if (mask >> i) & 1]
    
    def set_to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x-1))
        return m
    
    # Precompute pair sum masks
    # For each value v (1..128), precompute which pairs (a,b) sum to v
    # pair_sum_mask[v] = set of (a,b) with a<b and a+b=v
    
    # For checking if adding element x to set S creates a violation:
    # We need to check if there exist distinct a,b in S (a != b, a != x, b != x) such that:
    #   x + a in S, x + b in S, and a + b in S
    # This means: a and b are in S, a != b, a != x, b != x,
    #   (x+a) in S, (x+b) in S, (a+b) in S
    
    # Let's precompute for each x: the list of (a,b) pairs where a+b, a+x, b+x could all be in S
    # i.e., a < b, a != x, b != x, a+x <= 128, b+x <= 128, a+b <= 128
    
    # For each x (1..128), precompute list of candidate (a,b) pairs
    candidate_pairs = [None] * 129
    for x in range(1, 129):
        pairs = []
        for a in range(1, 129):
            if a == x:
                continue
            xa = x + a
            if xa > 128:
                continue
            for b in range(a+1, 129):
                if b == x:
                    continue
                xb = x + b
                if xb > 128:
                    continue
                ab = a + b
                if ab > 128:
                    continue
                # Check: if a,b,x in S and xa,xb,ab in S, violation
                # We need a != b (guaranteed by b > a), all distinct from x (checked)
                pairs.append((a, b, xa, xb, ab))
        candidate_pairs[x] = pairs
    
    # Also need to check triples entirely in S (not involving the new element)
    # But if S was valid before, adding x only creates violations involving x
    
    # For removal: removing element x from S can only help (fewer elements = fewer violations)
    # So removal is always safe
    
    # Greedy construction with bit-parallel checks
    def can_add(x, S_mask):
        """Check if x can be added to S (represented as bitmask)."""
        if (S_mask >> (x-1)) & 1:
            return False  # already in set
        # Check all candidate pairs for x
        for (a, b, xa, xb, ab) in candidate_pairs[x]:
            # Need a, b, xa, xb, ab all in S_mask
            # a, b must be in S (and != x, already ensured)
            if ((S_mask >> (a-1)) & 1) and ((S_mask >> (b-1)) & 1) and \
               ((S_mask >> (xa-1)) & 1) and ((S_mask >> (xb-1)) & 1) and \
               ((S_mask >> (ab-1)) & 1):
                return False
        return True
    
    # Fast version using numpy for bulk operations
    def can_add_fast(x, S_arr):
        """S_arr is a numpy boolean array of length 128 (0-indexed: S_arr[i] means (i+1) in S)"""
        if S_arr[x-1]:
            return False
        for (a, b, xa, xb, ab) in candidate_pairs[x]:
            if S_arr[a-1] and S_arr[b-1] and S_arr[xa-1] and S_arr[xb-1] and S_arr[ab-1]:
                return False
        return True
    
    # Start with upper half
    best_mask = 0
    for i in range(64, 128):
        best_mask |= (1 << i)
    
    best_size = 64
    best_arr = np.zeros(128, dtype=bool)
    for i in range(64, 128):
        best_arr[i] = True
    
    # Try greedy addition of lower elements
    def greedy_add(S_arr, S_mask, start_order=None):
        """Try to add elements greedily."""
        if start_order is None:
            start_order = list(range(1, 65))
            random.shuffle(start_order)
        added = 0
        for x in start_order:
            if S_arr[x-1]:
                continue
            if can_add_fast(x, S_arr):
                S_arr[x-1] = True
                S_mask |= (1 << (x-1))
                added += 1
        return added, S_mask
    
    # Local search: try swaps
    def local_search(S_arr, S_mask, time_limit=50.0):
        """Perform local search with swaps."""
        best = S_mask
        best_size = bin(S_mask).count('1')
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < time_limit and no_improve < max_no_improve:
            improved = False
            # Try adding an element
            candidates = [x for x in range(1, 129) if not S_arr[x-1]]
            random.shuffle(candidates)
            for x in candidates:
                if can_add_fast(x, S_arr):
                    S_arr[x-1] = True
                    S_mask |= (1 << (x-1))
                    best_size += 1
                    no_improve = 0
                    improved = True
                    break
            
            if not improved:
                # Try swaps: remove one, add one
                in_set = [x for x in range(1, 129) if S_arr[x-1]]
                not_in_set = [x for x in range(1, 129) if not S_arr[x-1]]
                random.shuffle(in_set)
                random.shuffle(not_in_set)
                
                found_swap = False
                for rem in in_set:
                    if found_swap:
                        break
                    # Temporarily remove rem
                    S_arr[rem-1] = False
                    S_mask &= ~(1 << (rem-1))
                    
                    for add in not_in_set:
                        if S_arr[add-1]:
                            continue
                        if can_add_fast(add, S_arr):
                            S_arr[add-1] = True
                            S_mask |= (1 << (add-1))
                            best_size += 1  # net zero change from swap
                            # Actually swap keeps size same, but we want to try
                            # Let's just track size properly
                            no_improve = 0
                            improved = True
                            found_swap = True
                            break
                    
                    if not found_swap:
                        # Restore rem
                        S_arr[rem-1] = True
                        S_mask |= (1 << (rem-1))
                
                if not improved:
                    no_improve += 1
            
            # Update best
            cur_size = bin(S_mask).count('1')
            if cur_size > best_size:
                best = S_mask
                best_size = cur_size
        
        return S_mask, best
    
    # Run multiple restarts
    best_result = best_mask
    best_result_size = 64
    
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Randomized initial set
        if restart == 0:
            # Start with upper half
            S_arr = np.zeros(128, dtype=bool)
            for i in range(64, 128):
                S_arr[i] = True
            S_mask = 0
            for i in range(64, 128):
                S_mask |= (1 << i)
        else:
            # Random subset biased toward upper half
            S_arr = np.zeros(128, dtype=bool)
            S_mask = 0
            # Include most of upper half
            for i in range(64, 128):
                if random.random() < 0.9:
                    S_arr[i] = True
                    S_mask |= (1 << i)
            # Include some of lower half
            for i in range(0, 64):
                if random.random() < 0.3:
                    S_arr[i] = True
                    S_mask |= (1 << i)
        
        # Greedy add
        order = list(range(1, 129))
        random.shuffle(order)
        for x in order:
            if not S_arr[x-1] and can_add_fast(x, S_arr):
                S_arr[x-1] = True
                S_mask |= (1 << (x-1))
        
        # Local search
        S_mask, best = local_search(S_arr, S_mask, time_limit=deadline - 5.0)
        
        cur_size = bin(S_mask).count('1')
        if cur_size > best_result_size:
            best_result = S_mask
            best_result_size = cur_size
    
    # Convert best result to list
    result = [i+1 for i in range(128) if (best_result >> i) & 1]
    return result

# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize this approach has issues. The local search logic is flawed - swaps don't increase size. Let me rethink.

The key insight is: we want to maximize |S|. The upper half {65,...,128} gives 64. Can we do better?

Let me think about whether we can include some lower elements. If we include a lower element x ≤ 64, we need to ensure no forbidden triple is created.

Actually, let me reconsider the problem. A forbidden triple is three DISTINCT a,b,c in S such that a+b, a+c, b+c are ALL in S.

If all three of a,b,c are > 64, then a+b > 128, so a+b ∉ S. Safe.
If two are > 64 and one ≤ 64: say a > 64, b > 64, c ≤ 64. Then a+b > 128, so a+b ∉ S. Safe.
If one is > 64 and two ≤ 64: say a > 64, b ≤ 64, c ≤ 64. Then a+b could be ≤ 128 (if b is small enough), a+c could be ≤ 128, and b+c ≤ 128. This is dangerous.
If all three ≤ 64: all sums ≤ 128. Dangerous.

So the constraint is really about the lower elements and their interaction with upper elements.

Let me try a cleaner approach: use the fact that if we keep the upper half, we just need to check that for any a in upper, b,c in lower, we don't have a+b, a+c, b+c all in S.

