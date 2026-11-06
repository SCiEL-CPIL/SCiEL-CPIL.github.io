import time
import sys
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    N = 256
    
    # Weights as Fractions for exact comparison, but float for speed in initial checks
    weights = [Fraction(0)] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = Fraction(1, i)
    
    # Bitset representation for speed
    # Use Python integers as bitsets. bit i is set if i is in S.
    # i ranges from 1 to 256. So we need bits 1..256.
    # Bit 0 is unused.
    
    def is_valid_addition_bitset(x, mask):
        """
        Check if x can be added to the set represented by mask.
        x is an integer 1..N.
        mask is an integer where bit i is 1 if i is in S.
        """
        # A 3-AP is a, a+d, a+2d with d != 0.
        # If we add x, we must not form a 3-AP with two existing elements.
        # The 3-APs involving x are:
        # 1. x is the first term: x, x+d, x+2d. We need x+d and x+2d in S.
        # 2. x is the middle term: x-d, x, x+d. We need x-d and x+d in S.
        # 3. x is the last term: x-2d, x-d, x. We need x-2d and x-d in S.
        
        # Iterate over possible differences d
        # Max d is roughly N/2 for middle, N/2 for ends.
        # Actually, d can be up to N.
        
        # Case 1: x is first term.
        # x, x+d, x+2d.
        # x+d <= N => d <= N-x
        # x+2d <= N => d <= N-x // 2
        max_d1 = (N - x) // 2
        for d in range(1, max_d1 + 1):
            y = x + d
            z = x + 2*d
            # Check if y and z are in mask
            if (mask >> y) & 1 and (mask >> z) & 1:
                return False
        
        # Case 2: x is middle term.
        # x-d, x, x+d.
        # x-d >= 1 => d <= x-1
        # x+d <= N => d <= N-x
        max_d2 = min(x - 1, N - x)
        for d in range(1, max_d2 + 1):
            y = x - d
            z = x + d
            if (mask >> y) & 1 and (mask >> z) & 1:
                return False
        
        # Case 3: x is last term.
        # x-2d, x-d, x.
        # x-2d >= 1 => d <= (x-1)//2
        max_d3 = (x - 1) // 2
        for d in range(1, max_d3 + 1):
            y = x - d
            z = x - 2*d
            if (mask >> y) & 1 and (mask >> z) & 1:
                return False
        
        return True

    def get_score(mask):
        # Sum weights of set bits
        s = Fraction(0)
        m = mask
        i = 1
        while m:
            if m & 1:
                s += weights[i]
            m >>= 1
            i += 1
        return s

    def mask_to_list(mask):
        return [i for i in range(1, N+1) if (mask >> i) & 1]

    # Greedy Initialization
    # Try to build a good initial set.
    # Strategy: Iterate 1..N, add if valid.
    mask = 0
    for x in range(1, N + 1):
        if is_valid_addition_bitset(x, mask):
            mask |= (1 << x)
    
    best_mask = mask
    best_score = get_score(mask)
    
    # Local Search (Hill Climbing)
    # Moves:
    # 1. Add x if valid.
    # 2. Remove x if it allows adding y with higher weight? (Swap)
    # 3. Swap x for y (remove x, add y) if valid and improves score.
    # 4. 2-Swap: Remove x1, x2, add y1, y2.
    
    # To speed up, we can precompute conflicts or just use the bitset check.
    # Bitset check is O(N) per addition check.
    # N=256. 256 * 256 ops ~ 65k ops per check.
    # 1000 checks ~ 65M ops. Python might be slow.
    # Let's optimize is_valid_addition_bitset.
    
    # Alternative: Precompute for each x, the list of pairs (y, z) that form a 3-AP with x.
    # If y and z are in S, x cannot be added.
    # conflicts[x] = list of (y, z) such that {x, y, z} is a 3-AP.
    # Then is_valid(x, mask) is True iff for all (y, z) in conflicts[x], not (mask has y and mask has z).
    
    conflicts = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x first
        for d in range(1, (N - x) // 2 + 1):
            y = x + d
            z = x + 2*d
            if z <= N:
                conflicts[x].append((y, z))
        # x middle
        for d in range(1, min(x - 1, N - x) + 1):
            y = x - d
            z = x + d
            conflicts[x].append((y, z))
        # x last
        for d in range(1, (x - 1) // 2 + 1):
            y = x - d
            z = x - 2*d
            if z >= 1:
                conflicts[x].append((y, z))
    
    def is_valid_fast(x, mask):
        for y, z in conflicts[x]:
            if (mask >> y) & 1 and (mask >> z) & 1:
                return False
        return True

    # Re-initialize with fast check
    mask = 0
    for x in range(1, N + 1):
        if is_valid_fast(x, mask):
            mask |= (1 << x)
    
    best_mask = mask
    best_score = get_score(mask)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = Fraction(0)
        best_move = None
        
        # Try Additions
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if not (mask >> x) & 1:
                if is_valid_fast(x, mask):
                    gain = weights[x]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('add', x)
        
        # Try Swaps (remove u, add v)
        # v not in mask, u in mask.
        # v must be valid in mask ^ (1<<u).
        # Gain = weights[v] - weights[u].
        
        # Optimize: For each v not in mask, find u in mask such that v is valid in mask without u.
        # v is invalid in mask if there exists (y, z) in conflicts[v] such that y, z in mask.
        # Removing u helps if u is y or u is z for all such conflicting pairs.
        
        for v in range(1, N + 1):
            if time.time() > deadline:
                break
            if (mask >> v) & 1:
                continue
            
            # Find all u in mask that conflict with v
            # Actually, we need to find u such that removing u makes v valid.
            # This means for every (y, z) in conflicts[v] where y, z are in mask,
            # we must have u == y or u == z.
            # So u must be in the intersection of all "blocking" pairs? No.
            # If there are two disjoint blocking pairs (y1, z1) and (y2, z2), removing one u can't fix both unless u is in both (impossible if disjoint).
            # So v can only be added by removing u if all blocking pairs share a common element u in mask.
            
            blocking_pairs = []
            for y, z in conflicts[v]:
                if (mask >> y) & 1 and (mask >> z) & 1:
                    blocking_pairs.append((y, z))
            
            if not blocking_pairs:
                # v is already valid, handled in 'add'
                continue
            
            # Find common elements in all blocking pairs
            # u must be in every pair.
            # Initialize candidate set with elements of first pair
            candidates = set(blocking_pairs[0])
            for y, z in blocking_pairs[1:]:
                candidates &= {y, z}
                if not candidates:
                    break
            
            # For each candidate u, check if removing u makes v valid.
            # It should, by construction, but let's verify or just calculate gain.
            # Actually, if u is in all blocking pairs, removing u breaks all those APs.
            # Are there other APs involving v? No, conflicts[v] contains all.
            # So yes, removing u makes v valid.
            
            for u in candidates:
                if u == v: # u is in mask, v is not, so u != v
                    continue
                gain = weights[v] - weights[u]
                if gain > best_gain:
                    best_gain = gain
                    best_move = ('swap', u, v)
        
        # Try 2-Swaps? (remove u1, u2, add v1, v2)
        # Too expensive for now.
        
        if best_move:
            if best_move[0] == 'add':
                x = best_move[1]
                mask |= (1 << x)
            else:
                u, v = best_move[1], best_move[2]
                mask ^= (1 << u)
                mask |= (1 << v)
            
            best_mask = mask
            best_score = get_score(mask)
            improved = True
        else:
            # No single move improves.
            # Try random restarts? Or 2-swaps?
            # Let's try a few random 2-swaps if time permits.
            # Or just break.
            
            # Simple improvement: try to add pairs of elements that are currently blocked by each other?
            # No, if they are blocked by existing elements, adding them requires removing those.
            
            # Let's try a limited 2-swap search.
            # Find best pair (u1, u2) to remove and (v1, v2) to add.
            # Gain = w[v1] + w[v2] - w[u1] - w[u2].
            # This is O(N^4) roughly. Too slow.
            
            # Break to avoid wasting time.
            break
    
    # Final Polish:
    # Try to add any remaining valid elements (in case we missed something due to order)
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if not (mask >> x) & 1:
            if is_valid_fast(x, mask):
                mask |= (1 << x)
                best_mask = mask
                best_score = get_score(mask)
    
    return mask_to_list(best_mask)

# Test locally? No, just define.
# The function returns a list of integers.

if __name__ == "__main__":
    # Quick test
    s = construct()
    print(s)
    # Check validity
    S = set(s)
    valid = True
    for a in S:
        for b in S:
            if a == b: continue
            c = 2*b - a
            if c in S and c != a and c != b:
                valid = False
    print("Valid:", valid)
    print("Sum:", sum(1.0/x for x in s))
