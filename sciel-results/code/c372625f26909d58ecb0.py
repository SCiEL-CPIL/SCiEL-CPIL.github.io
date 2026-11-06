import time
import random
from math import gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 128
    
    def _lcm(a, b):
        return a // gcd(a, b) * b

    # 1. Precompute LCMs and Bad Triples
    # lcm_val[a][b]
    # We need to identify all bad triples (a,b,c) with a<b<c
    # and build:
    # - pairwise_conflict[a][b] = True if exists c such that (a,b,c) is bad
    # - bad_pairs_for[x] = list of (u,v) such that (x,u,v) is bad (x distinct from u,v)
    
    bad_pairs_for = [set() for _ in range(n + 1)]
    adj_pairwise = [0] * (n + 1) # Bitmask for pairwise conflicts
    
    # Optimization: Group by LCM value
    # If lcm(a,b) = L, then for c to form a bad triple, lcm(a,c)=L and lcm(b,c)=L.
    # This implies a|L, b|L, c|L.
    # And lcm(a,c)=L means a*c/gcd(a,c)=L.
    
    # Faster enumeration:
    # Iterate L from 1 to some max? No, LCM can be large.
    # But a,b,c <= 128.
    # Max LCM of two numbers <= 128 is 128*127 = 16256.
    
    # Let's just iterate triples but optimize the check.
    # O(N^3) = 128^3 = 2M. This is fine in Python if optimized or in C.
    # 2M iterations in Python might take ~1-2 seconds. Acceptable.
    
    # To speed up:
    # For each pair (a,b), calculate L = lcm(a,b).
    # Find all c > b such that lcm(a,c)==L and lcm(b,c)==L.
    
    # Precompute divisors?
    # For a fixed L, the candidates c must divide L.
    
    lcm_to_divs = defaultdict(list)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            L = _lcm(i, j)
            lcm_to_divs[L].append((i, j))
            
    # For each L, we have a list of pairs (i,j) with lcm(i,j)=L.
    # Any pair (i,j) from this list is a "candidate pair" for L.
    # For a triple (i,j,k) to be bad with common LCM L:
    # 1. lcm(i,j)=L
    # 2. lcm(i,k)=L
    # 3. lcm(j,k)=L
    # This implies i,j,k are all divisors of L.
    # And for any pair, their LCM is L.
    
    # Let's process each L.
    # For a given L, let D_L be the set of divisors of L that are <= 128.
    # We only care about pairs (u,v) in D_L such that lcm(u,v)=L.
    # Let P_L be the set of such pairs.
    # If |P_L| >= 3, we need to check triples.
    # Actually, if we have 3 elements u,v,w in D_L, and all pairwise lcms are L, it's a bad triple.
    
    # This approach might be faster if we group by L.
    
    # Let's stick to the O(N^3) loop but with early breaks?
    # Or use the L-grouping.
    
    # L-grouping implementation:
    # 1. Build list of divisors for each L? No, L is up to ~16000.
    # 2. Just iterate a, b.
    
    # Let's use the direct O(N^3) but with bitmasks for speed?
    # Actually, 2M iterations is fast enough.
    
    # Let's refine the O(N^3) loop.
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            # c must be > b
            # Also c must divide ab? Yes, lcm(a,c)=ab implies c|ab.
            # So we can iterate only divisors of ab that are > b.
            # This reduces the inner loop significantly.
            
            # Get divisors of ab that are > b
            # Since n=128, we can just check all c > b.
            # But checking c|ab is a filter.
            for c in range(b + 1, n + 1):
                if ab % c != 0:
                    continue
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    # Bad triple (a, b, c)
                    bad_pairs_for[a].add((b, c))
                    bad_pairs_for[b].add((a, c))
                    bad_pairs_for[c].add((a, b))
                    
                    # Update pairwise conflict graph
                    adj_pairwise[a] |= 1 << (b - 1)
                    adj_pairwise[a] |= 1 << (c - 1)
                    adj_pairwise[b] |= 1 << (a - 1)
                    adj_pairwise[b] |= 1 << (c - 1)
                    adj_pairwise[c] |= 1 << (a - 1)
                    adj_pairwise[c] |= 1 << (b - 1)

    # Convert bad_pairs_for to lists for faster iteration
    bad_pairs_list = [list(s) for s in bad_pairs_for]

    def can_add(x, S_mask):
        """
        Check if x can be added to set represented by S_mask.
        x is 1-indexed. S_mask is a bitmask where bit i-1 corresponds to element i.
        """
        for u, v in bad_pairs_list[x]:
            # Check if u and v are both in S_mask
            if (S_mask >> (u - 1)) & 1 and (S_mask >> (v - 1)) & 1:
                return False
        return True

    # 2. Greedy Seed Generation
    def greedy_seed(order):
        S_mask = 0
        for x in order:
            if can_add(x, S_mask):
                S_mask |= 1 << (x - 1)
        return S_mask

    # Generate multiple seeds
    seeds = []
    
    # Seed 1: Natural order
    seeds.append(greedy_seed(range(1, n + 1)))
    
    # Seed 2: Reverse order
    seeds.append(greedy_seed(range(n, 0, -1)))
    
    # Seed 3: Sorted by degree in pairwise graph (ascending)
    degrees = [bin(adj_pairwise[i]).count('1') for i in range(n + 1)]
    order_deg = sorted(range(1, n + 1), key=lambda x: degrees[x])
    seeds.append(greedy_seed(order_deg))
    
    # Seed 4: Random shuffle
    for _ in range(10):
        order_rand = list(range(1, n + 1))
        random.shuffle(order_rand)
        seeds.append(greedy_seed(order_rand))

    best_mask = 0
    best_size = 0
    
    # 3. Local Search
    def local_search(S_mask):
        S = S_mask
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 1. Try to add any single element
            for x in range(1, n + 1):
                if not (S >> (x - 1)) & 1:
                    if can_add(x, S):
                        S |= 1 << (x - 1)
                        improved = True
                        break
            
            if not improved:
                # 2. Try 1-for-2 swaps
                # For each x not in S, find blockers.
                # If we remove one blocker, can we add x and another y?
                found_swap = False
                for x in range(1, n + 1):
                    if (S >> (x - 1)) & 1:
                        continue
                    
                    # Find all pairs (u,v) in S that block x
                    blockers = []
                    for u, v in bad_pairs_list[x]:
                        if (S >> (u - 1)) & 1 and (S >> (v - 1)) & 1:
                            blockers.append((u, v))
                    
                    if not blockers:
                        # Should have been caught by 'add any single element'
                        continue

                    # Try removing one element from each blocker pair
                    # To limit search, try removing u from first blocker, then v
                    # We want to find a removal that allows adding x AND another y.
                    
                    # Heuristic: Try removing the element with highest degree?
                    # Or just try all u,v in blockers?
                    # Number of blockers is small.
                    
                    for u, v in blockers:
                        # Try removing u
                        S_trial = S & ~(1 << (u - 1))
                        # Check if x can be added to S_trial
                        if can_add(x, S_trial):
                            S_trial |= 1 << (x - 1)
                            # Now try to add another y
                            for y in range(1, n + 1):
                                if (S_trial >> (y - 1)) & 1:
                                    continue
                                if can_add(y, S_trial):
                                    # Success: removed u, added x, added y. Net gain +1
                                    S = S_trial | (1 << (y - 1))
                                    improved = True
                                    found_swap = True
                                    break
                        if found_swap:
                            break
                        # Try removing v
                        if not found_swap:
                            S_trial = S & ~(1 << (v - 1))
                            if can_add(x, S_trial):
                                S_trial |= 1 << (x - 1)
                                for y in range(1, n + 1):
                                    if (S_trial >> (y - 1)) & 1:
                                        continue
                                    if can_add(y, S_trial):
                                        S = S_trial | (1 << (y - 1))
                                        improved = True
                                        found_swap = True
                                        break
                        if found_swap:
                            break
                    if found_swap:
                        break
                
                if found_swap:
                    continue

                # 3. Try 2-for-3 swaps (if 1-for-2 fails)
                # This is more expensive. Only do if we have time.
                if time.time() < deadline - 10:
                    found_swap = False
                    # Iterate over pairs in S
                    s_list = [i for i in range(1, n + 1) if (S >> (i - 1)) & 1]
                    for i in range(len(s_list)):
                        if time.time() > deadline - 10: break
                        for j in range(i + 1, len(s_list)):
                            u = s_list[i]
                            v = s_list[j]
                            S_trial = S & ~(1 << (u - 1)) & ~(1 << (v - 1))
                            
                            # Find candidates to add
                            candidates = []
                            for x in range(1, n + 1):
                                if (S_trial >> (x - 1)) & 1:
                                    continue
                                if can_add(x, S_trial):
                                    candidates.append(x)
                            
                            if len(candidates) >= 2:
                                # Try to add 2
                                added = 0
                                for x in candidates:
                                    if can_add(x, S_trial):
                                        S_trial |= 1 << (x - 1)
                                        added += 1
                                        if added == 2:
                                            break
                                if added == 2:
                                    S = S_trial
                                    improved = True
                                    found_swap = True
                                    break
                        if found_swap:
                            break
                    if found_swap:
                        continue

        return S

    # Run local search on all seeds
    for seed in seeds:
        if time.time() > deadline - 5:
            break
        improved_mask = local_search(seed)
        size = bin(improved_mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = improved_mask

    # Final polish on best_mask
    best_mask = local_search(best_mask)

    # Convert mask to list
    result = []
    for i in range(1, n + 1):
        if (best_mask >> (i - 1)) & 1:
            result.append(i)
    
    # Ensure non-empty
    if not result:
        return [1]
        
    return result

# Test locally? No, just return the function.
# The function name must be construct.

if __name__ == "__main__":
    res = construct()
    print(len(res))
