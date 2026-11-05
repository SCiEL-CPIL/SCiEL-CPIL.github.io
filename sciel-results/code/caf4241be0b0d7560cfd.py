"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N = 512

    # Precompute pairwise LCM compatibility
    # compatible[i][j] = True if lcm(i+1, j+1) <= N
    # We'll use a bitset representation for speed
    # compatible_mask[i] is a bitset where bit j is set if i and j+1 are compatible
    
    # Actually, let's use a 2D boolean array for clarity, but we'll optimize with bitsets
    # For 512 elements, a bitset is 512 bits = 64 bytes per row, total ~32KB - very manageable
    
    # Build compatibility as list of bitmasks
    # compat_mask[i] is an integer where bit j is set if gcd-compatible (lcm <= N)
    compat_mask = [0] * (N + 1)  # 1-indexed, compat_mask[i] for number i
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if (i * j) // g <= N:
                mask |= (1 << j)
        compat_mask[i] = mask

    # Now we want to find a maximum clique in this compatibility graph
    # But maximum clique is NP-hard. However, the structure here is special.
    # 
    # Key insight: The optimal set is a down-set in the divisibility poset intersected 
    # with the constraint that lcm of any two elements <= N.
    #
    # Actually, let's think differently. If we select a set S, then for any a, b in S,
    # lcm(a,b) <= N. This means that the LCM of ALL elements in S must be <= N?
    # No! lcm(a,b) <= N for every pair, but lcm of all elements could be larger.
    # Example: a=2, b=3, c=5. lcm(2,3)=6, lcm(2,5)=10, lcm(3,5)=15, all <= 512.
    # But lcm(2,3,5)=30 <= 512 too. Actually if all pairwise lcms are <= N, 
    # then the lcm of all elements divides the product of pairwise lcms... 
    # Not exactly. Let's think: if lcm(a,b) <= N for all pairs, does that imply
    # lcm of all elements <= N? 
    # Counterexample: a=6, b=10, c=15. lcm(6,10)=30, lcm(6,15)=30, lcm(10,15)=30. All <= 512.
    # lcm(6,10,15) = 30. Still <= 512.
    # Another: a=4, b=6, c=9. lcm(4,6)=12, lcm(4,9)=36, lcm(6,9)=18. All <= 512.
    # lcm(4,6,9)=36. Still <= 512.
    # 
    # Actually, I think if all pairwise lcms are <= N, then the lcm of all elements is also <= N.
    # Proof sketch: Let L = lcm(a1, ..., ak). For any prime p, let e_p be the exponent of p in L.
    # Then some a_i has p^e_p dividing it. For any other a_j, lcm(a_i, a_j) must have p^e_p 
    # (since a_i has p^e_p), so p^e_p divides lcm(a_i, a_j) <= N. Thus p^e_p <= N for all p,
    # and since these are for different primes, L = product p^e_p... wait, that's not right.
    # L is the product of p^e_p over all primes, and each p^e_p <= N, but the product could exceed N.
    #
    # Counterexample: N=100. a=4, b=9, c=25. lcm(4,9)=36<=100, lcm(4,25)=100<=100, lcm(9,25)=225>100.
    # So this fails. Let me try: a=4, b=9. lcm=36<=100. Add c=25: lcm(9,25)=225>100. Not allowed.
    # 
    # Try: N=100. a=6, b=10, c=15. lcm(6,10)=30, lcm(6,15)=30, lcm(10,15)=30. All <=100.
    # lcm(6,10,15)=30<=100.
    # 
    # Try to find a counterexample where all pairwise lcms <= N but total lcm > N.
    # We need primes p1, p2, p3 such that p1^a * p2^b <= N, p1^a * p3^c <= N, p2^b * p3^c <= N,
    # but p1^a * p2^b * p3^c > N.
    # Let p1=2, p2=3, p3=5. N=100.
    # 2^a * 3^b <= 100, 2^a * 5^c <= 100, 3^b * 5^c <= 100.
    # Maximize 2^a * 3^b * 5^c.
    # a=2,b=2,c=1: 4*9=36, 4*5=20, 9*5=45. All <=100. Total=180>100. 
    # So a=4, b=9, c=5. lcm(4,9)=36, lcm(4,5)=20, lcm(9,5)=45. All <=100.
    # lcm(4,9,5)=180 > 100. So the total lcm can exceed N!
    #
    # So the constraint is truly pairwise, not global. This makes it harder.
    #
    # Let's use a greedy approach with local search, leveraging the bitmask compatibility.

    # Greedy: start with empty set, repeatedly add the element that is compatible with all current elements
    # and has the most connections (or some heuristic)
    
    def time_check():
        return time.time() - start_time > DEADLINE

    def greedy_start(seed_elem):
        """Build a set starting from seed_elem using greedy."""
        S = {seed_elem}
        S_mask = 0
        for x in S:
            S_mask |= (1 << x)
        
        # Find all elements compatible with all in S
        # An element j is compatible with all in S if for all x in S, compat_mask[x] has bit j set
        # This is equivalent to: j is in the intersection of compat_mask[x] for all x in S
        # We can compute this as: start with all bits set, then AND with each compat_mask[x]
        
        # Actually, let's just check each candidate
        candidates = []
        for j in range(1, N + 1):
            if j in S:
                continue
            compatible_with_all = True
            for x in S:
                if not (compat_mask[x] & (1 << j)):
                    compatible_with_all = False
                    break
            if compatible_with_all:
                candidates.append(j)
        
        # Greedily add candidates in order of decreasing "degree" within the candidate set
        # Or just add all of them? No, we need to check pairwise among candidates too.
        # Actually, if j is compatible with all in S, we still need j to be compatible with other candidates we add.
        # So we should add them one at a time.
        
        # Sort candidates by some heuristic: prefer those with smaller values? Or larger?
        # Let's try: prefer elements that are divisors of many other elements (hubs)
        
        while candidates and not time_check():
            # Pick the candidate with the most connections to other candidates
            best = None
            best_score = -1
            for j in candidates:
                score = 0
                for k in candidates:
                    if j != k and (compat_mask[j] & (1 << k)):
                        score += 1
                if score > best_score:
                    best_score = score
                    best = j
            
            if best is None:
                break
            
            S.add(best)
            candidates.remove(best)
        
        return S

    def local_search(S):
        """Try 1-swap and 2-add improvements."""
        S = set(S)
        improved = True
        while improved and not time_check():
            improved = False
            
            # 1-swap: try removing one element and adding another
            for a in list(S):
                S_without_a = S - {a}
                # Find best element to add
                for b in range(1, N + 1):
                    if b in S_without_a:
                        continue
                    # Check if b is compatible with all in S_without_a
                    compatible = True
                    for x in S_without_a:
                        if not (compat_mask[x] & (1 << b)):
                            compatible = False
                            break
                    if compatible:
                        new_S = S_without_a | {b}
                        # This is a valid swap (same size, or we could also just add)
                        # For now, just swap
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
            
            # 2-add: try adding two elements at once
            if not improved:
                # Find all elements compatible with all in S
                addable = []
                for b in range(1, N + 1):
                    if b in S:
                        continue
                    compatible = True
                    for x in S:
                        if not (compat_mask[x] & (1 << b)):
                            compatible = False
                            break
                    if compatible:
                        addable.append(b)
                
                # Try adding pairs from addable
                for i in range(len(addable)):
                    for j in range(i + 1, len(addable)):
                        b1, b2 = addable[i], addable[j]
                        if compat_mask[b1] & (1 << b2):
                            S = S | {b1, b2}
                            improved = True
                            break
                    if improved:
                        break
        
        return S

    # Multi-start: try different seeds
    best_set = set()
    
    # Seed with single elements, prioritizing those with high degree
    degrees = []
    for i in range(1, N + 1):
        deg = bin(compat_mask[i]).count('1') - 1  # exclude self
        degrees.append((deg, i))
    degrees.sort(reverse=True)
    
    # Try top seeds
    for deg, seed in degrees[:50]:
        if time_check():
            break
        S = greedy_start(seed)
        S = local_search(S)
        if len(S) > len(best_set):
            best_set = S
    
    # Also try the simple seed: all numbers up to sqrt(512)
    simple_seed = set(range(1, isqrt(N) + 1))
    if len(simple_seed) > len(best_set):
        best_set = simple_seed

    # Final local search on the best set
    if not time_check():
        best_set = local_search(best_set)

    return list(best_set)
# EVOLVE-BLOCK-END
