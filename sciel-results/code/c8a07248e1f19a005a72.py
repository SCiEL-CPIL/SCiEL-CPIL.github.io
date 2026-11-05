# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

N = 192
K = 3
MAX_CARD = 48

def _precompute_conflicts():
    """Precompute for each element a the bitmask of 3-subsets of [1..N] whose sum is divisible by a.
    Returns: for each a in 1..N, a list of bitmasks (each a 192-bit int) representing 3-subsets.
    Also precompute pair_bitmask[i][j] = bitmask of all k such that {i,j,k} is a valid triple.
    """
    # pair_mask[a][i][j] = bitmask of k values such that i+j+k is divisible by a
    # But we need: for a given set S and element a in S, count triples from S\{a} with sum % a == 0
    
    # Better approach: for each a, precompute all pairs (i,j) and the set of k such that (i+j+k)%a==0
    # Then for a set S, for element a, conflicts = sum over pairs (i,j) in S\{a} of |{k in S\{a,i,j} : (i+j+k)%a==0}|
    
    # Precompute: for each a (1..N), for each pair (i,j) with i<j, the list of k values where (i+j+k)%a==0
    # This is too large. Instead, let's use a different approach.
    
    # For each a, and for each pair (i,j), compute the residue r = (i+j) % a
    # Then k must satisfy k % a == (-r) % a
    # So for a set S, the number of k in S\{a,i,j} with k%a == (-r)%a gives the count.
    
    # Precompute: for each a, residue_counts can be computed on the fly from the set.
    # But for speed, let's precompute pair_residues[a][i][j] = (i+j) % a
    # And then during evaluation, for each a in S, for each pair (i,j) in S\{a}, 
    # count how many k in S\{a,i,j} have k%a == (-(i+j))%a
    
    # Actually, let's use a smarter counting:
    # For a fixed a and set T = S\{a}, we want to count triples (i,j,k) from T with (i+j+k)%a==0.
    # This equals sum over residues r of C(count_r, 3) where count_r = |{x in T : x%a == r}|
    # PLUS cross terms... no, that's for all triples with the same residue.
    # Actually: number of triples with sum ≡ 0 mod a = 
    # sum over (r1,r2,r3) with r1+r2+r3 ≡ 0 mod a of (count_{r1}*count_{r2}*count_{r3} adjusted for distinct)
    
    # This is complex. Let me use a different strategy:
    # Precompute for each a a 2D array: conflict_pair[a][i][j] = bitmask of k such that (i+j+k)%a==0
    # Then for set S and element a: 
    #   For each pair (i,j) in S\{a}: count bits in conflict_pair[a][i][j] & S_mask & ~(1<<a) & ~(1<<i) & ~(1<<j)
    
    # This is O(|S|^2) per element, O(|S|^3) total. With |S| up to 48, that's ~110K ops per evaluation.
    # For simulated annealing with ~100K steps, that's 10^10 ops - too slow in Python.
    
    # Alternative: use numpy for vectorized operations.
    # Or: precompute for each a, a matrix M_a where M_a[i][j] = bitmask of k with (i+j+k)%a==0
    # Then for a set S (as a bitmask), for element a:
    #   conflicts = 0
    #   For each pair (i,j) in S\{a}: conflicts += popcount(M_a[i][j] & S & ~(1<<a) & ~(1<<i) & ~(1<<j))
    
    # Still O(|S|^2) per element.
    
    # Let me think of a faster approach. 
    # Key insight: For a fixed a, the condition depends only on residues mod a.
    # Let T = S\{a}. Let c[r] = |{x in T : x%a == r}| for r in 0..a-1.
    # Number of triples (i,j,k) from T with (i+j+k)%a==0:
    # We need to count ordered/unordered triples of distinct elements from T with residue sum ≡ 0 mod a.
    
    # For unordered triples of distinct elements:
    # Case 1: all three have the same residue r, need 3r ≡ 0 mod a. Count: C(c[r], 3)
    # Case 2: two have residue r, one has residue s (r≠s), need 2r+s ≡ 0 mod a. Count: C(c[r],2)*c[s]
    # Case 3: three distinct residues r,s,t, need r+s+t ≡ 0 mod a. Count: c[r]*c[s]*c[t]
    
    # But we need to be careful about overcounting. Actually for unordered triples:
    # total = sum over all multisets of residues {r1,r2,r3} with r1<=r2<=r3 and r1+r2+r3≡0 mod a of:
    #   if r1=r2=r3: C(c[r1],3)
    #   if r1=r2<r3: C(c[r1],2)*c[r3]
    #   if r1<r2=r3: c[r1]*C(c[r2],2)
    #   if r1<r2<r3: c[r1]*c[r2]*c[r3]
    
    # This is O(a^3) per element a, which is too slow for large a.
    
    # Alternative: just do the O(|T|^2) pair enumeration but with numpy vectorization.
    
    # Let me try a hybrid: precompute residue-based counts and use the formula above.
    # For a up to 192, a^3 is up to 7 million - too slow.
    
    # Let me just use the pair enumeration with early termination and numpy.
    
    # Actually, let me reconsider. The key optimization is:
    # For a given set S (bitmask), and element a in S:
    #   T = S without a
    #   We want to count triples from T with sum ≡ 0 mod a.
    
    # Precompute: for each a, for each i in 1..N, the residue i%a.
    # Then for set T, compute residue histogram in O(|T|).
    # Then count valid triples in O(a^2) using the formula.
    
    # Total per evaluation: sum over a in S of (|S| + a^2) ≈ |S|^2 + sum(a^2 for a in S)
    # For |S|=48 and a up to 192, sum(a^2) ≈ 48 * avg(a^2) ≈ 48 * 10000 = 480K. Too slow.
    
    # Let me just use the direct O(|S|^3) approach but with numpy for the inner loop.
    # Or accept that each evaluation is ~O(48^3) = 110K ops and limit SA steps.
    
    # Given 60s budget, if each eval takes ~0.1ms, we can do ~600K evals.
    # Let me measure and optimize.
    
    pass

def _conflict_count_for_element(a, S, S_mask):
    """Count number of triples in S\{a} whose sum is divisible by a."""
    T = [x for x in S if x != a]
    m = len(T)
    count = 0
    for i in range(m):
        for j in range(i+1, m):
            s2 = T[i] + T[j]
            for k in range(j+1, m):
                if (s2 + T[k]) % a == 0:
                    count += 1
    return count

def _total_conflicts(S):
    """Total number of (element, triple) violations in S."""
    total = 0
    for a in S:
        total += _conflict_count_for_element(a, S, None)
    return total

def _conflicts_after_add(S, c):
    """Return the number of new conflicts if c is added to S.
    New conflicts:
    1. c divides sum of some triple in S
    2. Some a in S divides sum of (c, x, y) where x,y in S, x≠a, y≠a
    """
    new_conflicts = 0
    n = len(S)
    
    # 1. c divides sum of some triple in S
    for i in range(n):
        for j in range(i+1, n):
            s2 = S[i] + S[j]
            for k in range(j+1, n):
                if (s2 + S[k]) % c == 0:
                    new_conflicts += 1
    
    # 2. For each a in S, count triples (c, x, y) with x,y in S\{a}, x≠y, and (c+x+y)%a==0
    for a in S:
        for i in range(n):
            xi = S[i]
            if xi == a:
                continue
            for j in range(n):
                if j == i or S[j] == a:
                    continue
                if (c + xi + S[j]) % a == 0:
                    new_conflicts += 1
    
    return new_conflicts

def _conflicts_after_remove(S, a):
    """Return the number of conflicts removed if a is removed from S."""
    # Conflicts involving a as divisor
    removed = _conflict_count_for_element(a, S, None)
    
    # Conflicts where a is part of a triple that sums to multiple of some other element b
    for b in S:
        if b == a:
            continue
        # Count triples (a, x, y) with x,y in S\{a,b}, x≠y, and (a+x+y)%b==0
        T = [x for x in S if x != a and x != b]
        m = len(T)
        for i in range(m):
            for j in range(i+1, m):
                if (a + T[i] + T[j]) % b == 0:
                    removed += 1
    
    return removed

def _greedy_construct(order, deadline):
    """Greedy construction: add elements in given order if no new conflicts."""
    S = []
    for c in order:
        if time.time() > deadline:
            break
        if len(S) >= MAX_CARD:
            break
        if _conflicts_after_add(S, c) == 0:
            S.append(c)
    return S

def _local_search(S, deadline, rng, max_iters=5000):
    """Local search: try swaps and moves to reduce conflicts."""
    S = list(S)
    current_conflicts = _total_conflicts(S)
    best_S = list(S)
    best_conflicts = current_conflicts
    
    all_elements = set(range(1, N+1))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Random move: swap an element in S with one not in S, or remove one
        if len(S) > 5 and rng.random() < 0.3:
            # Remove a random element
            idx = rng.randrange(len(S))
            a = S[idx]
            S.pop(idx)
            new_conflicts = _total_conflicts(S)
            if new_conflicts < current_conflicts or (new_conflicts == current_conflicts and len(S) > len(best_S)):
                current_conflicts = new_conflicts
                if len(S) > len(best_S) or (len(S) == len(best_S) and new_conflicts < best_conflicts):
                    best_S = list(S)
                    best_conflicts = new_conflicts
            else:
                S.append(a)
        else:
            # Swap
            idx = rng.randrange(len(S))
            a = S[idx]
            candidates = list(all_elements - set(S))
            if not candidates:
                continue
            c = rng.choice(candidates)
            S[idx] = c
            new_conflicts = _total_conflicts(S)
            if new_conflicts < current_conflicts or (new_conflicts == current_conflicts and len(S) > len(best_S)):
                current_conflicts = new_conflicts
                if len(S) > len(best_S) or (len(S) == len(best_S) and new_conflicts < best_conflicts):
                    best_S = list(S)
                    best_conflicts = new_conflicts
            else:
                S[idx] = a
    
    return best_S, best_conflicts

def _simulated_annealing(S, deadline, rng, max_iters=20000):
    """Simulated annealing to find low-conflict sets."""
    S = list(S)
    current_conflicts = _total_conflicts(S)
    best_S = list(S)
    best_conflicts = current_conflicts
    
    all_elements = list(range(1, N+1))
    T_start = 50.0
    T_end = 0.5
    cooling = (T_end / T_start) ** (1.0 / max(max_iters, 1))
    T = T_start
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Propose move
        move_type = rng.random()
        if move_type < 0.4:
            # Swap
            idx = rng.randrange(len(S))
            a = S[idx]
            c = rng.choice(all_elements)
            if c == a:
                continue
            S[idx] = c
            new_conflicts = _total_conflicts(S)
            delta = new_conflicts - current_conflicts
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_conflicts = new_conflicts
                if new_conflicts < best_conflicts or (new_conflicts == best_conflicts and len(S) > len(best_S)):
                    best_S = list(S)
                    best_conflicts = new_conflicts
            else:
                S[idx] = a
        elif move_type < 0.7:
            # Add an element (if below max)
            if len(S) < MAX_CARD:
                c = rng.choice(all_elements)
                if c not in S:
                    S.append(c)
                    new_conflicts = _total_conflicts(S)
                    delta = new_conflicts - current_conflicts
                    if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                        current_conflicts = new_conflicts
                        if new_conflicts < best_conflicts or (new_conflicts == best_conflicts and len(S) > len(best_S)):
                            best_S = list(S)
                            best_conflicts = new_conflicts
                    else:
                        S.pop()
        else:
            # Remove an element
            if len(S) > 5:
                idx = rng.randrange(len(S))
                a = S[idx]
                S.pop(idx)
                new_conflicts = _total_conflicts(S)
                delta = new_conflicts - current_conflicts
                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
                    current_conflicts = new_conflicts
                    if new_conflicts < best_conflicts or (new_conflicts == best_conflicts and len(S) > len(best_S)):
                        best_S = list(S)
                        best_conflicts = new_conflicts
                else:
                    S.append(a)
        
        T *= cooling
    
    return best_S, best_conflicts

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    rng = random.Random(42)
    
    best_set = []
    best_size = 0
    best_conflicts = float('inf')
    
    # Phase 1: Greedy constructions with different orderings
    orderings = [
        list(range(N, 0, -1)),  # descending
        list(range(1, N+1)),    # ascending
        [x for x in range(N, 0, -1) if x > N//2] + [x for x in range(N//2, 0, -1)],  # upper first
        [x for x in range(1, N+1) if x > N//2] + [x for x in range(1, N//2+1)],  # upper first asc
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        S = _greedy_construct(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Phase 2: Local search from best greedy result
    if best_set and time.time() < deadline - 10:
        S, conf = _local_search(best_set, deadline, rng, max_iters=3000)
        if len(S) > best_size or (len(S) == best_size and conf < best_conflicts):
            best_set = S
            best_size = len(S)
            best_conflicts = conf
    
    # Phase 3: Simulated annealing from multiple starting points
    if time.time() < deadline - 15:
        start_points = [best_set]
        # Add some random starting points
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            rp = rng.sample(range(1, N+1), min(40, N))
            S, conf = _simulated_annealing(rp, deadline, rng, max_iters=5000)
            if len(S) > best_size or (len(S) == best_size and conf < best_conflicts):
                best_set = S
                best_size = len(S)
                best_conflicts = conf
    
    # Final: try to fill up to MAX_CARD by adding non-conflicting elements
    if best_set and time.time() < deadline - 2:
        S = list(best_set)
        for c in range(1, N+1):
            if time.time() > deadline - 1:
                break
            if c in S:
                continue
            if len(S) >= MAX_CARD:
                break
            if _conflicts_after_add(S, c) == 0:
                S.append(c)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
    
    return best_set
# EVOLVE-BLOCK-END
