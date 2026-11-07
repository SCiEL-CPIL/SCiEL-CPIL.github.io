import time
import random

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    limit = P['n']
    all_nums = list(range(2, limit + 1))
    n = len(all_nums)
    num_to_idx = {v: i for i, v in enumerate(all_nums)}
    
    # Precompute conflict graph:
    # edge (u,v) exists if there exists w in [2,256], w != u, w != v, such that w | (u*v)
    # For a set S to be valid, no three distinct a,b,c in S can have a | b*c.
    # If we include u and v in S, then we must NOT include any w in S such that w | u*v and w != u, v.
    # So the conflict is: u and v are incompatible with w if w | u*v.
    # But we want pairwise: u and v are "in conflict" if including both forces us to exclude some w.
    # Actually, the constraint is: for any three distinct a,b,c in S, a does not divide b*c.
    # This means: if we have b,c in S, then no a in S (a != b, a != c) can divide b*c.
    # So for each pair (b,c), the set of "forbidden" a's are those that divide b*c.
    # For the set to be valid, S cannot contain any element from the forbidden set of any pair in S.
    # This is equivalent to: S is an independent set in a graph where... hmm, it's not directly pairwise.
    # Let me think differently.
    # Actually, the constraint is symmetric in b,c but not in a.
    # Let me define: for each element x, and each pair (y,z) with y != z, y != x, z != x:
    #   if x | y*z, then x, y, z cannot all be in S.
    # This is a 3-uniform hypergraph constraint.
    # To reduce to pairwise: x and y are "conflicting" if there exists z such that x | y*z or y | x*z.
    # No wait, that's not right either.
    
    # Let me use the direct approach: precompute for each pair (u,v) the set of w that divide u*v.
    # Then in a greedy construction, when considering adding a new element, check if it conflicts with any existing pair.
    
    # Actually, let me just use the randomized greedy with direct validation.
    # For each restart, build a set greedily, randomly ordering candidates.
    
    # Precompute: for each x, the list of (y,z) pairs where x | y*z, y != x, z != x, y != z
    # This is expensive. Let me instead precompute divisors.
    
    # Precompute: for each number x, its divisors in [2, 256]
    divisors = [set() for _ in range(limit + 1)]
    for d in range(2, limit + 1):
        for m in range(2, limit // d + 1):
            if d * m <= limit:
                divisors[d * m].add(d)
    
    # For a set S to be valid: for every three distinct a,b,c in S, a does not divide b*c.
    # Equivalently: for every a in S, and every pair (b,c) in S with b != c, b != a, c != a:
    #   a does not divide b*c.
    # So a is "blocked" by pair (b,c) if a | b*c.
    
    # Greedy approach: maintain a set S. When adding a new element x:
    #   Check that for every pair (b,c) already in S (b != c), x does not divide b*c.
    #   Also check that for every y in S (y != x), x does not divide y*z for any z in S, z != x, z != y.
    #   And y does not divide x*z for any z in S, z != x, z != y.
    
    # This is O(|S|^2) per addition. With |S| up to ~80, that's 6400 per addition, 80 additions = 512000 per restart.
    # With many restarts, this could be slow. Let me optimize.
    
    # Alternative: precompute for each x, the set of pairs (b,c) such that x | b*c.
    # Then when adding x, check if any such pair is already fully in S.
    
    # Let me precompute: for each x, blocked_pairs[x] = set of frozenset({b,c}) where b != c, b != x, c != x, and x | b*c
    # This is O(n^3) to precompute, which is 255^3 ~ 16.5M, too slow.
    
    # Let me just do the direct check with early termination.
    
    best_set = None
    best_size = 0
    
    random.seed(42)
    
    def is_valid_addition(x, S):
        """Check if x can be added to S without violating constraints."""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                b = S_list[i]
                c = S_list[j]
                # Check x | b*c
                if (b * c) % x == 0:
                    return False
                # Check b | x*c
                if (x * c) % b == 0:
                    return False
                # Check c | x*b
                if (x * b) % c == 0:
                    return False
        return True
    
    # Try multiple randomized restarts
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.shuffle(all_nums)
        S = set()
        
        for x in all_nums:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            # Check if we've found a large enough set to be confident
            if best_size >= 80:
                # Keep going for a bit to see if we can do better
                pass
    
    # Final validation
    if best_set is None:
        # Fallback: all primes
        is_prime = [True] * (limit + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    result = sorted(best_set)
    
    # Final sanity check
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            for k in range(j + 1, len(result)):
                a, b, c = result[i], result[j], result[k]
                if (b * c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to primes
        is_prime = [True] * (limit + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    return result

# EVOLVE-BLOCK-END
