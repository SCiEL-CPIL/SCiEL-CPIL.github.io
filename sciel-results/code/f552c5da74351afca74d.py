"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'n': 256}

def construct():
    deadline = time.time() + 85  # Safety margin for 90s limit
    
    n = P['n']
    candidates = list(range(2, n + 1))
    m = len(candidates)
    
    # 1. Precompute conflict graph using bitsets for speed
    # conflict[i] is a bitset of all j such that candidates[i] divides candidates[j] * candidates[k] for some k?
    # Actually, the condition is: for distinct a, b, c in S, a does NOT divide b*c.
    # This is a 3-uniform hypergraph constraint.
    # However, we can reframe:
    # If we pick a set S, for any a in S, and any pair {b, c} in S\{a}, we must have a !| b*c.
    
    # Strategy:
    # We want the maximum subset S of candidates such that no triple (a,b,c) in S violates a | b*c.
    
    # Let's build a graph where edges represent "incompatibility" in a pairwise sense?
    # No, the incompatibility is 3-way.
    # But we can use a greedy construction with backtracking.
    
    # Precompute: for each a, and for each b, which c's make a | b*c?
    # If a | b*c, then (a, b, c) is a bad triple.
    
    # Let's define a "danger" matrix for a fixed 'a'.
    # For a given a, if we include b and c in our set (b,c != a), we must ensure a does not divide b*c.
    
    # Efficient Search:
    # Use a recursive backtracking that builds the set.
    # State: current_set, candidates_remaining
    # Optimization: Order candidates by "dangerousness" (how many conflicts they create).
    
    # Let's precompute for each pair (b, c), the set of 'a's that divide b*c.
    # If we have b and c in our set, we must EXCLUDE all a that divide b*c.
    
    # This allows a very efficient check:
    # If we add a new element x to our current set S, we must ensure that for all pairs (b, c) in S, 
    # x does not divide b*c AND b does not divide x*c AND c does not divide x*b.
    
    # Let's precompute divisors.
    # divisors_of_prod[b][c] = set of a such that a | b*c.
    # Since n=256, b*c can be up to 65536.
    # We only care about a in 2..256.
    
    # Precomputation:
    # For each b in 2..256:
    #   For each c in 2..256:
    #      prod = b*c
    #      Find all a in 2..256 such that a | prod.
    #      Store this in a lookup table or bitset.
    
    # Let's use a list of bitsets.
    # bitset_a[b][c] is a bitset of size m (indices 0..m-1 corresponding to 2..256)
    # where bit i is 1 if candidates[i] divides b*c.
    
    # Actually, simpler:
    # For each a (index i), precompute a set of pairs (j, k) such that candidates[i] | candidates[j]*candidates[k].
    # This is O(N^3) precomputation, which is 256^3 ~ 16M ops. Doable in Python? Maybe slow.
    
    # Better:
    # For each a, compute the set of "forbidden pairs".
    # If a is in S, then no pair (b,c) from S\{a} can be in forbidden_pairs[a].
    
    # Let's try a different angle: Maximum Independent Set in a graph derived from the hypergraph?
    # If we treat each triple as an edge in a 3-graph, finding max independent set is NP-hard.
    
    # However, N=256 is small enough for a smart greedy + local search.
    
    # Greedy Heuristic:
    # 1. Start with empty set.
    # 2. Iterate candidates in a specific order.
    # 3. Add candidate if it doesn't violate constraints with existing pairs.
    
    # To maximize size, we should prioritize numbers that are "less conflicting".
    # A number a conflicts with a pair (b,c) if a | b*c.
    # Primes are "safe" because a prime p divides b*c only if p|b or p|c.
    # If we include many primes, we restrict composites.
    
    # Let's use a randomized greedy with restarts, or a simulated annealing approach.
    # Given the 60s limit, we can run many fast greedy tries.
    
    # Precompute divisibility:
    # divs[a] = set of b such that a | b.
    # This helps check a | b*c quickly?
    # a | b*c iff gcd(a, b) * (a // gcd(a, b)) | c? No.
    # Let g = gcd(a, b). Then a | b*c iff (a/g) | c.
    
    # This is a fast check!
    # Check(a, b, c):
    #   g = gcd(a, b)
    #   return (c % (a//g) != 0)
    
    # We will use this fast check in a greedy construction.
    
    from math import gcd
    
    def is_valid_addition(a, current_set):
        """Check if a can be added to current_set without violation."""
        for b in current_set:
            for c in current_set:
                if b == c: continue
                # Check a | b*c
                g_ab = gcd(a, b)
                if (c % (a // g_ab)) == 0:
                    return False
                # Check b | a*c
                g_ab2 = gcd(b, a)
                if (c % (b // g_ab2)) == 0:
                    return False
                # Check c | a*b
                g_ac = gcd(c, a)
                if (b % (c // g_ac)) == 0:
                    return False
        return True
    
    # This is O(|S|^2) per addition. Total O(N * |S|^2).
    # If |S| ~ 100, N=256, ops ~ 256 * 10000 = 2.5M. Very fast.
    
    # Strategy:
    # 1. Generate a random permutation of candidates.
    # 2. Greedily build a valid set.
    # 3. Keep the best set found.
    # 4. Repeat until time limit.
    
    # To improve greedy, we can try different orderings.
    # "Safe" numbers (primes, prime powers) are likely to be in the max set.
    # "Dangerous" numbers (products of small primes) are likely to be excluded.
    
    # Let's score candidates by "conflict potential".
    # Potential conflicts for a = count of pairs (b,c) in 2..256 such that a | b*c.
    # Lower potential = better candidate to add early.
    
    def conflict_score(a):
        count = 0
        for b in range(2, n + 1):
            for c in range(2, n + 1):
                if b == c: continue
                g = gcd(a, b)
                if c % (a // g) == 0:
                    count += 1
        return count
    
    # Precompute scores
    scores = {}
    for a in candidates:
        scores[a] = conflict_score(a)
    
    # Sort candidates by score (ascending) - add "safe" ones first
    sorted_candidates = sorted(candidates, key=lambda x: scores[x])
    
    best_set = set()
    best_size = 0
    
    # Try the deterministic greedy first
    current = set()
    for a in sorted_candidates:
        if is_valid_addition(a, current):
            current.add(a)
    
    if len(current) > best_size:
        best_size = len(current)
        best_set = set(current)
        
    # Now try randomized restarts with perturbation
    import random
    
    while time.time() < deadline:
        # Randomly shuffle, but bias towards lower score
        # Or just shuffle and greedily build
        random.shuffle(candidates)
        current = set()
        for a in candidates:
            if is_valid_addition(a, current):
                current.add(a)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
            
            # If we found a very large set, we might be near optimal
            # We can stop early if we hit a theoretical upper bound?
            # Upper bound: Total numbers - (numbers that MUST be excluded?)
            # Not obvious.
            
    # Final local search on best_set to see if we can add anything?
    # The greedy already tried to add. But maybe removing one allows adding two?
    # Simple swap heuristic:
    # For each x in best_set, try removing x.
    # Then try to add any y not in best_set that was previously blocked.
    
    # This is complex. Given the time, the greedy with random restarts is likely strong enough for N=256.
    
    result = sorted(best_set)
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
