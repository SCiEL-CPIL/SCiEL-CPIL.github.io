import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 256
    candidates = list(range(2, n + 1))
    num_candidates = len(candidates)  # 255
    
    # Map each candidate to an index 0..254
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    
    # Precompute: for each pair (i, j) with i < j, which values a divide b*c?
    # conflict_mask[a_idx] = bitmask of indices that cannot coexist with a_idx in certain triples
    # Actually, let's precompute: for each triple (a, b, c) distinct, if a | b*c, mark conflict.
    # We'll use a different approach: for each pair (b, c), precompute which a's conflict.
    
    # Precompute conflict structure:
    # For a set S, S is valid iff for all distinct a,b,c in S: a does not divide b*c.
    # Equivalently: for all distinct a,b,c in S: b*c % a != 0.
    
    # Let's precompute for each pair (b_idx, c_idx) the set of a_idx that conflict.
    # conflict_bc[b_idx][c_idx] = set of a_idx such that candidates[a_idx] | (candidates[b_idx] * candidates[c_idx])
    
    # But that's O(n^3) space. Instead, let's precompute:
    # For each a, the set of pairs (b,c) where a | b*c.
    # Then a conflicts with the set S if there exist distinct b,c in S\{a} such that a | b*c.
    
    # Better approach for bitmask:
    # For each a_idx, precompute a bitmask of "forbidden pairs" - but that's complex.
    
    # Alternative: precompute for each a_idx and each other_idx whether a_idx | candidates[a_idx] * candidates[other_idx]... no.
    
    # Let me think differently. The constraint is: for ALL distinct a,b,c in S, a ∤ b*c.
    # This is equivalent to: for each a in S, there do NOT exist distinct b,c in S\{a} such that a | b*c.
    
    # So for a given a and a set S, a is "blocked" by S if ∃ distinct b,c in S\{a} with a | b*c.
    
    # Precompute: for each a_val, list of pairs (b_val, c_val) with b≠c, b≠a, c≠a, such that a | b*c.
    # Then a is blocked by S iff ∃ such pair (b,c) where both b,c ∈ S.
    
    # For efficiency, let's precompute for each a_idx:
    #   blocked_by[a_idx] = list of (b_idx, c_idx) pairs where a | candidates[b]*candidates[c], b≠c, b≠a, c≠a
    
    # This could be large. Let's estimate: for a=2, b*c must be even. Many pairs.
    # For a=255, fewer.
    
    # Actually, let's use a different representation:
    # For each a_idx, precompute a bitmask pair_mask[a_idx] where bit (b_idx * n + c_idx) or something...
    # That's too large.
    
    # Simpler: for each a_idx, store a list of frozensets or just check on the fly with precomputed divisibility.
    
    # Let me precompute: div_table[a_val][b_val] = whether a_val divides b_val
    # And for the triple check: a | b*c. We can precompute for each a, the set of b values where... 
    # No, it depends on both b and c.
    
    # Precompute: for each a in 2..256, for each b in 2..256, b != a:
    #   rem_a_b = b % a  (not directly useful)
    #   Actually: a | b*c iff (b mod a) * (c mod a) ≡ 0 (mod a)? No, that's not right either.
    #   a | b*c is a direct check.
    
    # Let's precompute a 3D-like structure but compressed:
    # For each a (2..256), create a set of "conflicting pairs".
    # conflict_pairs[a] = set of frozenset({b,c}) where b≠c, b≠a, c≠a, and a | b*c.
    
    # Then to check if a can be added to S: check if any frozenset({b,c}) ⊆ S for (b,c) in conflict_pairs[a].
    
    # This is O(|conflict_pairs[a]|) per check. For a=2, this could be ~O(n^2/4) which is ~16k. 
    # That's too slow for local search.
    
    # Better: for each a, precompute for each b≠a: the set of c values such that a | b*c and c≠a, c≠b.
    # Then a is blocked by S iff ∃ b in S\{a} such that (conflict_c[a][b] ∩ S\{a,b}) is non-empty.
    
    # Let's precompute: conflict_c[a_val][b_val] = set of c_val where a_val | b_val * c_val, c_val ≠ a_val, c_val ≠ b_val.
    
    # For local search, we can maintain for each a not in S, a count of how many "witness pairs" exist in S.
    # But that's complex. Let's just do direct checks with precomputed data.
    
    # Precompute divisibility: a | b*c
    # For speed, precompute for each a (2..256) and each b (2..256):
    #   gcd_ab = gcd(a, b)
    #   a | b*c iff (a/gcd_ab) | c
    # So: a | b*c iff c is a multiple of (a // gcd(a,b)).
    
    from math import gcd
    
    # Precompute: for each a (2..256), for each b (2..256), b != a:
    #   req_c[a][b] = a // gcd(a, b)  (the minimum requirement on c: c must be divisible by this)
    #   Then a | b*c iff c % req_c[a][b] == 0 (and c != a, c != b)
    
    req_c = {}
    for a in range(2, n + 1):
        req_c[a] = {}
        for b in range(2, n + 1):
            if b == a:
                continue
            g = gcd(a, b)
            req_c[a][b] = a // g
    
    # Now, to check if a can be added to set S:
    # For each b in S (b != a), check if there exists c in S (c != a, c != b) such that c % req_c[a][b] == 0.
    # If yes, then a | b*c for some pair, so a cannot be added.
    
    # For efficiency, for each a, precompute which c values satisfy c % req_c[a][b] == 0 for each b.
    # But that's still per-b. Let's just do it directly.
    
    def can_add_to_set(S_set, a):
        """Check if a can be added to S_set (S_set is a set of values, a not in S_set)."""
        if a in S_set:
            return True
        S_list = list(S_set)
        for b in S_list:
            if b == a:
                continue
            r = req_c[a][b]
            if r == 1:
                # a | b*c for ANY c != a, b. So if |S_set| >= 2 (excluding a), conflict.
                # Actually we need c != a and c != b. Since a not in S_set, just need c != b.
                # So if there's any c in S_set with c != b, conflict.
                if len(S_set) >= 2:
                    return False
            else:
                # Check if any c in S_set, c != b, has c % r == 0
                for c in S_list:
                    if c == b or c == a:
                        continue
                    if c % r == 0:
                        return False
        return True
    
    # Phase 1: Greedy construction
    S = set()
    # Sort candidates by some heuristic - maybe by number of conflicts (fewer conflicts first)
    # Or just start with primes
    
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    if n >= 1:
        is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Start with all primes
    S = set(primes)
    best_S = set(S)
    best_size = len(S)
    
    # Try to add non-primes
    for x in candidates:
        if x in S:
            continue
        if can_add_to_set(S, x):
            S.add(x)
    
    if len(S) > best_size:
        best_size = len(S)
        best_S = set(S)
    
    # Local search: try 1-for-1 swaps
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for x in candidates:
            if x in S:
                continue
            if can_add_to_set(S, x):
                S.add(x)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                improved = True
    
    # Try removing one and adding another
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        if not S:
            break
        to_remove = random.choice(list(S))
        new_S = set(S)
        new_S.discard(to_remove)
        
        added_any = False
        for x in candidates:
            if x in new_S:
                continue
            if can_add_to_set(new_S, x):
                new_S.add(x)
                added_any = True
        
        if len(new_S) > len(S):
            S = new_S
            if len(S) > best_size:
                best_size = len(S)
                best_S = set(S)
    
    # Phase 2: Branch and bound for maximum independent set in the hypergraph
    # Use the current best as initial solution
    # Represent set as a Python set of values
    
    # Precompute for faster conflict checking using bitmasks
    # Map values to indices 0..254
    # For each index i, precompute a list of (j, k) pairs that conflict with i
    # i.e., candidates[i] | candidates[j] * candidates[k] or permutations
    
    # Actually, let's precompute: for each i, the set of pairs (j,k) j<k, j≠i, k≠i, such that
    # candidates[i] | candidates[j]*candidates[k] OR candidates[j] | candidates[i]*candidates[k] OR candidates[k] | candidates[i]*candidates[j]
    
    # This is the "conflict pairs" for vertex i in the 3-uniform hypergraph.
    # i cannot be in the independent set with both j and k if (j,k) is a conflict pair of i.
    
    # For branch and bound, we need to check if adding a vertex is compatible with the current partial solution.
    
    # Let's use a different approach: precompute for each pair (j, k), the set of i that conflict with (j,k).
    # conflict_with_pair[j][k] = set of i such that i conflicts with pair (j,k).
    # Then i can be added to S iff for all pairs (j,k) in S (j<k), i not in conflict_with_pair[j][k].
    
    # Precompute conflict_with_pair
    conflict_with_pair = {}
    for j in range(num_candidates):
        conflict_with_pair[j] = {}
        for k in range(j + 1, num_candidates):
            cv_j = candidates[j]
            cv_k = candidates[k]
            conflict_set = set()
            for i in range(num_candidates):
                if i == j or i == k:
                    continue
                cv_i = candidates[i]
                # Check if any of the three divisibility conditions hold
                if (cv_j * cv_k) % cv_i == 0:  # cv_i | cv_j * cv_k
                    conflict_set.add(i)
                elif (cv_i * cv_k) % cv_j == 0:  # cv_j | cv_i * cv_k
                    conflict_set.add(i)
                elif (cv_i * cv_j) % cv_k == 0:  # cv_k | cv_i * cv_j
                    conflict_set.add(i)
            conflict_with_pair[j][k] = conflict_set
    
    # For branch and bound, we'll work with the current best set and try to improve.
    # Use a simpler approach: iterated local search with restarts.
    
    def validate_set(S_val):
        """Validate that S_val is a valid set."""
        S_list = list(S_val)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True
    
    # Iterated local search
    for iteration in range(50):
        if time.time() > deadline - 5:
            break
        
        # Start from current best with some random perturbation
        current_S = set(best_S)
        
        # Randomly remove a few elements
        if current_S:
            num_remove = min(3, len(current_S) // 4)
            to_remove = random.sample(list(current_S), num_remove)
            for x in to_remove:
                current_S.discard(x)
        
        # Greedily add back
        for x in candidates:
            if x in current_S:
                continue
            if can_add_to_set(current_S, x):
                current_S.add(x)
        
        # Local search
        improved_local = True
        while improved_local and time.time() < deadline - 5:
            improved_local = False
            for x in candidates:
                if x in current_S:
                    continue
                if can_add_to_set(current_S, x):
                    current_S.add(x)
                    improved_local = True
            # Try swaps
            if not improved_local:
                for x in candidates:
                    if x in current_S:
                        continue
                    for y in list(current_S):
                        new_S = set(current_S)
                        new_S.discard(y)
                        if can_add_to_set(new_S, x):
                            new_S.add(x)
                            if len(new_S) > len(current_S):
                                current_S = new_S
                                improved_local = True
                                break
                    if improved_local:
                        break
        
        if len(current_S) > best_size:
            best_size = len(current_S)
            best_S = set(current_S)
    
    # Final validation
    if not validate_set(best_S):
        # Fallback to primes
        best_S = set(primes)
    
    result = [int(x) for x in best_S]
    return result
