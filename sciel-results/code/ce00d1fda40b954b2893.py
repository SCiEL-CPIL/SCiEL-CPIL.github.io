import time
import sys
from math import gcd, isqrt
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin for safety

    MAX_N = 256

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(MAX_N)
    num_primes = len(primes)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    # For each number 2..256, compute its prime factorization (distinct primes)
    # and the set of primes it "uses"
    num_primes_set = [set() for _ in range(MAX_N + 1)]
    for n in range(2, MAX_N + 1):
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                num_primes_set[n].add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            num_primes_set[n].add(temp)

    # Group numbers by their prime set
    # For each subset of primes, find the best (max value) number that uses exactly those primes
    # Actually, we need: for each prime, what's the best number we can assign to it?
    # But a number can use multiple primes.
    
    # Key insight: each prime can appear in at most ONE selected number.
    # So we can think of it as: assign each prime to at most one number,
    # where a number "uses" a set of primes, and its value is the number itself.
    
    # Branch and bound:
    # - Process primes in order (largest first for tighter bounds)
    # - At each step, decide which number (if any) gets this prime
    # - Upper bound: for remaining primes, sum of max possible value per prime
    
    # Precompute: for each prime p, what numbers use p?
    # And for each prime p, the maximum value of any number that uses ONLY p (i.e., p^k)
    # Actually, the upper bound should be: for each unassigned prime, the max value 
    # of any number that could be assigned to it (using only unassigned primes)
    
    # Simpler upper bound: for each unassigned prime p, the best number n <= 256 
    # such that all prime factors of n are among unassigned primes, and p is one of them.
    # This is expensive to compute dynamically. Use a simpler bound:
    # For each unassigned prime p, the max value is p * (product of other unassigned primes that divide some n<=256 with p)
    # Even simpler: just use p itself as a lower bound on contribution, 
    # and for upper bound use the max n<=256 whose prime factors are all unassigned.
    
    # Let me use a different approach:
    # Sort primes descending. For prime at position i, the upper bound contribution
    # is the maximum n in [2,256] such that all prime factors of n are in primes[i:].
    # Precompute this for each suffix of the prime list.
    
    # Actually, let me think more carefully. The problem is:
    # We have primes p1, p2, ..., pk (sorted descending).
    # We want to select a set of numbers S such that no prime divides two different numbers in S.
    # Maximize sum(S).
    
    # Equivalent: partition some primes into groups, where each group G corresponds to a number
    # n = product of primes in G (possibly with powers), n <= 256, and we get value n.
    # Plus we can include 1 (value 1, uses no primes).
    
    # B&B: process primes from largest to smallest.
    # State: which primes have been "used" (assigned to some number).
    # Decision: for the current prime p, either:
    #   (a) don't use p in any number (p is wasted)
    #   (b) assign p to a number that is p * (product of some already-assigned-or-currently-being-built primes)
    #       - but this is complex. 
    # 
    # Alternative B&B formulation:
    # Build numbers one at a time. Each number is a product of distinct primes (with powers) <= 256.
    # Once a prime is used in a number, it can't be used again.
    # 
    # Let's try: at each step, pick the next number to add.
    # State: set of used primes, current sum.
    # Branch: try each valid number (not yet used, all its prime factors unused).
    # Bound: current_sum + sum over unused primes of (best number using that prime and only unused primes)
    #        but this overcounts since one number can use multiple primes.
    #        Better bound: current_sum + sum of all unused primes (since each unused prime can contribute at most itself to some number, and numbers are >= their prime factors)
    #        Actually, a number using primes {p1,p2} has value p1*p2 >= p1+p2 for p1,p2 >= 2.
    #        So the sum of unused primes is a valid lower bound on what they can contribute.
    #        For an UPPER bound: the maximum value from unused primes is at most the product of all unused primes (but that's way too loose).
    #        Better: for each unused prime p, the max n<=256 using p and only unused primes. Sum these, but subtract overlaps.
    #        Simplest valid upper bound: current_sum + sum of (max n<=256 with all prime factors in unused_primes and containing p) for each unused p, 
    #        but this double-counts numbers that use multiple unused primes.
    #        A valid (but loose) upper bound: current_sum + sum(unused_primes) * max_ratio
    #        where max_ratio accounts for the fact that p^k can be larger than p.
    #        For p=2: 2^7=128, so ratio up to 64x. For p=3: 3^5=243, ratio 81x. 
    #        This is too loose.
    
    # Let me use a tighter bound:
    # For each unused prime p, compute best[p] = max{n <= 256 : all prime factors of n are in unused_primes, p | n}
    # Then upper_bound = current_sum + sum(best[p] for p in unused_primes) - (correction for double counting)
    # The correction is hard. Instead, use:
    # upper_bound = current_sum + sum of the top values, where we greedily assign each unused prime to its best number,
    # but a number can only be assigned once. This is a greedy upper bound.
    
    # Actually, the simplest correct upper bound: 
    # For each unused prime p, let M_p = max{n<=256 : p|n, all prime factors of n are in unused_primes}
    # Then sum(M_p for all unused p) is an upper bound on the additional sum, because each number n that uses primes {p1,...,pk} 
    # will be counted k times (once for each prime it uses), and n <= M_pi for each pi, so n <= min(M_p1,...,M_pk) <= each M_pi.
    # Wait, that's not right either. n could be larger than individual primes.
    # 
    # Correct approach: The sum of all selected numbers using unused primes is at most 
    # the maximum total we can get by optimally packing unused primes into numbers.
    # An upper bound on this is: for each unused prime p, the maximum n<=256 using p and only unused primes.
    # Sum these over all unused primes, then divide by the minimum "share" each prime gets.
    # This is getting complicated.
    
    # Let me just use: upper_bound = current_sum + sum(unused_primes)
    # This is valid because any number n composed of primes {p1,...,pk} satisfies n >= max(pi), 
    # and the total contribution of those primes is n, while the sum of primes is p1+...+pk.
    # For k=1: n = p^m >= p. OK.
    # For k=2: n = p*q^m. Is p*q^m <= p + q? No! 2*3=6 > 2+3=5.
    # So sum(unused_primes) is NOT a valid upper bound.
    
    # Valid upper bound: for each unused prime p, let U_p = max{n<=256 : all prime factors of n are in unused_primes, p|n}
    # Then the total value from unused primes is at most sum(U_p) / min_count, where min_count is... 
    # Actually, each number n using primes {p1,...,pk} contributes n to the sum, and n <= U_pi for each i.
    # So n <= min(U_p1, ..., U_pk). The total sum is sum over numbers of n, each n <= min of its U's.
    # A valid upper bound: sum(U_p for p in unused) is valid because each number n using primes S satisfies 
    # n <= U_p for each p in S, so n <= U_p for a specific p, and summing over all numbers, 
    # each number is bounded by U_p for any p it uses. But we need to sum over all numbers.
    # If we have numbers n1, n2, ... where ni uses primes Si, then sum(ni) <= sum over i of min(U_p for p in Si).
    # And sum over i of min(U_p for p in Si) <= sum over i of U_p1i (for any fixed prime in Si).
    # This doesn't directly give sum(U_p).
    
    # OK let me just use a simple valid upper bound:
    # UB = current_sum + sum over unused primes p of (largest power of p that is <= 256)
    # Because any number using p has value at most p^floor(log_p(256)) * (other primes), but the other primes 
    # would be counted separately. Actually no, this isn't valid either.
    
    # Simplest valid UB: current_sum + sum(unused_primes) * 256 // min(unused_primes) -- no, too loose.
    
    # Let me just use: UB = current_sum + sum(unused_primes)
    # And accept it's not always valid... wait, let me reconsider.
    # 
    # Counter-example: unused primes {2, 3}. Best number is 6 (uses both). Sum of primes = 5. 6 > 5. Invalid.
    # 
    # Valid UB: for each unused prime p, the maximum n<=256 whose prime factors are all in unused_primes and p divides n.
    # Let this be B_p. Then for any valid selection of numbers using unused primes, 
    # the sum of those numbers is at most sum(B_p) because:
    # Each number n in the selection uses some set S_n of unused primes. n <= B_p for each p in S_n.
    # So n <= min_{p in S_n} B_p <= B_p for any specific p in S_n.
    # Now, sum over all numbers n of n <= sum over all numbers n of B_{p_n} for any choice of p_n in S_n.
    # If we choose for each number n a prime p_n in S_n, and if each unused prime p is chosen by at most one number,
    # then sum(n) <= sum(B_p for p that are chosen). But primes can be shared in the sense that multiple numbers 
    # might "want" the same prime, but in a valid selection each prime is used by at most one number.
    # So in a valid selection, each unused prime p is used by at most one number. 
    # For that number n, n <= B_p. So the total sum from unused primes is <= sum(B_p for p used) <= sum(B_p for all unused p).
    # YES! This is valid!
    
    # So: UB = current_sum + sum(B_p for p in unused_primes)
    # where B_p = max{n<=256 : p|n, all prime factors of n are in unused_primes}
    
    # Precomputing B_p dynamically is expensive. Let me precompute for each prime p and each mask of "available" primes... 
    # that's 2^54 states, too many.
    
    # Alternative: since we process primes in order, the "unused primes" are always a suffix of the sorted prime list.
    # So I can precompute B_p_suffix[i] = max{n<=256 : p_i|n, all prime factors of n are in primes[i:]}
    # for each i and each prime p_i in primes[i:].
    # That's O(num_primes^2 * 256) which is manageable.
    
    # Actually, let me simplify. Process primes in descending order.
    # At step i, unused primes are primes[i:].
    # B_p for prime p (where p is in primes[i:]) = max{n<=256 : p|n, all prime factors of n are in primes[i:]}
    
    # Precompute: for each suffix index i, and for each prime p in primes[i:], 
    # B[p][i] = max{n<=256 : p|n, all prime factors of n are in primes[i:]}
    
    # This is O(num_primes^2 * 256 * cost_of_check) which might be slow. Let me optimize.
    
    # Actually, let me just compute it on the fly with memoization, or use a simpler bound.
    
    # Given the complexity, let me use a heuristic approach:
    # 1. Greedy: sort numbers descending, pick if coprime to all selected.
    # 2. Local search with the current program's moves.
    # 3. Add a B&B with a simple bound: current_sum + sum(unused_primes) * 2 (heuristic, may not be valid)
    #    Actually, let me just use the valid bound but compute it approximately.
    
    # For now, let me implement the B&B with the valid bound, computing B_p on the fly.
    
    # Sort primes descending for B&B
    primes_desc = sorted(primes, reverse=True)
    num_primes_desc = len(primes_desc)
    prime_desc_to_idx = {p: i for i, p in enumerate(primes_desc)}
    
    # For each prime p, precompute all numbers n<=256 that use exactly the prime set containing p
    # and no other primes outside a given suffix.
    
    # Let me precompute: for each n in 2..256, its prime set as a frozenset.
    n_prime_sets = [frozenset() for _ in range(MAX_N + 1)]
    for n in range(2, MAX_N + 1):
        n_prime_sets[n] = frozenset(num_primes_set[n])
    
    # For B&B, process primes in descending order.
    # At step i (0-indexed), we've decided for primes_desc[0..i-1].
    # Unused primes: primes_desc[i:]
    
    # B_p for prime p at step i: max{n<=256 : p|n, prime_set(n) ⊆ set(primes_desc[i:])}
    
    # Precompute for each i (suffix starting at i) and each prime p in primes_desc[i:]:
    # best_val[i][p] = max{n<=256 : p|n, prime_set(n) ⊆ set(primes_desc[i:])}
    
    # This is O(num_primes * num_primes * 256) = O(54^2 * 256) ≈ 750K, fine.
    
    best_val = [[0] * num_primes_desc for _ in range(num_primes_desc + 1)]
    # best_val[i][j] = max{n<=256 : primes_desc[j]|n, prime_set(n) ⊆ set(primes_desc[i:])}
    # for j >= i
    
    # For suffix i, the set of allowed primes is set(primes_desc[i:])
    # For each n in 2..256, check if prime_set(n) ⊆ allowed and if so, for each p in prime_set(n), update best
    
    # More efficient: for each i, iterate n and check.
    for i in range(num_primes_desc):
        allowed = set(primes_desc[i:])
        for n in range(2, MAX_N + 1):
            ps = n_prime_sets[n]
            if ps and ps.issubset(allowed):
                for p in ps:
                    j = prime_desc_to_idx[p]
                    if n > best_val[i][j]:
                        best_val[i][j] = n
    
    # best_val[num_primes_desc][*] = 0 (no unused primes)
    
    # B&B:
    # State: i (next prime to decide), used_mask (which of primes_desc[0..i-1] have been used), current_sum
    # Actually, we need to track which primes have been "consumed" by selected numbers.
    # 
    # Let me reframe: we're building a set of numbers. Each number uses a set of primes.
    # No prime can be used by two numbers.
    # 
    # B&B state: set of used primes, current sum, best number found.
    # Branch: try adding each valid number (all its primes unused).
    # Bound: current_sum + sum(best_val[i][j] for j where primes_desc[j] is unused and j >= i)
    #        where i is the index of the smallest unused prime... this is getting complicated.
    
    # Simpler B&B: 
    # Process primes in order. For each prime, decide which number it belongs to (or none).
    # But a number can contain multiple primes, so this is a packing problem.
    
    # Given the time constraints, let me just do a good greedy + local search, 
    # plus a limited B&B for the hardest cases.
    
    # First, greedy solution
    S = []
    for n in range(MAX_N, 0, -1):
        if all(gcd(n, x) == 1 for x in S):
            S.append(n)
    
    best_set = set(S)
    cur_sum = sum(S)
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = sorted(list(best_set), reverse=True)
        
        # Try to add any missing number
        for b in range(MAX_N, 1, -1):
            if b in best_set:
                continue
            if all(gcd(b, x) == 1 for x in best_set):
                best_set.add(b)
                cur_sum += b
                improved = True
        
        # 1-for-1 swaps
        for a in S_list:
            if a == 1:
                continue
            for b in range(MAX_N, a, -1):
                if b in best_set:
                    continue
                if all(gcd(b, x) == 1 for x in best_set if x != a):
                    new_sum = cur_sum - a + b
                    if new_sum > cur_sum:
                        best_set.remove(a)
                        best_set.add(b)
                        cur_sum = new_sum
                        improved = True
        
        # 2-for-1 swaps
        if time.time() < deadline:
            S_list = sorted(list(best_set))
            for i in range(len(S_list)):
                if time.time() >= deadline:
                    break
                a1 = S_list[i]
                if a1 == 1:
                    continue
                for j in range(i + 1, len(S_list)):
                    a2 = S_list[j]
                    if a2 == 1:
                        continue
                    for b in range(MAX_N, a1 + a2, -1):
                        if b in best_set or b == a1 or b == a2:
                            continue
                        if all(gcd(b, x) == 1 for x in best_set if x not in (a1, a2)):
                            best_set.remove(a1)
                            best_set.remove(a2)
                            best_set.add(b)
                            cur_sum = cur_sum - a1 - a2 + b
                            improved = True
                            break
        
        # 2-for-2 swaps
        if time.time() < deadline:
            S_list = sorted(list(best_set))
            for i in range(len(S_list)):
                if time.time() >= deadline:
                    break
                a1 = S_list[i]
                if a1 == 1:
                    continue
                for j in range(i + 1, len(S_list)):
                    a2 = S_list[j]
                    if a2 == 1:
                        continue
                    for b1 in range(MAX_N, a1, -1):
                        if b1 in best_set or b1 == a1 or b1 == a2:
                            continue
                        if not all(gcd(b1, x) == 1 for x in best_set if x not in (a1, a2)):
                            continue
                        for b2 in range(MAX_N, max(a1, a2), -1):
                            if b2 in best_set or b2 == a1 or b2 == a2 or b2 == b1:
                                continue
                            if not all(gcd(b2, x) == 1 for x in best_set if x not in (a1, a2)):
                                continue
                            if gcd(b1, b2) != 1:
                                continue
                            new_sum = cur_sum - a1 - a2 + b1 + b2
                            if new_sum > cur_sum:
                                best_set.remove(a1)
                                best_set.remove(a2)
                                best_set.add(b1)
                                best_set.add(b2)
                                cur_sum = new_sum
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
    
    # Now try B&B to improve
    if time.time() < deadline - 5:
        # B&B: try to find a better solution
        # Use the prime-packing structure
        # Process primes in descending order
        # State: index i, used_primes (set), current_numbers (list), current_sum
        # Bound: current_sum + sum(best_val[i][j] for j in range(i, num_primes_desc) if primes_desc[j] not in used_primes)
        
        # This is still complex. Let me do a simpler targeted search:
        # For each prime, check if we can replace the current assignment with a better one.
        
        pass  # B&B not implemented in this iteration; local search should suffice
    
    # Final: try to add 1
    if 1 not in best_set:
        best_set.add(1)
        cur_sum += 1
    
    return [int(x) for x in sorted(best_set)]
