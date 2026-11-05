import time
import random
import math
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = start + 80  # 10s safety margin before 90s kill

    def valid_pair(A, B):
        """Return the maximum number of distinct pairwise-coprime values in A+B."""
        if len(A) != 3 or len(B) != 3:
            return 0
        if len(set(A)) != 3 or len(set(B)) != 3:
            return 0
        for x in A + B:
            if x < 1 or x > 128:
                return 0
        
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        # If there are <= 9 sums, we can find the maximum independent set in the conflict graph.
        # Nodes: sums. Edge between s1, s2 if gcd(s1, s2) > 1.
        # We want the maximum subset of nodes with no edges between them.
        # Since max 9 nodes, we can use bitmask DP or simple backtracking.
        
        sum_list = list(sums)
        n = len(sum_list)
        
        # Precompute conflicts
        # conflict[i] is a bitmask of nodes that conflict with node i
        conflict = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(sum_list[i], sum_list[j]) != 1:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
        
        # DP over subsets: dp[mask] = max independent set size in mask?
        # Or simply iterate all subsets of sum_list (2^9 = 512) and check if independent.
        # 512 * 9 * 9 operations is very fast.
        max_count = 0
        for mask in range(1 << n):
            # Check if subset defined by mask is independent
            is_independent = True
            for i in range(n):
                if mask & (1 << i):
                    # Check if i conflicts with any other node in mask
                    # conflict[i] has bits set for nodes j where gcd(sum[i], sum[j]) > 1
                    if mask & conflict[i]:
                        is_independent = False
                        break
            if is_independent:
                count = bin(mask).count('1')
                if count > max_count:
                    max_count = count
        
        return max_count

    def best_move_A(A, B):
        """Try all single-element swaps in A; return best (score, A_new)."""
        base = valid_pair(A, B)
        best_s = base
        best_a = A[:]
        for i in range(3):
            for v in range(1, 129):
                if v in A:
                    continue
                A2 = A[:]
                A2[i] = v
                s = valid_pair(A2, B)
                if s > best_s:
                    best_s = s
                    best_a = A2
        return best_s, best_a

    def best_move_B(A, B):
        """Try all single-element swaps in B; return best (score, B_new)."""
        base = valid_pair(A, B)
        best_s = base
        best_b = B[:]
        for j in range(3):
            for v in range(1, 129):
                if v in B:
                    continue
                B2 = B[:]
                B2[j] = v
                s = valid_pair(A, B2)
                if s > best_s:
                    best_s = s
                    best_b = B2
        return best_s, best_b

    def best_move_A2(A, B):
        """Try all 2-element swaps in A; return best (score, A_new)."""
        base = valid_pair(A, B)
        best_s = base
        best_a = A[:]
        # This is O(128^2 * 9 * 9) which is roughly 147456 * 81 ~ 12M ops. Might be too slow if called often.
        # We will only use this when stuck.
        return best_s, best_a # Placeholder, we'll implement a smarter version if needed

    # --- Direct construction strategy ---
    # The sums a+b for a in A, b in B. We want 9 distinct sums, all pairwise coprime.
    # A powerful strategy: pick A and B such that all 9 sums are distinct primes.
    
    _prime_cache = {}
    def _is_prime(n):
        if n in _prime_cache:
            return _prime_cache[n]
        if n < 2:
            _prime_cache[n] = False
            return False
        if n < 4:
            _prime_cache[n] = True
            return True
        if n % 2 == 0 or n % 3 == 0:
            _prime_cache[n] = False
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                _prime_cache[n] = False
                return False
            i += 6
        _prime_cache[n] = True
        return True

    def all_distinct_primes(A, B):
        sums = []
        for a in A:
            for b in B:
                s = a + b
                if s <= 2 or not _is_prime(s):
                    return False
                if s in sums:
                    return False
                sums.append(s)
        return True

    # Try to find A, B where all 9 sums are distinct primes.
    # Phase 1: Deterministic scan over small anchor sets.
    found_prime_9 = False
    small_range = list(range(1, 25)) # Expanded range
    small_sets = list(combinations(small_range, 3))
    # To speed up, we can filter B for a given A
    for A in small_sets:
        if time.time() >= DEADLINE - 10:
            break
        A = list(A)
        for B in small_sets:
            B = list(B)
            if all_distinct_primes(A, B):
                return [A, B]

    # Phase 2: Random search with local improvement
    rng = random.Random(42)
    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]

    def hill_climb(A, B, time_limit):
        """Iterated local search. Returns improved (score, A, B)."""
        t = time.time()
        A_cur = A[:]
        B_cur = B[:]
        s = valid_pair(A_cur, B_cur)
        best_local_s = s
        best_local_A = A_cur[:]
        best_local_B = B_cur[:]

        while time.time() < time_limit and time.time() < DEADLINE:
            # Try to improve A
            s_new, A_new = best_move_A(A_cur, B_cur)
            if s_new > s:
                s = s_new
                A_cur = A_new
                if s > best_local_s:
                    best_local_s = s
                    best_local_A = A_cur[:]
                    best_local_B = B_cur[:]
                    if s >= 9:
                        return s, A_cur, B_cur
                continue
            # Try to improve B
            s_new, B_new = best_move_B(A_cur, B_cur)
            if s_new > s:
                s = s_new
                B_cur = B_new
                if s > best_local_s:
                    best_local_s = s
                    best_local_A = A_cur[:]
                    best_local_B = B_cur[:]
                    if s >= 9:
                        return s, A_cur, B_cur
                continue
            # Stuck. Perturb.
            if rng.random() < 0.5:
                i = rng.randrange(3)
                while True:
                    v = rng.randint(1, 128)
                    if v not in A_cur:
                        break
                A_cur[i] = v
            else:
                j = rng.randrange(3)
                while True:
                    v = rng.randint(1, 128)
                    if v not in B_cur:
                        break
                B_cur[j] = v
            s = valid_pair(A_cur, B_cur)
            if s > best_local_s:
                best_local_s = s
                best_local_A = A_cur[:]
                best_local_B = B_cur[:]
                if s >= 9:
                    return s, A_cur, B_cur

        return best_local_s, best_local_A, best_local_B

    restarts = 0
    while time.time() < DEADLINE - 5:
        A = sorted(rng.sample(range(1, 129), 3))
        B = sorted(rng.sample(range(1, 129), 3))
        s = valid_pair(A, B)
        if s > best_score:
            best_score = s
            best_A, best_B = A[:], B[:]
        if s >= 9:
            break
        # Local search with a time budget
        per_climb = min(2.0, (DEADLINE - time.time()) * 0.1)
        s2, A2, B2 = hill_climb(A, B, time.time() + per_climb)
        if s2 > best_score:
            best_score = s2
            best_A, best_B = A2[:], B2[:]
        if best_score >= 9:
            break
        restarts += 1

    # Final verification
    final_s = valid_pair(best_A, best_B)
    return [best_A, best_B]
