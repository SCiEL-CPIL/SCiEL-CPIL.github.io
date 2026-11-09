import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill

    n_max = 256
    candidates = list(range(2, n_max + 1))

    # Precompute prime factorizations
    def prime_factorize(num):
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors

    factorizations = {}
    for i in range(2, n_max + 1):
        factorizations[i] = prime_factorize(i)

    # Precompute gcd table for speed
    gcd_table = [[0] * (n_max + 1) for _ in range(n_max + 1)]
    for i in range(2, n_max + 1):
        for j in range(i, n_max + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    def divides_product(a, b, c):
        """Check if a divides b*c using factorizations"""
        for p, exp_a in factorizations[a].items():
            exp_bc = factorizations[b].get(p, 0) + factorizations[c].get(p, 0)
            if exp_bc < exp_a:
                return False
        return True

    def can_add_fast(S_list, S_set, x):
        """Check if S ∪ {x} is valid. Only need to check triples involving x.
        Uses gcd-based fast check instead of full factorization."""
        m = len(S_list)
        if m < 2:
            return True
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                # Check a | x*b
                d1 = a // gcd_table[a][x]
                if d1 == 1:
                    return False
                if b % d1 == 0:
                    return False
                # Check b | x*a
                d2 = b // gcd_table[b][x]
                if d2 == 1:
                    return False
                if a % d2 == 0:
                    return False
                # Check x | a*b
                d3 = x // gcd_table[x][a]
                if d3 == 1:
                    return False
                if b % d3 == 0:
                    return False
                d4 = x // gcd_table[x][b]
                if d4 == 1:
                    return False
                if a % d4 == 0:
                    return False
        return True

    def is_valid(S):
        """Full O(n^3) validation."""
        S_list = sorted(S)
        m = len(S_list)
        if m < 3:
            return True
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                d = a // gcd_table[a][b]
                if d == 1:
                    return False
                for k in range(m):
                    if k == i or k == j:
                        continue
                    if S_list[k] % d == 0:
                        return False
        return True

    def greedy_fill(S_list, S_set, order):
        """Greedily add elements from order to S."""
        for x in order:
            if time.time() > deadline - 3:
                break
            if x in S_set:
                continue
            if can_add_fast(S_list, S_set, x):
                S_list.append(x)
                S_set.add(x)
        return S_list, S_set

    def try_2_for_1_swap(current_list, current_set):
        """Try removing one element y and adding two elements x1, x2.
        Returns new list/set if successful, else None."""
        n = len(current_list)
        # Try removing each element
        for idx in range(n):
            if time.time() > deadline - 3:
                return None
            y = current_list[idx]
            trial_list = [z for z in current_list if z != y]
            trial_set = set(trial_list)
            
            # Try to add two elements
            added = 0
            new_list = trial_list[:]
            new_set = trial_set.copy()
            
            for x in candidates:
                if x in new_set:
                    continue
                if can_add_fast(new_list, new_set, x):
                    new_list.append(x)
                    new_set.add(x)
                    added += 1
                    if added >= 2:
                        break
            
            if added >= 2:
                return new_list, new_set
        return None

    def try_1_for_1_swap(current_list, current_set):
        """Try removing one element y and adding one element x (same size).
        This might enable further additions later."""
        n = len(current_list)
        for idx in range(n):
            if time.time() > deadline - 3:
                return None
            y = current_list[idx]
            trial_list = [z for z in current_list if z != y]
            trial_set = set(trial_list)
            
            for x in candidates:
                if x in trial_set:
                    continue
                if can_add_fast(trial_list, trial_set, x):
                    new_list = trial_list + [x]
                    new_set = trial_set | {x}
                    # Try to greedily fill more
                    new_list, new_set = greedy_fill(new_list, new_set, candidates)
                    if len(new_list) > len(current_list):
                        return new_list, new_set
        return None

    best_S = None
    best_size = 0

    # Phase 1: Start with all large numbers [129, 256]
    S = list(range(129, 257))
    S_set = set(S)
    S_list = S[:]

    if is_valid(S_list):
        best_S = S_list[:]
        best_size = len(S_list)

    # Phase 2: Greedily add small numbers in increasing order
    if time.time() < deadline:
        S_list, S_set = greedy_fill(S_list, S_set, range(2, 129))
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_S = S_list[:]

    # Phase 3: Try a second pass with decreasing small numbers
    if time.time() < deadline - 10:
        S2 = list(range(129, 257))
        S2_set = set(S2)
        S2_list = S2[:]
        S2_list, S2_set = greedy_fill(S2_list, S2_set, range(128, 1, -1))
        if len(S2_list) > best_size:
            best_size = len(S2_list)
            best_S = S2_list[:]

    # Phase 4: Try starting with [120, 256]
    if time.time() < deadline - 10:
        S3 = list(range(120, 257))
        S3_set = set(S3)
        S3_list = S3[:]
        if is_valid(S3_list):
            S3_list, S3_set = greedy_fill(S3_list, S3_set, range(2, 120))
            if len(S3_list) > best_size:
                best_size = len(S3_list)
                best_S = S3_list[:]

    # Phase 5: Try [100, 256]
    if time.time() < deadline - 10:
        S4 = list(range(100, 257))
        S4_set = set(S4)
        S4_list = S4[:]
        if is_valid(S4_list):
            S4_list, S4_set = greedy_fill(S4_list, S4_set, range(2, 100))
            if len(S4_list) > best_size:
                best_size = len(S4_list)
                best_S = S4_list[:]

    # Phase 6: Randomized restarts with different ordering strategies
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        current_set = set()
        current_list = []
        
        # Different ordering strategies
        if r % 4 == 0:
            shuffled = candidates.copy()
            random.shuffle(shuffled)
        elif r % 4 == 1:
            shuffled = sorted(candidates, key=lambda x: (len(factorizations[x]), random.random()))
        elif r % 4 == 2:
            shuffled = sorted(candidates, key=lambda x: (-x, random.random()))
        else:
            shuffled = sorted(candidates, key=lambda x: (-max(factorizations[x].keys()) if factorizations[x] else 0, random.random()))
        
        for a in shuffled:
            if time.time() > deadline - 15:
                break
            if can_add_fast(current_list, current_set, a):
                current_list.append(a)
                current_set.add(a)
        
        if len(current_list) > best_size:
            best_size = len(current_list)
            best_S = current_list[:]

    # Phase 7: Local improvement with swaps
    if best_S is not None and time.time() < deadline - 10:
        current = best_S[:]
        current_set = set(current)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try 2-for-1 swaps
            result = try_2_for_1_swap(current, current_set)
            if result is not None:
                current, current_set = result
                if len(current) > best_size:
                    best_size = len(current)
                    best_S = current[:]
                improved = True
                continue
            
            # Try 1-for-1 swaps that enable growth
            result = try_1_for_1_swap(current, current_set)
            if result is not None:
                current, current_set = result
                if len(current) > best_size:
                    best_size = len(current)
                    best_S = current[:]
                improved = True

    # Final validation
    if best_S is None:
        is_prime = [True] * (n_max + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(n_max**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n_max + 1, i):
                    is_prime[j] = False
        best_S = [i for i in range(2, n_max + 1) if is_prime[i]]

    if not is_valid(best_S):
        is_prime = [True] * (n_max + 1)
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(n_max**0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n_max + 1, i):
                    is_prime[j] = False
        best_S = [i for i in range(2, n_max + 1) if is_prime[i]]

    return list(best_S)
