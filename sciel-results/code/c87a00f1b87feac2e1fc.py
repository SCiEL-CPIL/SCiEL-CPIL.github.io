import time
import random
import numpy as np

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)

def _build_count_np(residues, L_MAX):
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = int(residues[i])
        start = p if r == 0 else r
        count[start::p] += 1
    return count

def _find_prefix_L_np(count, L_MAX):
    if L_MAX < 1:
        return 0
    # Find first index where count < 2
    below = np.where(count[1:L_MAX+1] < 2)[0]
    if len(below) == 0:
        return L_MAX
    return below[0]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Smart initialization: Fix small primes to maximize early coverage density
    # For small primes, try residues that cover many small numbers
    random.seed(42)
    residues = [0] * _NPRIMES

    # For the first ~20 primes (p <= 71), try to maximize coverage of [1, 500]
    # Greedy: for each small prime, pick residue that maximizes count of x in [1,500] with count[x]==1
    L_INIT = 500
    count_init = np.zeros(L_INIT + 1, dtype=np.int32)

    for i, p in enumerate(_PRIMES[:20]):
        best_r = 0
        best_score = -1
        for r in range(p):
            start = p if r == 0 else r
            # Count how many x in [1, L_INIT] with x%p==r and count_init[x]==1
            if start <= L_INIT:
                positions = np.arange(start, L_INIT + 1, p)
                score = np.sum(count_init[positions] == 1)
            else:
                score = 0
            # Also consider if this helps cover gaps
            if score > best_score:
                best_score = score
                best_r = r
        residues[i] = best_r
        # Update count_init
        r = best_r
        start = p if r == 0 else r
        if start <= L_INIT:
            count_init[start::p] += 1

    # For remaining primes, use random initialization
    for i in range(20, _NPRIMES):
        p = _PRIMES[i]
        residues[i] = random.randint(0, p - 1)

    # Build initial count with numpy
    L_LIMIT = 20000
    count = _build_count_np(residues, L_LIMIT)
    current_L = _find_prefix_L_np(count, L_LIMIT)

    best_L = current_L
    best_residues = [int(r) for r in residues]

    # Main Loop: Hybrid Greedy + SA with numpy acceleration
    stagnation_count = 0
    max_stagnation = 30

    while time.time() < deadline - 10:
        # Check if we can extend L_LIMIT if we are close to it
        if current_L > L_LIMIT - 1000:
            new_limit = L_LIMIT * 2
            count = _build_count_np(residues, new_limit)
            L_LIMIT = new_limit
            current_L = _find_prefix_L_np(count, L_LIMIT)

        # 1. Greedy Repair Phase
        improved = False
        gap = current_L + 1

        if gap > L_LIMIT:
            # Expand window
            new_limit = L_LIMIT * 2
            count = _build_count_np(residues, new_limit)
            L_LIMIT = new_limit
            current_L = _find_prefix_L_np(count, L_LIMIT)
            continue

        if count[gap] >= 2:
            # Extend L
            while gap <= L_LIMIT and count[gap] >= 2:
                current_L = gap
                gap += 1
            if current_L > best_L:
                best_L = current_L
                best_residues = [int(r) for r in residues]
            improved = True
            stagnation_count = 0
            continue

        # Find a move to cover 'gap'
        # Focus on primes that could efficiently cover the gap
        # Prioritize smaller primes for higher frequency coverage
        best_move = None
        min_cost = float('inf')

        # Iterate primes in order of increasing p (smaller primes first)
        for i, p in enumerate(_PRIMES):
            old_r = int(residues[i])
            new_r = gap % p

            if old_r == new_r:
                continue

            # Calculate cost using numpy
            # How many x in [1, current_L] with x%p==old_r and count[x]==2?
            if old_r == 0:
                start_x = p
            else:
                start_x = old_r

            if start_x > current_L:
                cost = 0
            else:
                positions = np.arange(start_x, current_L + 1, p)
                cost = int(np.sum(count[positions] == 2))

            if cost < min_cost:
                min_cost = cost
                best_move = (i, new_r)
                if cost == 0:
                    break

        if best_move is not None:
            i, new_r = best_move
            p = _PRIMES[i]
            old_r = int(residues[i])

            # Apply move using numpy
            if old_r == 0:
                start_old = p
            else:
                start_old = old_r
            if new_r == 0:
                start_new = p
            else:
                start_new = new_r

            count[start_old::p] -= 1
            count[start_new::p] += 1

            residues[i] = new_r

            new_L = _find_prefix_L_np(count, L_LIMIT)

            if new_L > current_L:
                current_L = new_L
                improved = True
                stagnation_count = 0
                if current_L > best_L:
                    best_L = current_L
                    best_residues = [int(r) for r in residues]
            else:
                # Revert
                count[start_new::p] -= 1
                count[start_old::p] += 1
                residues[i] = old_r
                improved = False
                stagnation_count += 1

        if not improved:
            stagnation_count += 1

        # 2. SA Burst if Stagnant
        if stagnation_count > max_stagnation:
            # Perform SA for a short time with numpy
            T = 5.0
            cooling = 0.99
            min_T = 0.5
            sa_start_time = time.time()

            while time.time() < sa_start_time + 5.0 and T > min_T:
                i = random.randint(0, _NPRIMES - 1)
                p = _PRIMES[i]
                old_r = int(residues[i])
                new_r = random.randint(0, p - 1)

                if new_r == old_r:
                    continue

                if old_r == 0:
                    start_old = p
                else:
                    start_old = old_r
                if new_r == 0:
                    start_new = p
                else:
                    start_new = new_r

                count[start_old::p] -= 1
                count[start_new::p] += 1
                residues[i] = new_r

                new_L = _find_prefix_L_np(count, L_LIMIT)

                delta = new_L - current_L
                if delta > 0:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = [int(r) for r in residues]
                        stagnation_count = 0
                else:
                    if delta == 0:
                        pass
                    else:
                        prob = 2.718281828 ** (delta / T)
                        if random.random() > prob:
                            count[start_new::p] -= 1
                            count[start_old::p] += 1
                            residues[i] = old_r
                        else:
                            current_L = new_L

                T *= cooling

            stagnation_count = 0

    # Final Verification
    L_MAX_FINAL = 100000
    count_final = _build_count_np(best_residues, L_MAX_FINAL)
    final_L = _find_prefix_L_np(count_final, L_MAX_FINAL)

    return [int(r) for r in best_residues]
