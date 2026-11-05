"""
Rewritten program using a structural change: constraint-satisfaction via
deficient-position targeting and greedy repair, combined with multi-start
local search that explicitly tracks the set of deficient positions.
"""
import time
import random
import numpy as np

_DEADLINE = 142.0  # safety margin before 150s wall-clock limit


def _primes_up_to(n):
    """Sieve of Eratosthenes for primes up to n."""
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE

    primes = _primes_up_to(1000)
    n = len(primes)
    MAX_L = 100000

    # Precompute: for each prime index, the list of positions 1..MAX_L it covers
    # for each possible residue (too large to precompute fully, so we use the
    # coverage array approach)

    def build_coverage(residues):
        """Build the coverage array from scratch."""
        cov = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r != 0 else p
            if start <= MAX_L:
                cov[start::p] += 1
        return cov

    def find_first_gap(cov):
        """Find the first x >= 1 where cov[x] < 2."""
        bad = np.where(cov[1:] < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return int(bad[0])

    def update_coverage(cov, idx, old_r, new_r):
        """Update coverage array in-place when changing residue at idx."""
        p = primes[idx]
        old_start = old_r if old_r != 0 else p
        if old_start <= MAX_L:
            cov[old_start::p] -= 1
        new_start = new_r if new_r != 0 else p
        if new_start <= MAX_L:
            cov[new_start::p] += 1

    def get_deficient_positions(cov, up_to):
        """Get positions in [1, up_to] where coverage < 2."""
        if up_to <= 1:
            return np.array([], dtype=np.int32)
        end = min(up_to, MAX_L)
        return np.where(cov[1:end+1] < 2)[0] + 1

    best_L_global = 0
    best_residues_global = [0] * n

    def greedy_repair(residues, coverage, current_L, max_iter=200):
        """
        Greedily fix deficient positions by reassigning residues.
        For each deficient position x, find a prime that can cover it
        without breaking other coverage, and reassign.
        """
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline - 5:
            improved = False
            iters += 1
            target = current_L + 1
            if target > MAX_L:
                break

            # Find all primes whose current residue does NOT cover target
            # and check if reassigning to cover target improves things
            best_gain = 0
            best_idx = -1
            best_r = -1

            for idx in range(n):
                p = primes[idx]
                target_r = target % p
                if residues[idx] == target_r:
                    continue

                # Try reassigning
                old_r = residues[idx]
                update_coverage(coverage, idx, old_r, target_r)
                new_L = find_first_gap(coverage)
                gain = new_L - current_L
                update_coverage(coverage, idx, target_r, old_r)

                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_r = target_r

            if best_idx >= 0 and best_gain > 0:
                update_coverage(coverage, best_idx, residues[best_idx], best_r)
                residues[best_idx] = best_r
                current_L += best_gain
                improved = True

        return current_L

    def local_search_deficient(residues, coverage, current_L, max_passes=50):
        """
        Local search that targets deficient positions.
        For each deficient position, try reassigning primes to cover it.
        """
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 5:
            improved = False
            passes += 1

            # Get deficient positions up to current_L + 500
            check_end = min(current_L + 500, MAX_L)
            deficient = get_deficient_positions(coverage, check_end)

            if len(deficient) == 0:
                # No deficient positions in range, check if we can extend
                new_L = find_first_gap(coverage)
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                continue

            # Process deficient positions in order (earliest first)
            for x in deficient[:100]:
                if time.time() > deadline - 5:
                    break

                # Try to cover x by reassigning a prime
                best_gain = 0
                best_idx = -1
                best_r = -1

                # Prefer primes that don't currently cover x
                for idx in range(n):
                    p = primes[idx]
                    target_r = x % p
                    if residues[idx] == target_r:
                        continue

                    old_r = residues[idx]
                    update_coverage(coverage, idx, old_r, target_r)
                    new_L = find_first_gap(coverage)
                    gain = new_L - current_L
                    update_coverage(coverage, idx, target_r, old_r)

                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        best_r = target_r

                if best_idx >= 0 and best_gain > 0:
                    update_coverage(coverage, best_idx, residues[best_idx], best_r)
                    residues[best_idx] = best_r
                    current_L += best_gain
                    improved = True

        return current_L

    def local_search_random(residues, coverage, current_L, max_passes=30):
        """Standard random local search."""
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 5:
            improved = False
            passes += 1
            shuffled = list(range(n))
            random.shuffle(shuffled)

            for idx in shuffled:
                if time.time() > deadline - 5:
                    break

                p = primes[idx]
                old_r = residues[idx]
                best_r_for_idx = old_r
                best_L_for_idx = current_L

                # Generate candidates
                candidates = set()
                candidates.add(old_r)

                target = current_L + 1
                if target <= MAX_L:
                    candidates.add(target % p)

                # Random candidates
                num_random = min(30, p)
                for _ in range(num_random):
                    candidates.add(random.randint(0, p - 1))

                # Neighbors
                candidates.add((old_r + 1) % p)
                candidates.add((old_r - 1) % p)

                if p <= 30:
                    candidates = set(range(p))

                for r in candidates:
                    if r == old_r:
                        continue
                    if time.time() > deadline - 5:
                        break

                    update_coverage(coverage, idx, old_r, r)
                    new_L = find_first_gap(coverage)
                    update_coverage(coverage, idx, r, old_r)

                    if new_L > best_L_for_idx:
                        best_L_for_idx = new_L
                        best_r_for_idx = r

                if best_r_for_idx != old_r:
                    update_coverage(coverage, idx, old_r, best_r_for_idx)
                    residues[idx] = best_r_for_idx
                    current_L = best_L_for_idx
                    improved = True

        return current_L

    def init_greedy_frontier():
        """Initialize by greedily covering the frontier."""
        r = [0] * n
        # For each position 1..3000, ensure at least 2 primes cover it
        for x in range(1, 3001):
            count = 0
            for i, p in enumerate(primes):
                if (x - r[i]) % p == 0:
                    count += 1
            while count < 2:
                assigned = False
                for i in range(n):
                    p = primes[i]
                    # Only assign if this prime doesn't already cover x
                    if (x - r[i]) % p != 0:
                        r[i] = x % p
                        count += 1
                        assigned = True
                        break
                if not assigned:
                    break
        for i in range(n):
            if r[i] == 0:
                r[i] = random.randint(0, primes[i] - 1)
        return r

    def init_random():
        """Random initialization."""
        return [random.randint(0, p - 1) for p in primes]

    def init_perturbed(base):
        """Perturb a base solution."""
        r = base[:]
        for _ in range(20):
            idx = random.randint(0, n - 1)
            r[idx] = random.randint(0, primes[idx] - 1)
        return r

    def init_clustered():
        """Initialize with clustered residues for small primes."""
        r = [0] * n
        for i, p in enumerate(primes):
            if p <= 100:
                r[i] = random.randint(0, min(10, p - 1))
            else:
                r[i] = random.randint(0, p - 1)
        return r

    # Multi-start search
    initializers = [
        init_greedy_frontier,
        init_random,
        init_clustered,
        init_greedy_frontier,
        init_random,
    ]

    for init_idx, init_func in enumerate(initializers):
        if time.time() > deadline - 20:
            break

        residues = init_func()
        coverage = build_coverage(residues)
        current_L = find_first_gap(coverage)

        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]

        # Phase 1: Greedy repair
        current_L = greedy_repair(residues, coverage, current_L, max_iter=100)
        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]

        # Phase 2: Deficient-position local search
        current_L = local_search_deficient(residues, coverage, current_L, max_passes=40)
        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]

        # Phase 3: Random local search
        current_L = local_search_random(residues, coverage, current_L, max_passes=20)
        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]

        # Phase 4: More greedy repair
        current_L = greedy_repair(residues, coverage, current_L, max_iter=100)
        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]

        # Phase 5: Another round of deficient search
        current_L = local_search_deficient(residues, coverage, current_L, max_passes=30)
        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]

    # Final: try to extend from the best solution
    if best_L_global > 0 and time.time() < deadline - 5:
        residues = best_residues_global[:]
        coverage = build_coverage(residues)
        current_L = find_first_gap(coverage)

        # Aggressive frontier extension
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            target = current_L + 1
            if target > MAX_L:
                break

            best_gain = 0
            best_idx = -1
            best_r = -1

            for idx in range(n):
                p = primes[idx]
                target_r = target % p
                if residues[idx] == target_r:
                    continue

                old_r = residues[idx]
                update_coverage(coverage, idx, old_r, target_r)
                new_L = find_first_gap(coverage)
                gain = new_L - current_L
                update_coverage(coverage, idx, target_r, old_r)

                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    best_r = target_r

            if best_idx >= 0 and best_gain > 0:
                update_coverage(coverage, best_idx, residues[best_idx], best_r)
                residues[best_idx] = best_r
                current_L += best_gain
                if current_L > best_L_global:
                    best_L_global = current_L
                    best_residues_global = residues[:]
            else:
                break

    return [int(r) for r in best_residues_global]
