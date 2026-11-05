import numpy as np
import time
import random
import math

def _primes_up_to(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    primes = _primes_up_to(1000)
    n_primes = len(primes)
    MAX_L = 100000

    residues = np.zeros(n_primes, dtype=np.int32)
    counts = np.zeros(MAX_L + 1, dtype=np.int8)

    def build_counts():
        counts[:] = 0
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                counts[p::p] += 1
            else:
                counts[r::p] += 1
        counts[0] = 100

    def find_L(current_L=0):
        if current_L >= MAX_L:
            return MAX_L
        if current_L > 0:
            min_cov = np.min(counts[1:current_L + 1])
            if min_cov < 2:
                bad = np.where(counts[1:current_L + 1] < 2)[0]
                return int(bad[0])
        rest = counts[current_L + 1:]
        bad_idx = np.where(rest < 2)[0]
        if len(bad_idx) == 0:
            return MAX_L
        return int(current_L + bad_idx[0])

    def apply_change(i, new_r):
        p = primes[i]
        old_r = residues[i]
        if old_r == new_r:
            return
        if old_r == 0:
            counts[p::p] -= 1
        else:
            counts[old_r::p] -= 1
        residues[i] = new_r
        if new_r == 0:
            counts[p::p] += 1
        else:
            counts[new_r::p] += 1

    def revert_change(i, old_r):
        p = primes[i]
        new_r = residues[i]
        if old_r == new_r:
            return
        if new_r == 0:
            counts[p::p] -= 1
        else:
            counts[new_r::p] -= 1
        residues[i] = old_r
        if old_r == 0:
            counts[p::p] += 1
        else:
            counts[old_r::p] += 1

    # --- Phase 1: Greedy Forward Construction ---
    # Build residues one prime at a time, choosing each to maximize the doubly-covered prefix
    # Use a temporary smaller range for speed during construction
    CONSTRUCT_MAX = 20000
    temp_counts = np.zeros(CONSTRUCT_MAX + 1, dtype=np.int8)

    for idx in range(n_primes):
        if time.time() > deadline - 60.0:
            break
        p = primes[idx]
        best_r = 0
        best_score = -1

        for r_candidate in range(p):
            # Add this residue to temp_counts
            if r_candidate == 0:
                temp_counts[p::p] += 1
            else:
                temp_counts[r_candidate::p] += 1

            # Find L from temp_counts
            deficient = np.where(temp_counts[1:CONSTRUCT_MAX + 1] < 2)[0]
            if len(deficient) == 0:
                score = CONSTRUCT_MAX
            else:
                score = int(deficient[0])

            # Remove
            if r_candidate == 0:
                temp_counts[p::p] -= 1
            else:
                temp_counts[r_candidate::p] -= 1

            if score > best_score:
                best_score = score
                best_r = r_candidate

        residues[idx] = best_r

    # Build full counts
    build_counts()
    current_L = find_L(0)
    best_L = current_L
    best_residues = residues.copy()

    # --- Phase 2: Targeted Greedy Extension ---
    # Try to extend L by finding primes that can safely cover L+1
    extension_rounds = 0
    max_extensions = 200
    while current_L < MAX_L and time.time() < deadline - 40.0 and extension_rounds < max_extensions:
        extension_rounds += 1
        x = current_L + 1
        candidates = []
        for i, p in enumerate(primes):
            target_r = x % p
            if residues[i] == target_r:
                continue
            old_r = residues[i]
            # Check if removing old_r coverage breaks any y in [1, x-1]
            if old_r == 0:
                if x <= p:
                    candidates.append(i)
                    continue
                if x > p:
                    min_cov = np.min(counts[p:x:p])
                    if min_cov >= 3:
                        candidates.append(i)
            else:
                if old_r >= x:
                    candidates.append(i)
                    continue
                if old_r > 0:
                    min_cov = np.min(counts[old_r:x:p])
                    if min_cov >= 3:
                        candidates.append(i)

        if candidates:
            i = random.choice(candidates)
            apply_change(i, x % primes[i])
            new_L = find_L(current_L)
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                old_r = residues[i]
                revert_change(i, old_r)
        else:
            break

    # --- Phase 3: Simulated Annealing ---
    if time.time() < deadline - 15.0:
        T = 5.0
        cooling_rate = 0.9999
        min_T = 0.05

        while time.time() < deadline - 10.0:
            i = np.random.randint(n_primes)
            p = primes[i]
            old_r = residues[i]
            new_r = np.random.randint(p)

            if new_r == old_r:
                continue

            apply_change(i, new_r)
            new_L = find_L(current_L)

            delta = new_L - current_L

            if delta > 0:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
            else:
                if delta > -200:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_L = new_L
                    else:
                        revert_change(i, old_r)
                else:
                    revert_change(i, old_r)

            T *= cooling_rate
            if T < min_T:
                T = min_T

    # --- Phase 4: Final Polish - targeted local search near frontier ---
    if time.time() < deadline - 3.0:
        polish_deadline = deadline - 1.0
        for _ in range(30):
            if time.time() > polish_deadline:
                break
            progress = False
            for i in range(n_primes):
                if time.time() > polish_deadline:
                    break
                p = primes[i]
                old_r = residues[i]
                for offset in [-3, -2, -1, 1, 2, 3, -7, 7, -13, 13]:
                    new_r = (old_r + offset) % p
                    if new_r == old_r:
                        continue
                    apply_change(i, new_r)
                    new_L = find_L(current_L)
                    if new_L > current_L:
                        current_L = new_L
                        if new_L > best_L:
                            best_L = new_L
                            best_residues = residues.copy()
                        progress = True
                        break
                    else:
                        revert_change(i, old_r)
            if not progress:
                break

    # Final verification
    final_counts = np.zeros(MAX_L + 1, dtype=np.int8)
    for i, p in enumerate(primes):
        r = best_residues[i]
        if r == 0:
            final_counts[p::p] += 1
        else:
            final_counts[r::p] += 1
    final_counts[0] = 100

    if np.min(final_counts[1:best_L + 1]) < 2:
        bad = np.where(final_counts[1:best_L + 1] < 2)[0]
        final_L = int(bad[0])
    else:
        final_L = best_L
        if final_L < MAX_L:
            rest = final_counts[final_L + 1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) > 0:
                final_L = int(final_L + bad_idx[0])

    return best_residues.tolist()
