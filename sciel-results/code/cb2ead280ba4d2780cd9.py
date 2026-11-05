import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000

    # State
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
        if current_L > 0 and np.min(counts[1:current_L + 1]) < 2:
            bad = np.where(counts[1:current_L + 1] < 2)[0]
            return bad[0]
        if current_L < MAX_L:
            rest = counts[current_L + 1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) == 0:
                return MAX_L
            return current_L + bad_idx[0]
        return current_L

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

    # --- Phase 1: Submodular Greedy Initialization ---
    # For each prime, choose the residue that maximizes marginal gain in doubly-covered positions
    GREEDY_WINDOW = 8000
    greedy_counts = np.zeros(GREEDY_WINDOW + 1, dtype=np.int32)
    
    for idx in range(n_primes):
        p = primes[idx]
        best_r = 0
        best_score = -10**9
        
        # For small primes, try all residues; for large primes, sample
        if p <= 100:
            candidates = list(range(p))
        elif p <= 500:
            candidates = list(range(p))
        else:
            # Sample residues, but always include 0 and some strategic ones
            candidates = [0]
            # Add residues that cover early positions
            for r in range(1, min(p, 30)):
                candidates.append(r)
            # Add random samples
            extra = np.random.choice(p, size=min(p, 60), replace=False)
            for r in extra:
                if r not in candidates:
                    candidates.append(r)
        
        for r in candidates:
            start = r if r > 0 else p
            if start > GREEDY_WINDOW:
                continue
            positions = np.arange(start, GREEDY_WINDOW + 1, p)
            # Marginal gain: positions that go from <2 to >=2
            was_deficient = (greedy_counts[positions] < 2)
            now_covered = (greedy_counts[positions] + 1 >= 2)
            gain = np.sum(was_deficient & now_covered)
            # Penalty for over-covering (wasting coverage on already-doubly-covered positions)
            over_cover = np.sum(greedy_counts[positions] >= 2)
            score = gain - 0.2 * over_cover
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[idx] = best_r
        r = best_r
        start = r if r > 0 else p
        if start <= GREEDY_WINDOW:
            greedy_counts[start::p] += 1

    build_counts()
    current_L = find_L(0)
    best_L = current_L
    best_residues = residues.copy()

    # --- Phase 2: Targeted Greedy Extension ---
    # Fix the next uncovered position by finding a prime whose residue can be changed
    # to cover it, without breaking existing coverage
    while current_L < MAX_L and time.time() < deadline - 35.0:
        x = current_L + 1
        candidates = []
        for i, p in enumerate(primes):
            target_r = x % p
            if residues[i] == target_r:
                continue
            old_r = residues[i]
            # Check if changing this prime to target_r would break coverage
            # in the range [1, x]
            if old_r == 0:
                old_positions = np.arange(p, x + 1, p)
            else:
                old_positions = np.arange(old_r, x + 1, p) if old_r > 0 else np.arange(p, x + 1, p)
            
            if len(old_positions) == 0:
                candidates.append((i, target_r, 0))
                continue
            
            # Check minimum coverage in old positions after removal
            min_cov = np.min(counts[old_positions])
            if min_cov >= 3:
                # Safe to change
                candidates.append((i, target_r, 0))
            elif min_cov == 2:
                # Risky but might work if the new position adds coverage elsewhere
                candidates.append((i, target_r, 1))

        if candidates:
            # Prefer safe candidates
            safe = [c for c in candidates if c[2] == 0]
            if safe:
                i, new_r, _ = random.choice(safe)
            else:
                i, new_r, _ = random.choice(candidates)
            
            apply_change(i, new_r)
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
    T = 8.0
    cooling_rate = 0.99997
    min_T = 0.1

    while time.time() < deadline - 15.0:
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
            if delta > -500:
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

    # --- Phase 4: Local Search Polish ---
    if time.time() < deadline - 5.0:
        polish_deadline = deadline - 2.0
        for _ in range(30):
            if time.time() > polish_deadline:
                break
            progress = False
            for i in range(n_primes):
                if time.time() > polish_deadline:
                    break
                p = primes[i]
                old_r = residues[i]
                # Try nearby residues
                for offset in [-3, -2, -1, 1, 2, 3]:
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

    # Final evaluation
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
        final_L = bad[0]
    else:
        final_L = best_L
        if final_L < MAX_L:
            rest = final_counts[final_L + 1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) > 0:
                final_L = final_L + bad_idx[0]

    return best_residues.tolist()
