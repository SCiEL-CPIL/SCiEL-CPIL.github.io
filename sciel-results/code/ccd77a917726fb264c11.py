"""Combined solver: greedy forward construction + vectorized backward refinement + local search."""
import time
import random
import numpy as np

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def _build_counts(residues, primes, max_L):
    """Build coverage count array for positions 1..max_L."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for i in range(len(primes)):
        r = residues[i]
        p = primes[i]
        r_eff = p if r == 0 else r
        if r_eff <= max_L:
            positions = np.arange(r_eff, max_L + 1, p)
            if len(positions) > 0:
                counts[positions] += 1
    return counts

def _find_L(counts, max_L):
    """Find largest L such that all positions 1..L have count >= 2."""
    deficient = np.where(counts[1:max_L+1] < 2)[0]
    if len(deficient) == 0:
        return max_L
    return int(deficient[0])

def _score(residues, primes, max_L):
    counts = _build_counts(residues, primes, max_L)
    return _find_L(counts, max_L)

def _greedy_construct(primes, max_L_check=5000):
    """
    Forward greedy: process primes in order, for each prime choose the residue
    that maximizes coverage of currently-deficient positions.
    """
    n = len(primes)
    residues = [0] * n
    counts = np.zeros(max_L_check + 1, dtype=np.int32)

    for idx in range(n):
        p = primes[idx]

        search_range = min(max_L_check, 500)
        deficient_positions = []
        for pos in range(1, search_range + 1):
            if counts[pos] < 2:
                deficient_positions.append(pos)
                if len(deficient_positions) >= 100:
                    break

        if not deficient_positions:
            best_r = 0
            best_count = -1
            for r in range(min(p, 200)):
                r_eff = p if r == 0 else r
                if r_eff > max_L_check:
                    continue
                num_positions = (max_L_check - r_eff) // p + 1
                if num_positions > best_count:
                    best_count = num_positions
                    best_r = r
        else:
            residue_weighted = np.zeros(p, dtype=np.int32)
            for pos in deficient_positions:
                r_needed = pos % p
                weight = 3 if counts[pos] == 0 else 1
                residue_weighted[r_needed] += weight
            best_r = int(np.argmax(residue_weighted))

        residues[idx] = best_r

        r = best_r
        r_eff = p if r == 0 else r
        if r_eff <= max_L_check:
            positions = np.arange(r_eff, max_L_check + 1, p)
            if len(positions) > 0:
                counts[positions] += 1

    return residues

def _backward_refine(residues, primes, max_L_check, deadline, num_passes=3):
    """
    Backward refinement: iterate through primes in reverse, for each prime
    try all residues (or a subset for large primes) and keep the one that gives best score.
    Uses incremental count updates for speed.
    """
    n = len(primes)
    best_score = _score(residues, primes, max_L_check)

    for pass_num in range(num_passes):
        if time.time() > deadline - 5:
            break

        improved = False
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline - 5:
                break

            p = primes[idx]
            old_r = residues[idx]

            # Build counts without this prime
            counts = _build_counts(residues, primes, max_L_check)
            r_eff = p if old_r == 0 else old_r
            if r_eff <= max_L_check:
                positions = np.arange(r_eff, max_L_check + 1, p)
                if len(positions) > 0:
                    counts[positions] -= 1

            # Baseline score without this prime
            base_score = _find_L(counts, max_L_check)
            best_r = old_r
            best_local_score = base_score

            # For large primes, limit candidates
            if p <= 200:
                candidates = list(range(p))
            else:
                # Sample candidates: include 0, and positions near deficient points
                # Get deficient positions
                def_pos = np.where(counts[1:max_L_check+1] < 2)[0]
                if len(def_pos) == 0:
                    candidates = list(range(p))
                else:
                    # Focus on first 500 deficient positions
                    focus = def_pos[:500]
                    candidate_set = set()
                    candidate_set.add(0)
                    for pos in focus:
                        candidate_set.add(int(pos % p))
                    candidates = list(candidate_set)
                    if len(candidates) > 300:
                        random.seed(42 + idx + pass_num)
                        candidates = random.sample(candidates, 300)
                    if old_r not in candidates:
                        candidates.append(old_r)

            for new_r in candidates:
                if time.time() > deadline - 5:
                    break

                r_eff_new = p if new_r == 0 else new_r
                if r_eff_new <= max_L_check:
                    positions = np.arange(r_eff_new, max_L_check + 1, p)
                    if len(positions) > 0:
                        counts[positions] += 1

                new_score = _find_L(counts, max_L_check)

                if new_score > best_local_score:
                    best_local_score = new_score
                    best_r = new_r

                if r_eff_new <= max_L_check:
                    positions = np.arange(r_eff_new, max_L_check + 1, p)
                    if len(positions) > 0:
                        counts[positions] -= 1

            if best_r != old_r:
                residues[idx] = best_r
                improved = True

        if not improved:
            break

    return residues

def _local_search(residues, primes, max_L_check, deadline):
    """
    Local search: randomly perturb residues and accept improvements.
    Uses incremental count updates.
    """
    n = len(primes)
    best_res = residues[:]
    best_L = _score(best_res, primes, max_L_check)
    no_improve_count = 0
    max_no_improve = 50

    random.seed(42)

    while time.time() < deadline - 10 and no_improve_count < max_no_improve:
        improved = False
        indices = list(range(n))
        random.shuffle(indices)

        for i in indices:
            if time.time() > deadline - 10:
                break
            p = primes[i]
            old_r = best_res[i]

            # Build current counts
            counts = _build_counts(best_res, primes, max_L_check)
            r_eff = p if old_r == 0 else old_r
            if r_eff <= max_L_check:
                positions = np.arange(r_eff, max_L_check + 1, p)
                if len(positions) > 0:
                    counts[positions] -= 1

            best_local_L = best_L
            best_local_r = old_r

            # Try a few random new values
            num_tries = min(p, 20)
            for _ in range(num_tries):
                if time.time() > deadline - 10:
                    break
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue

                r_eff_new = p if new_r == 0 else new_r
                if r_eff_new <= max_L_check:
                    positions = np.arange(r_eff_new, max_L_check + 1, p)
                    if len(positions) > 0:
                        counts[positions] += 1

                new_L = _find_L(counts, max_L_check)

                if new_L > best_local_L:
                    best_local_L = new_L
                    best_local_r = new_r

                if r_eff_new <= max_L_check:
                    positions = np.arange(r_eff_new, max_L_check + 1, p)
                    if len(positions) > 0:
                        counts[positions] -= 1

            if best_local_r != old_r:
                best_res[i] = best_local_r
                best_L = best_local_L
                improved = True

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

    return best_res, best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    primes = _get_primes(1000)
    n_primes = len(primes)

    # Phase 1: Greedy forward construction
    residues = _greedy_construct(primes, max_L_check=5000)

    # Phase 2: Backward refinement with a moderate max_L
    max_L_check = 20000
    residues = _backward_refine(residues, primes, max_L_check, deadline, num_passes=3)

    # Phase 3: Local search
    residues, current_L = _local_search(residues, primes, max_L_check, deadline)

    # Phase 4: Final backward refinement with larger max_L if time permits
    if time.time() < deadline - 20:
        max_L_check2 = min(50000, 100000)
        residues = _backward_refine(residues, primes, max_L_check2, deadline, num_passes=2)

    # Phase 5: Final local search with larger max_L if time permits
    if time.time() < deadline - 15:
        max_L_check3 = min(50000, 100000)
        residues, _ = _local_search(residues, primes, max_L_check3, deadline)

    # Final verification with full range
    final_L = _score(residues, primes, 100000)

    return residues
