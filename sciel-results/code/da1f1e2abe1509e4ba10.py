import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i in range(2, limit+1) if sieve[i]]

    primes = get_primes(1000)
    n = len(primes)  # 168

    # --- Fast numpy-based scoring ---
    def _build_count_array(residues, primes, max_L):
        """Build coverage count array for positions 1..max_L."""
        counts = np.zeros(max_L + 1, dtype=np.int32)
        for i in range(len(primes)):
            r = residues[i]
            p = primes[i]
            if r == 0:
                r_eff = p
            else:
                r_eff = r
            if r_eff <= max_L:
                positions = np.arange(r_eff, max_L + 1, p)
                counts[positions] += 1
        return counts

    def _find_L(counts, max_L):
        """Find largest L such that all positions 1..L have count >= 2."""
        # We need counts[1] through counts[L] all >= 2
        # Find first index in 1..max_L where count < 2
        # Use np.where on the slice
        deficient = np.where(counts[1:max_L+1] < 2)[0]
        if len(deficient) == 0:
            return max_L
        return int(deficient[0])

    def _score(residues, primes, max_L=20000):
        """Compute the largest L such that every integer 1..L is covered at least twice."""
        counts = _build_count_array(residues, primes, max_L)
        return _find_L(counts, max_L)

    def _score_full(residues, primes, max_L=100000):
        """Full range score for final verification."""
        counts = _build_count_array(residues, primes, max_L)
        return _find_L(counts, max_L)

    # --- Greedy forward construction ---
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

            # Find deficient positions in early range
            # Focus on positions 1..min(max_L_check, 500) for speed
            search_range = min(max_L_check, 500)
            deficient_positions = []
            for pos in range(1, search_range + 1):
                if counts[pos] < 2:
                    deficient_positions.append(pos)
                    if len(deficient_positions) >= 100:
                        break

            if not deficient_positions:
                # No deficient positions in early range
                # Pick residue that covers the most positions overall
                best_r = 0
                best_count = -1
                for r in range(min(p, 100)):
                    r_eff = p if r == 0 else r
                    if r_eff > max_L_check:
                        continue
                    num_positions = (max_L_check - r_eff) // p + 1
                    if num_positions > best_count:
                        best_count = num_positions
                        best_r = r
            else:
                # Weighted histogram: positions with count 0 get weight 3, count 1 get weight 1
                residue_weighted = np.zeros(p, dtype=np.int32)
                for pos in deficient_positions:
                    r_needed = pos % p
                    weight = 3 if counts[pos] == 0 else 1
                    residue_weighted[r_needed] += weight
                best_r = int(np.argmax(residue_weighted))

            residues[idx] = best_r

            # Update counts
            r = best_r
            r_eff = p if r == 0 else r
            if r_eff <= max_L_check:
                positions = np.arange(r_eff, max_L_check + 1, p)
                counts[positions] += 1

        return residues

    # --- Backward refinement ---
    def _backward_refine(residues, primes, max_L_check, deadline, num_passes=5):
        """
        Backward refinement: iterate through primes in reverse, for each prime
        try all residues and keep the one that gives best score.
        """
        n = len(primes)
        best_score = _score(residues, primes, max_L_check)

        for pass_num in range(num_passes):
            if time.time() > deadline:
                break

            improved = False
            # Process in reverse order
            for idx in range(n - 1, -1, -1):
                if time.time() > deadline:
                    break

                p = primes[idx]
                old_r = residues[idx]

                # Build counts without this prime
                counts = _build_count_array(residues, primes, max_L_check)
                r = old_r
                r_eff = p if r == 0 else r
                if r_eff <= max_L_check:
                    positions = np.arange(r_eff, max_L_check + 1, p)
                    counts[positions] -= 1

                # Now try each possible residue for this prime
                best_r = old_r
                best_local_score = _find_L(counts, max_L_check)

                # For large primes, we can limit candidates to speed up
                # But for small primes, try all
                if p <= 50:
                    candidates = list(range(p))
                elif p <= 200:
                    candidates = list(range(p))
                else:
                    # For large primes, try a subset: 0, and positions near deficient points
                    # Actually, let's just try all but break early if time is tight
                    candidates = list(range(p))

                for new_r in candidates:
                    if time.time() > deadline:
                        break

                    r_eff_new = p if new_r == 0 else new_r
                    if r_eff_new <= max_L_check:
                        positions = np.arange(r_eff_new, max_L_check + 1, p)
                        counts[positions] += 1

                    new_score = _find_L(counts, max_L_check)

                    if new_score > best_local_score:
                        best_local_score = new_score
                        best_r = new_r

                    # Subtract again for next candidate
                    if r_eff_new <= max_L_check:
                        positions = np.arange(r_eff_new, max_L_check + 1, p)
                        counts[positions] -= 1

                if best_r != old_r:
                    residues[idx] = best_r
                    improved = True

            if not improved:
                break

        return residues, best_score

    # --- Local search with random perturbations ---
    def _local_search(residues, primes, max_L_check, deadline):
        """
        Local search: randomly perturb residues and accept improvements.
        """
        n = len(primes)
        best_res = residues[:]
        best_L = _score(best_res, primes, max_L_check)
        no_improve_count = 0
        max_no_improve = 30

        random.seed(42)

        while time.time() < deadline:
            improved = False
            indices = list(range(n))
            random.shuffle(indices)

            for i in indices:
                if time.time() > deadline:
                    break
                p = primes[i]
                old_r = best_res[i]

                # Try a few random new values
                for _ in range(3):
                    new_r = random.randint(0, p - 1)
                    if new_r == old_r:
                        continue
                    best_res[i] = new_r
                    new_L = _score(best_res, primes, max_L_check)
                    if new_L > best_L:
                        best_L = new_L
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        best_res[i] = old_r

            if not improved:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    break
            else:
                no_improve_count = 0

        return best_res, best_L

    # --- Main algorithm ---
    # Step 1: Greedy forward construction
    residues = _greedy_construct(primes, max_L_check=5000)
    current_L = _score(residues, primes, 20000)

    # Step 2: Backward refinement
    residues, current_L = _backward_refine(residues, primes, 20000, deadline, num_passes=3)

    # Step 3: Local search
    residues, current_L = _local_search(residues, primes, 20000, deadline)

    # Step 4: Final verification with full range
    final_L = _score_full(residues, primes, 100000)

    # Ensure pure Python ints
    return [int(r) for r in residues]
