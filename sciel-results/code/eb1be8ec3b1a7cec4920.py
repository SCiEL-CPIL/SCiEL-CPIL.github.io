"""Combined: Diverse prime sets + Max-Impact Greedy Residue Assignment + Multi-Heuristic Evaluation."""
import time
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def _time_ok():
        return time.time() < deadline

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]

    primes = sieve(1000)
    small_primes = [2, 3, 5, 7, 11]
    large_primes = [p for p in primes if p > 11]

    MAX_L = 100000

    # ---- Build candidate prime sets ----
    seen_sets = set()
    candidate_sets = []

    def add_set(P):
        P = sorted(P)
        if not P:
            return
        key = tuple(P)
        if key in seen_sets:
            return
        # budget check
        s = sum(Fraction(1, p) for p in P)
        if s > 1:
            return
        seen_sets.add(key)
        candidate_sets.append(P)

    # Seed: primes >= 13
    seed_set = [p for p in primes if p >= 13]
    add_set(seed_set)

    # Subsets of small primes, fill rest with smallest large primes under budget
    import itertools
    for r in range(len(small_primes) + 1):
        if not _time_ok():
            break
        for subset in itertools.combinations(small_primes, r):
            current_sum = sum(Fraction(1, p) for p in subset)
            if current_sum > 1:
                continue
            current_set = list(subset)
            for p in large_primes:
                if current_sum + Fraction(1, p) <= 1:
                    current_set.append(p)
                    current_sum += Fraction(1, p)
                else:
                    break
            add_set(current_set)

    # Targeted swaps: remove some large primes, add a small prime
    for small in [11, 7, 5, 3, 2]:
        if not _time_ok():
            break
        for remove_list in [[], [13], [17], [19], [13, 17], [13, 19], [17, 19],
                            [13, 17, 19], [13, 17, 19, 23], [23, 29], [13, 17, 23]]:
            s = [p for p in seed_set if p not in remove_list]
            s_sum = sum(Fraction(1, p) for p in s)
            if s_sum + Fraction(1, small) <= 1:
                s.append(small)
                add_set(s)

    # ---- Precompute index arrays for each prime ----
    # For prime p, residue r: indices of numbers in 1..MAX_L congruent to r mod p
    # r=0: p, 2p, 3p, ...
    # r>0: r, r+p, r+2p, ...
    prime_indices = {}
    for p in primes:
        arrays = []
        for r in range(p):
            if r == 0:
                idx = np.arange(p, MAX_L + 1, p, dtype=np.int64)
            else:
                idx = np.arange(r, MAX_L + 1, p, dtype=np.int64)
            arrays.append(idx)
        prime_indices[p] = arrays

    # ---- Evaluation with Max-Impact Greedy ----
    def evaluate_set(P, max_check=MAX_L, heuristic="impact"):
        """
        Assign residues to cover the longest prefix 1..L.
        Heuristic 'impact': For each uncovered n, pick p that covers most currently uncovered numbers.
        Heuristic 'small': Pick smallest unused p.
        Heuristic 'large': Pick largest unused p.
        """
        if not P:
            return 0, {}
        P_sorted = sorted(P)
        
        covered = np.zeros(max_check + 1, dtype=np.uint8)
        residues = {}
        unused_primes = set(P)

        for n in range(1, max_check + 1):
            if not _time_ok():
                break
            if covered[n]:
                continue
            
            if not unused_primes:
                break

            best_p = None
            best_score = -1
            best_r = 0

            # Determine candidate primes
            if heuristic == "impact":
                candidates = list(unused_primes)
            elif heuristic == "small":
                candidates = [p for p in P_sorted if p in unused_primes][:1]
            elif heuristic == "large":
                candidates = [p for p in reversed(P_sorted) if p in unused_primes][:1]
            else:
                candidates = list(unused_primes)

            for p in candidates:
                r = n % p
                idx = prime_indices[p][r]
                if idx.size == 0:
                    continue
                score = np.sum(covered[idx] == 0)
                
                if score > best_score:
                    best_score = score
                    best_p = p
                    best_r = r

            if best_p is None:
                break

            residues[best_p] = best_r
            unused_primes.discard(best_p)

            # Update coverage
            idx = prime_indices[best_p][best_r]
            if idx.size:
                covered[idx] = 1

        # Calculate L
        def get_L():
            if max_check < 1:
                return 0
            seg = covered[1:max_check + 1]
            z = np.nonzero(seg == 0)[0]
            if z.size:
                return int(z[0])
            return max_check

        return get_L(), residues

    # Prioritize candidate sets
    def small_prime_count(P):
        return sum(1 for p in P if p <= 11)

    candidate_sets.sort(key=lambda P: (-small_prime_count(P), -len(P)))

    best_L = 0
    best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    # Evaluate each set with multiple heuristics
    for P in candidate_sets:
        if not _time_ok():
            break
            
        # Try 'impact' heuristic first
        L, residues = evaluate_set(P, max_check=MAX_L, heuristic="impact")
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in sorted(P) if p in residues]
        
        if not _time_ok():
            break

        # Try 'small' heuristic if time permits
        L_small, residues_small = evaluate_set(P, max_check=MAX_L, heuristic="small")
        if L_small > best_L:
            best_L = L_small
            best_classes = [[p, residues_small[p]] for p in sorted(P) if p in residues_small]
            
        if not _time_ok():
            break

        # Try 'large' heuristic if time permits
        L_large, residues_large = evaluate_set(P, max_check=MAX_L, heuristic="large")
        if L_large > best_L:
            best_L = L_large
            best_classes = [[p, residues_large[p]] for p in sorted(P) if p in residues_large]

    # ---- Local Refinement / Polish ----
    # Try to extend the best solution by fixing the first gap via residue reassignment.
    if best_classes and _time_ok():
        best_P = sorted(p for p, r in best_classes)
        
        def compute_L_from_residues(residues_dict):
            """Compute coverage and L from a residue assignment."""
            cov = np.zeros(MAX_L + 1, dtype=np.uint8)
            for p, r in residues_dict.items():
                idx = prime_indices[p][r]
                if idx.size:
                    cov[idx] = 1
            seg = cov[1:MAX_L + 1]
            z = np.nonzero(seg == 0)[0]
            if z.size:
                return int(z[0])
            return MAX_L

        # Try to fix the first gap
        for polish_iter in range(3):
            if not _time_ok():
                break
            
            current_residues = {p: r for p, r in best_classes}
            current_L = compute_L_from_residues(current_residues)
            
            if current_L >= MAX_L:
                break
            
            gap = current_L  # first uncovered number is current_L (0-indexed in seg, so actual number is current_L)
            # Wait: if z[0] = k, then the first uncovered number in 1..MAX_L is k+1 (1-indexed)
            # Actually seg = cov[1:MAX_L+1], so seg[i] = cov[i+1]
            # If z[0] = k, then cov[k+1] = 0, so the gap is at number k+1
            gap = current_L  # This is the 0-based index in seg, so actual number is gap+1
            # Let me re-check: get_L returns z[0] which is the index in seg where first 0 appears
            # seg[0] = cov[1], seg[1] = cov[2], etc.
            # If z[0] = k, then cov[k+1] = 0, so first uncovered number is k+1
            # But get_L returns k, and the score is "largest L such that every integer 1..L lies in some class"
            # So if first uncovered is at number k+1, then L = k. That matches.
            
            gap_number = current_L + 1  # actual integer that's uncovered
            
            # Try to find a prime p and new residue r' such that:
            # 1. r' = gap_number % p (so gap_number is covered)
            # 2. Changing p's residue to r' doesn't uncover any number in 1..current_L
            # 3. This might also cover some numbers beyond current_L
            
            improved = False
            for p in best_P:
                if not _time_ok():
                    break
                
                old_r = current_residues.get(p, None)
                if old_r is None:
                    continue
                
                new_r = gap_number % p
                
                if new_r == old_r:
                    continue  # Already covers gap_number? Shouldn't happen if gap is uncovered
                
                # Check: does changing p from old_r to new_r uncover any number in 1..current_L?
                # Numbers covered by old assignment: prime_indices[p][old_r]
                # Numbers covered by new assignment: prime_indices[p][new_r]
                # We need: all numbers in 1..current_L that were ONLY covered by p under old_r 
                #          must still be covered by some other prime under the new assignment.
                
                old_idx = prime_indices[p][old_r]
                new_idx = prime_indices[p][new_r]
                
                # Find numbers in 1..current_L that are in old_idx but not in new_idx
                # and check if they're covered by other primes.
                
                # Efficient approach: check if any number in 1..gap_number-1 (i.e., 1..current_L) 
                # is in old_idx but not in new_idx and not covered by other primes.
                
                # Let's build a set of numbers covered by other primes in 1..current_L
                # This is expensive. Alternative: just check a few critical numbers.
                
                # Simpler heuristic: check if the number of newly uncovered numbers in 1..current_L
                # is zero. We can compute this by checking the symmetric difference.
                
                # Numbers in old_idx ∩ [1, current_L] but not in new_idx ∩ [1, current_L]
                # For each such number, check if it's covered by another prime.
                
                # To keep it fast, let's just try it and verify with a full recomputation if it looks promising.
                
                # Quick check: if old_r and new_r are "close" in some sense, less disruption.
                # Actually, let's just try the swap and verify.
                
                trial_residues = dict(current_residues)
                trial_residues[p] = new_r
                
                trial_L = compute_L_from_residues(trial_residues)
                
                if trial_L > current_L:
                    best_classes = [[p, r] for p, r in sorted(trial_residues.items())]
                    current_residues = trial_residues
                    current_L = trial_L
                    improved = True
                    break  # Found an improvement, restart polish loop
            
            if not improved:
                break  # No improvement possible, stop

    # Final verification
    if best_classes:
        final_sum = sum(Fraction(1, p) for p, r in best_classes)
        if final_sum > 1:
            # Fallback to seed
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]
        prime_list = [p for p, r in best_classes]
        if len(prime_list) != len(set(prime_list)):
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    # Safety: ensure we return within time
    if time.time() > deadline - 5:
        # Emergency: return whatever we have
        if best_classes:
            final_sum = sum(Fraction(1, p) for p, r in best_classes)
            if final_sum > 1:
                best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]
    
    return best_classes
# EVOLVE-BLOCK-END
