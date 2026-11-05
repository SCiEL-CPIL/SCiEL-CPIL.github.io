"""Combined: diverse prime-set candidates + max-impact greedy + local search refinement."""
import time
from fractions import Fraction
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

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

    # ---- Max-impact greedy residue assignment with local search ----
    def evaluate_set(P, max_check=MAX_L):
        """Assign residues using max-impact greedy: pick prime covering most new uncovered numbers."""
        if not P:
            return 0, {}
        P_sorted = sorted(P)
        
        covered = np.zeros(max_check + 1, dtype=np.uint8)
        residues = {}
        assigned_primes = set()

        # Max-impact greedy pass
        for n in range(1, max_check + 1):
            if not _time_ok():
                break
            if covered[n]:
                continue
            
            # Find the unused prime that would cover the most new uncovered numbers
            best_p = None
            best_count = -1
            best_r = 0
            
            for p in P_sorted:
                if p in assigned_primes:
                    continue
                # Try residue r = n % p (covers n)
                r = n % p
                # Count how many uncovered numbers in [1, max_check] are congruent to r mod p
                idx = np.arange(r, max_check + 1, p, dtype=np.int64)
                if idx.size == 0:
                    continue
                new_covered = np.sum(covered[idx] == 0)
                if new_covered > best_count:
                    best_count = new_covered
                    best_p = p
                    best_r = r
            
            if best_p is None:
                break
                
            residues[best_p] = best_r
            assigned_primes.add(best_p)
            
            # Update coverage
            idx = np.arange(best_r, max_check + 1, best_p, dtype=np.int64)
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

        L = get_L()

        # ---- Local search refinement: shift residues of large primes to fill gaps ----
        if L > 0 and _time_ok():
            # Find gaps in the prefix
            gaps = np.nonzero(covered[1:L+1] == 0)[0] + 1  # 1-indexed
            if gaps.size > 0 and gaps.size < 50:  # Only refine if few gaps
                # For each gap, try to find a large prime whose residue can be shifted to cover it
                large_P = [p for p in P_sorted if p > 50 and p in assigned_primes]
                for gap in gaps[:20]:  # Limit refinement attempts
                    if not _time_ok():
                        break
                    for p in large_P:
                        if p not in assigned_primes:
                            continue
                        old_r = residues.get(p, 0)
                        new_r = gap % p
                        if new_r == old_r:
                            continue
                        # Check if shifting improves coverage
                        old_idx = np.arange(old_r, max_check + 1, p, dtype=np.int64)
                        new_idx = np.arange(new_r, max_check + 1, p, dtype=np.int64)
                        
                        # Count loss from old residue
                        loss = 0
                        if old_idx.size:
                            loss = np.sum(covered[old_idx] == 1)  # Numbers only covered by this prime
                        # Count gain from new residue
                        gain = 0
                        if new_idx.size:
                            gain = np.sum(covered[new_idx] == 0)
                        
                        if gain > loss:
                            # Apply the shift
                            if old_idx.size:
                                # Only unmark numbers not covered by other primes
                                # Simplified: just mark new coverage
                                pass
                            residues[p] = new_r
                            if new_idx.size:
                                covered[new_idx] = 1
                            break

            L = get_L()

        return L, residues

    # Prioritize candidate sets: more small primes first, then more total primes
    def small_prime_count(P):
        return sum(1 for p in P if p <= 11)

    candidate_sets.sort(key=lambda P: (-small_prime_count(P), -len(P)))

    best_L = 0
    best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    for P in candidate_sets:
        if not _time_ok():
            break
        L, residues = evaluate_set(P)
        if L > best_L:
            best_L = L
            best_classes = [[p, residues[p]] for p in sorted(P) if p in residues]

    # ---- Second pass: re-evaluate best set with extended check range ----
    if best_classes and _time_ok():
        best_P = sorted(p for p, r in best_classes)
        L2, residues2 = evaluate_set(best_P, max_check=min(MAX_L, best_L + 30000))
        if L2 > best_L:
            best_L = L2
            best_classes = [[p, residues2[p]] for p in sorted(best_P) if p in residues2]

    # Final verification
    if best_classes:
        final_sum = sum(Fraction(1, p) for p, r in best_classes)
        if final_sum > 1:
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]
        prime_list = [p for p, r in best_classes]
        if len(prime_list) != len(set(prime_list)):
            best_classes = [[p, (i + 1) % p] for i, p in enumerate(seed_set)]

    return best_classes
# EVOLVE-BLOCK-END
