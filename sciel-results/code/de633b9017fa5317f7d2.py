import numpy as np
import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 145.0  # safety margin before 150s kill

    base = 51975
    # All divisors > 1 of 51975 = 3^3 * 5^2 * 7 * 11
    divs = [d for d in range(2, base + 1) if base % d == 0 and d > 1]
    n_divs = len(divs)
    divs_set = set(divs)
    divs_arr = np.array(divs, dtype=np.int32)

    def time_left():
        return DEADLINE - time.time()

    def greedy_build(order):
        """Build a covering system greedily using a given order of moduli."""
        cover_count = np.zeros(base, dtype=np.int32)
        classes = []
        for m in order:
            zeros = np.where(cover_count == 0)[0]
            if len(zeros) == 0:
                classes.append([m, 0])
                continue
            A = np.bincount(zeros % m, minlength=m)
            best_r = int(np.argmax(A))
            classes.append([m, best_r])
            cover_count[best_r::m] += 1
        score = base - int(np.sum(cover_count > 0))
        return classes, score, cover_count

    def local_search(classes, cover_count, cur_score, max_sweeps=30):
        """Improve via single-residue moves with marginal gain calculation."""
        moduli = [c[0] for c in classes]
        residues = [c[1] for c in classes]
        n = len(moduli)

        for sweep in range(max_sweeps):
            if time_left() < 3:
                break
            improved = False
            zeros = np.where(cover_count == 0)[0]

            # Randomize order for diversity
            idx = list(range(n))
            random.shuffle(idx)

            for i in idx:
                if time_left() < 2:
                    break
                m = moduli[i]
                r = residues[i]

                # Count uncovered per residue class mod m
                A = np.bincount(zeros % m, minlength=m)

                # Positions covered only by this class (would become uncovered if we move)
                old_pos = np.arange(r, base, m)
                L_fixed = int(np.sum(cover_count[old_pos] == 1))

                # Best alternative residue
                A_temp = A.copy()
                A_temp[r] = -1
                best_r_prime = int(np.argmax(A_temp))
                best_gain = A_temp[best_r_prime] - L_fixed

                if best_gain > 0:
                    new_pos = np.arange(best_r_prime, base, m)
                    cover_count[old_pos] -= 1
                    cover_count[new_pos] += 1
                    residues[i] = best_r_prime
                    cur_score -= best_gain
                    improved = True
                    zeros = np.where(cover_count == 0)[0]

            if not improved:
                break

        return [[moduli[i], residues[i]] for i in range(n)], cur_score

    def two_opt_move(classes, cover_count, cur_score):
        """Try swapping two classes' residues simultaneously."""
        moduli = [c[0] for c in classes]
        residues = [c[1] for c in classes]
        n = len(moduli)
        improved = False

        if n < 2:
            return classes, cur_score

        # Try a limited number of random pairs
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(pairs)

        for i, j in pairs[:50]:  # limit to 50 random pairs
            if time_left() < 3:
                break
            m_i, r_i = moduli[i], residues[i]
            m_j, r_j = moduli[j], residues[j]

            zeros = np.where(cover_count == 0)[0]

            A_i = np.bincount(zeros % m_i, minlength=m_i)
            A_j = np.bincount(zeros % m_j, minlength=m_j)

            old_pos_i = np.arange(r_i, base, m_i)
            old_pos_j = np.arange(r_j, base, m_j)

            L_i = int(np.sum(cover_count[old_pos_i] == 1))
            L_j = int(np.sum(cover_count[old_pos_j] == 1))

            # Try all combinations of new residues (limited)
            A_i_temp = A_i.copy()
            A_i_temp[r_i] = -1
            A_j_temp = A_j.copy()
            A_j_temp[r_j] = -1

            best_r_i = int(np.argmax(A_i_temp))
            best_r_j = int(np.argmax(A_j_temp))

            gain = (A_i_temp[best_r_i] + A_j_temp[best_r_j]) - (L_i + L_j)

            if gain > 0:
                new_pos_i = np.arange(best_r_i, base, m_i)
                new_pos_j = np.arange(best_r_j, base, m_j)
                cover_count[old_pos_i] -= 1
                cover_count[new_pos_i] += 1
                cover_count[old_pos_j] -= 1
                cover_count[new_pos_j] += 1
                residues[i] = best_r_i
                residues[j] = best_r_j
                cur_score -= gain
                improved = True

        return [[moduli[k], residues[k]] for k in range(n)], cur_score

    best_score = base
    best_classes = None

    # Generate diverse orderings
    orderings = []

    # Sorted ascending
    orderings.append(sorted(divs))
    # Sorted descending
    orderings.append(sorted(divs, reverse=True))

    # By number of divisors (fewer divisors = more "prime-like")
    def divisor_count(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    orderings.append(sorted(divs, key=lambda x: divisor_count(x)))
    orderings.append(sorted(divs, key=lambda x: -divisor_count(x)))

    # By size (smaller moduli first cover more residues per class)
    orderings.append(sorted(divs, key=lambda x: x))
    orderings.append(sorted(divs, key=lambda x: -x))

    # Random orderings with different seeds
    for seed in range(20):
        rng = random.Random(seed * 7 + 3)
        order = divs.copy()
        rng.shuffle(order)
        orderings.append(order)

    # Heuristic: prioritize small moduli that are prime powers
    def is_prime_power(n):
        for p in [3, 5, 7, 11]:
            while n % p == 0:
                n //= p
            if n == 1:
                return True
        return False

    pp_divs = [d for d in divs if is_prime_power(d)]
    non_pp_divs = [d for d in divs if not is_prime_power(d)]
    orderings.append(sorted(pp_divs) + sorted(non_pp_divs, reverse=True))

    for oi, order in enumerate(orderings):
        if time_left() < 10:
            break

        classes, score, cover_count = greedy_build(order)

        if score < best_score:
            best_score = score
            best_classes = classes

        # Local search
        if time_left() > 10:
            classes, score = local_search(classes, cover_count, score, max_sweeps=25)
            if score < best_score:
                best_score = score
                best_classes = classes

            # Two-opt
            if time_left() > 5:
                classes, score = two_opt_move(classes, cover_count, score)
                if score < best_score:
                    best_score = score
                    best_classes = classes

                # Another round of local search
                if time_left() > 5:
                    cover_count = np.zeros(base, dtype=np.int32)
                    for m, r in classes:
                        cover_count[r::m] += 1
                    classes, score = local_search(classes, cover_count, score, max_sweeps=15)
                    if score < best_score:
                        best_score = score
                        best_classes = classes

    # Final polish: try more aggressive local search on the best solution
    if best_classes is not None and time_left() > 5:
        cover_count = np.zeros(base, dtype=np.int32)
        for m, r in best_classes:
            cover_count[r::m] += 1
        cur_score = base - int(np.sum(cover_count > 0))
        best_classes, cur_score = local_search(best_classes, cover_count, cur_score, max_sweeps=40)
        if cur_score < best_score:
            best_score = cur_score

    return best_classes if best_classes is not None else [[3, 0]]
