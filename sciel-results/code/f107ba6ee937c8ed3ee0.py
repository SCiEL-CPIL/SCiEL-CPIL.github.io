"""Combined approach: optimized greedy + fast local search for odd covering."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    base = 51975
    deadline = time.time() + 110

    # Get all odd divisors of 51975 greater than 1
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    n_div = len(divisors)

    def score_classes(classes):
        """Count uncovered residues."""
        covered = bytearray(base)
        for m, r in classes:
            covered[r::m] = b'\x01' * len(range(r, base, m))
        return covered.count(0)

    def greedy(order):
        """Greedy construction with given modulus order."""
        covered = bytearray(base)
        classes = []
        for m in order:
            best_r = 0
            best_count = -1
            for r in range(m):
                count = covered[r::m].count(0)
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = b'\x01' * len(range(best_r, base, m))
        return classes

    def build_coverage(classes):
        """Build coverage array from classes."""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            covered[r::m] = 1
        return covered

    def fast_local_search(classes, deadline):
        """
        Fast local search: for each modulus, try all residues.
        Uses incremental delta computation to avoid rebuilding full coverage.
        """
        best_classes = [list(c) for c in classes]
        covered = build_coverage(best_classes)
        best_score = int(base - np.sum(covered))

        improved = True
        passes = 0
        max_passes = 100
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            for i in range(n_div):
                m, r = best_classes[i]
                # Compute coverage without class i
                # covered_wo_i[x] = 1 if some other class covers x
                covered_wo_i = np.zeros(base, dtype=np.uint8)
                for j in range(n_div):
                    if j == i:
                        continue
                    mj, rj = best_classes[j]
                    covered_wo_i[rj::mj] = 1

                uncov_wo_i = int(base - np.sum(covered_wo_i))

                best_r = r
                best_uncov = uncov_wo_i
                for new_r in range(m):
                    newly_covered = int(np.sum(covered_wo_i[new_r::m] == 0))
                    uncov = uncov_wo_i - newly_covered
                    if uncov < best_uncov:
                        best_uncov = uncov
                        best_r = new_r

                if best_r != r:
                    best_classes[i][1] = best_r
                    # Update covered array
                    covered = build_coverage(best_classes)
                    best_score = best_uncov
                    improved = True
                    if best_score == 0:
                        return best_classes, best_score

        return best_classes, best_score

    def two_swap_search(classes, deadline):
        """Try swapping residues between pairs of moduli."""
        best_classes = [list(c) for c in classes]
        best_score = score_classes(best_classes)
        improved = True
        passes = 0
        max_passes = 20
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            for i in range(n_div):
                if time.time() > deadline:
                    break
                mi, ri = best_classes[i]
                for j in range(i + 1, n_div):
                    if time.time() > deadline:
                        break
                    mj, rj = best_classes[j]
                    # Try swapping residues
                    old_ri, old_rj = ri, rj
                    # Try a few candidate swaps
                    for new_ri in [rj % mi, (ri + 1) % mi, (ri + mj) % mi]:
                        for new_rj in [ri % mj, (rj + 1) % mj, (rj + mi) % mj]:
                            if new_ri == ri and new_rj == rj:
                                continue
                            best_classes[i][1] = new_ri
                            best_classes[j][1] = new_rj
                            new_score = score_classes(best_classes)
                            if new_score < best_score:
                                best_score = new_score
                                ri, rj = new_ri, new_rj
                                improved = True
                                if best_score == 0:
                                    return best_classes, best_score
                            else:
                                best_classes[i][1] = ri
                                best_classes[j][1] = rj

        return best_classes, best_score

    # Initial greedy with increasing order
    best_classes = greedy(divisors)
    best_score = score_classes(best_classes)

    # Try decreasing order
    classes_dec = greedy(divisors[::-1])
    score_dec = score_classes(classes_dec)
    if score_dec < best_score:
        best_score = score_dec
        best_classes = classes_dec

    # Try random orders
    random.seed(42)
    num_random_tries = 0
    while time.time() < deadline - 30 and num_random_tries < 300:
        order = divisors[:]
        random.shuffle(order)
        classes = greedy(order)
        score = score_classes(classes)
        if score < best_score:
            best_score = score
            best_classes = classes
            if best_score == 0:
                break
        num_random_tries += 1

    # Local search on the best found
    if time.time() < deadline - 5:
        best_classes, best_score = fast_local_search(best_classes, deadline - 5)

    # Two-swap search if time remains
    if time.time() < deadline - 2 and best_score > 0:
        best_classes, best_score = two_swap_search(best_classes, deadline - 2)

    # Final local search polish
    if time.time() < deadline and best_score > 0:
        best_classes, best_score = fast_local_search(best_classes, deadline)

    return best_classes
# EVOLVE-BLOCK-END
