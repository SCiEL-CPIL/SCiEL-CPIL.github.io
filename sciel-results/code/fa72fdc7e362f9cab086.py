"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Fast sampled greedy + local search (redundancy removal, best-swap, ILS perturb/accept).
    """
    if n == 0:
        return [[]]

    total_words = q ** n

    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word

    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors as sets. Fast path for radius=1.
    neighbors = [None] * total_words
    if radius == 1:
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            for pos in range(n):
                cur = word[pos]
                for val in range(q):
                    if val == cur:
                        continue
                    word[pos] = val
                    neigh.append(word_to_int(word, q))
                word[pos] = cur
            neighbors[w] = set(neigh)
    else:
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            if radius >= 1:
                for pos in range(n):
                    cur = word[pos]
                    for val in range(q):
                        if val == cur:
                            continue
                        word[pos] = val
                        neigh.append(word_to_int(word, q))
                    word[pos] = cur
            if radius >= 2:
                word = int_to_word(w, n, q)
                for pos1 in range(n):
                    c1 = word[pos1]
                    for val1 in range(q):
                        if val1 == c1:
                            continue
                        word[pos1] = val1
                        for pos2 in range(pos1 + 1, n):
                            c2 = word[pos2]
                            for val2 in range(q):
                                if val2 == c2:
                                    continue
                                word[pos2] = val2
                                neigh.append(word_to_int(word, q))
                            word[pos2] = c2
                    word[pos1] = c1
            neighbors[w] = set(neigh)

    # Precompute neighbor lists for fast iteration (list form)
    neigh_lists = [list(s) for s in neighbors]

    def greedy_init():
        """Sampled greedy: pick candidate covering most uncovered words."""
        covered = [False] * total_words
        codewords = []
        uncovered = set(range(total_words))

        while uncovered:
            if len(uncovered) > 400:
                sample_uncovered = random.sample(list(uncovered), min(80, len(uncovered)))
            else:
                sample_uncovered = list(uncovered)

            candidates = set()
            for u in sample_uncovered:
                candidates.update(neighbors[u])
            if not candidates:
                break

            best_word = -1
            best_new_cov = -1
            for w in candidates:
                new_cov = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        new_cov += 1
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_word = w
                    if new_cov == len(uncovered):
                        break

            if best_word == -1:
                u = next(iter(uncovered))
                best_word = next(iter(neighbors[u]))

            codewords.append(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = True
                    uncovered.discard(u)

        return codewords

    def build_coverage(codewords):
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neigh_lists[cw]:
                coverage_count[u] += 1
        return coverage_count

    def remove_redundant(codewords, coverage_count, cw_set):
        removed_any = False
        changed = True
        while changed:
            changed = False
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            for i in indices:
                cw = codewords[i]
                can_remove = True
                for u in neigh_lists[cw]:
                    if coverage_count[u] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for u in neigh_lists[cw]:
                        coverage_count[u] -= 1
                    codewords.pop(i)
                    cw_set.discard(cw)
                    changed = True
                    removed_any = True
                    break
        return removed_any

    def try_best_swap(codewords, coverage_count, cw_set):
        """Try to swap one codeword for a better one (maximizes redundant coverage)."""
        if not codewords:
            return False
        idx = random.randrange(len(codewords))
        old_cw = codewords[idx]
        would_uncover = [u for u in neigh_lists[old_cw] if coverage_count[u] == 1]

        if not would_uncover:
            for u in neigh_lists[old_cw]:
                coverage_count[u] -= 1
            codewords.pop(idx)
            cw_set.discard(old_cw)
            return True

        # Candidate pool: intersection of neighbors of all would_uncover words
        candidates = None
        for u in would_uncover:
            if candidates is None:
                candidates = set(neighbors[u])
            else:
                candidates.intersection_update(neighbors[u])
                if not candidates:
                    break
        if not candidates:
            return False
        candidates.discard(old_cw)
        if not candidates:
            return False

        # Pick candidate maximizing redundant coverage (words with coverage>=2 it will cover)
        best_cw = None
        best_redundancy = -1
        cand_list = list(candidates)
        # If too many, sample
        if len(cand_list) > 40:
            cand_list = random.sample(cand_list, 40)
        for new_cw in cand_list:
            redundancy = 0
            for u in neigh_lists[new_cw]:
                if coverage_count[u] >= 2:
                    redundancy += 1
            if redundancy > best_redundancy:
                best_redundancy = redundancy
                best_cw = new_cw

        if best_cw is None:
            return False

        for u in neigh_lists[old_cw]:
            coverage_count[u] -= 1
        for u in neigh_lists[best_cw]:
            coverage_count[u] += 1
        codewords[idx] = best_cw
        cw_set.discard(old_cw)
        cw_set.add(best_cw)
        return True

    def optimize(codewords, time_limit):
        coverage_count = build_coverage(codewords)
        cw_set = set(codewords)
        remove_redundant(codewords, coverage_count, cw_set)

        start_time = time.time()
        no_improve = 0

        while time.time() - start_time < time_limit:
            # Remove redundant
            if remove_redundant(codewords, coverage_count, cw_set):
                no_improve = 0
                continue

            # Try a best-swap
            if try_best_swap(codewords, coverage_count, cw_set):
                no_improve = 0
                # After a swap, try to remove redundant again
                remove_redundant(codewords, coverage_count, cw_set)
                continue

            # No simple improvement: perturb via multi-swap (ILS) to escape local minima
            no_improve += 1
            if no_improve >= 3 and codewords:
                # Perturb: do k random valid swaps
                k = random.randint(2, 4)
                for _ in range(k):
                    if not codewords:
                        break
                    idx = random.randrange(len(codewords))
                    old_cw = codewords[idx]
                    would_uncover = [u for u in neigh_lists[old_cw] if coverage_count[u] == 1]
                    if not would_uncover:
                        for u in neigh_lists[old_cw]:
                            coverage_count[u] -= 1
                        codewords.pop(idx)
                        cw_set.discard(old_cw)
                        continue
                    candidates = None
                    for u in would_uncover:
                        if candidates is None:
                            candidates = set(neighbors[u])
                        else:
                            candidates.intersection_update(neighbors[u])
                            if not candidates:
                                break
                    if not candidates:
                        continue
                    candidates.discard(old_cw)
                    if not candidates:
                        continue
                    new_cw = random.choice(list(candidates))
                    for u in neigh_lists[old_cw]:
                        coverage_count[u] -= 1
                    for u in neigh_lists[new_cw]:
                        coverage_count[u] += 1
                    codewords[idx] = new_cw
                    cw_set.discard(old_cw)
                    cw_set.add(new_cw)
                # After perturbation, run a short local search to recover
                for _ in range(50):
                    if time.time() - start_time > time_limit:
                        break
                    if remove_redundant(codewords, coverage_count, cw_set):
                        continue
                    if try_best_swap(codewords, coverage_count, cw_set):
                        remove_redundant(codewords, coverage_count, cw_set)
                        continue
                    break
                no_improve = 0

            if time.time() - start_time > time_limit - 1:
                break

        remove_redundant(codewords, coverage_count, cw_set)
        return codewords

    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []

    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        if remaining < 3:
            break
        # Allocate: greedy is fast; give rest to optimize
        opt_time = max(1.0, remaining - 2)
        cw = greedy_init()
        cw = optimize(cw, opt_time)

        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = cw

        if time.time() - start_time > time_limit - 5:
            break

    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
