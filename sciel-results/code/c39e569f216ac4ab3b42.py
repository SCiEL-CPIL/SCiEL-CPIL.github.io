"""Greedy set-cover + randomized local improvement for the ternary covering code K_3(7,1)."""
import itertools
import random
import time


def construct(q, n, radius):
    start = time.time()
    deadline = start + 130.0  # safety margin before 150s kill

    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n

    # Index each word
    word_index = {w: i for i, w in enumerate(all_words)}
    # Precompute neighborhood (frozenset of word indices) for every word
    neighborhoods = []
    for w in all_words:
        nb = [w]
        for i in range(n):
            old = w[i]
            for v in range(q):
                if v == old:
                    continue
                lst = list(w)
                lst[i] = v
                nb.append(tuple(lst))
        neighborhoods.append(frozenset(word_index[x] for x in nb))

    def greedy_cover(seed=None, time_limit=None):
        if seed is not None:
            random.seed(seed)
        uncovered = set(range(total))
        chosen_set = set()
        chosen = []
        while uncovered:
            best_idx = None
            best_count = -1
            for idx in range(total):
                if idx in chosen_set:
                    continue
                nb = neighborhoods[idx]
                cnt = 0
                for x in nb:
                    if x in uncovered:
                        cnt += 1
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
            if best_idx is None:
                break
            chosen.append(best_idx)
            chosen_set.add(best_idx)
            uncovered -= neighborhoods[best_idx]
            if time_limit is not None and time.time() > time_limit:
                break
        return chosen

    def local_improve(code, time_limit):
        """Remove redundant codewords and try 2-for-1 swaps to reduce size."""
        code = list(code)
        changed = True
        while changed and (time_limit is None or time.time() < time_limit):
            changed = False
            # Redundancy removal
            for idx in list(code):
                remaining = [c for c in code if c != idx]
                covered = set()
                for c in remaining:
                    covered |= neighborhoods[c]
                if len(covered) == total:
                    code.remove(idx)
                    changed = True
                    break
            if changed:
                continue
            # Try 2-for-1 swap: remove two codewords, add one new codeword
            for i in range(len(code)):
                for j in range(i + 1, len(code)):
                    remaining = [c for c in code if c != code[i] and c != code[j]]
                    covered = set()
                    for c in remaining:
                        covered |= neighborhoods[c]
                    # Need a new codeword that covers all uncovered
                    uncovered = set(range(total)) - covered
                    if not uncovered:
                        code.pop(j)
                        code.pop(i)
                        changed = True
                        break
                    # Find a single codeword that covers all uncovered
                    for new_idx in range(total):
                        if new_idx in code:
                            continue
                        if uncovered <= neighborhoods[new_idx]:
                            code.pop(j)
                            code.pop(i)
                            code.append(new_idx)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        return code

    best = None
    best_len = None

    # First greedy run
    code = greedy_cover(seed=12345, time_limit=deadline - 10)
    code = local_improve(code, deadline - 5)
    best = code
    best_len = len(code)

    # Randomized restarts
    attempt = 0
    while time.time() < deadline - 5:
        attempt += 1
        seed = 100000 + attempt
        code = greedy_cover(seed=seed, time_limit=deadline - 5)
        code = local_improve(code, deadline - 5)
        if len(code) < best_len:
            best = code
            best_len = len(code)
        if best_len <= 186:
            break

    if best is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    return [list(all_words[i]) for i in best]
