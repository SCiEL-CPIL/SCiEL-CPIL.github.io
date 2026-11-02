import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor bitmasks for all words
    masks = [0] * total_space
    for i in range(total_space):
        m = 1 << i
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                m |= (1 << (i + diff * powers[pos]))
        masks[i] = m

    full_mask = (1 << total_space) - 1

    best_size = total_space
    best_selected = list(range(total_space))

    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // powers[i]) % q
        return res

    def polish(selected, time_limit):
        """Remove redundant codewords iteratively."""
        selected = selected[:]
        if not selected:
            return selected

        # Build coverage counts
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        changed = True
        while changed and time.time() < time_limit:
            changed = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                removable = True
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        removable = False
                        break
                    mm ^= lowbit

                if removable:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    changed = True

        return selected

    def local_search(selected, time_limit):
        """Try to reduce the set by swapping out one word for another that covers
        more uncovered words while allowing removal of some covered-only-by-one words."""
        selected = selected[:]
        if not selected:
            return selected

        # Build coverage counts
        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

        # Find words that are uniquely covered by some codeword
        # Try: for each codeword, see if we can replace it with a different word
        # that covers all its uniquely-covered words plus more
        for _ in range(50):
            if time.time() > time_limit:
                break
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)

            for idx in indices:
                if idx >= len(selected):
                    continue
                w = selected[idx]
                m = masks[w]

                # Find words uniquely covered by w
                unique_words = []
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_words.append(pos)
                    mm ^= lowbit

                if not unique_words:
                    # Can just remove
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    improved = True
                    continue

                # Try to find a replacement word that covers all unique_words
                # and provides the most additional coverage
                best_gain = -1
                best_word = -1
                for w2 in range(total_space):
                    if w2 == w:
                        continue
                    # Check if w2 covers all unique_words
                    m2 = masks[w2]
                    covers_all = True
                    for uw in unique_words:
                        if not (m2 >> uw) & 1:
                            covers_all = False
                            break
                    if not covers_all:
                        continue

                    # Gain = number of words covered by w2 but not by w
                    gain = (m2 & ~m).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w2

                if best_word >= 0 and best_gain > 0:
                    # Remove w, add best_word
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit

                    mm = masks[best_word]
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] += 1
                        mm ^= lowbit

                    selected[idx] = best_word
                    improved = True

            if not improved:
                break

        return selected

    def run_greedy(randomized=False):
        covered = 0
        selected = []
        remaining = full_mask

        while remaining and time.time() < deadline - 5:
            best_gain = -1
            candidates = []

            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    candidates = [w]
                elif gain == best_gain and randomized:
                    candidates.append(w)

            if not candidates:
                break

            if randomized and len(candidates) > 1:
                best_word = random.choice(candidates)
            else:
                best_word = candidates[0]

            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        return selected

    # Run standard greedy
    if time.time() < deadline - 10:
        sol = run_greedy(randomized=False)
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            polished = polish(best_selected[:], deadline - 5)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Run randomized greedy restarts
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break

        sol = run_greedy(randomized=True)
        if len(sol) < best_size:
            best_size = len(sol)
            best_selected = sol[:]
            polished = polish(best_selected[:], deadline - 5)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Final polish and local search on the best solution
    if time.time() < deadline - 3:
        polished = polish(best_selected[:], deadline - 2)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    if time.time() < deadline - 1:
        ls = local_search(best_selected[:], deadline - 1)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
