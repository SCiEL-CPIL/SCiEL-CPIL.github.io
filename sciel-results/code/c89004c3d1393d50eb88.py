import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s; leave safety margin
    deadline = start_time + 142.0

    total_space = q ** n

    # Precompute positional powers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # Precompute neighbor lists (sparse representation)
    # Each word covers itself + n*(q-1) neighbors
    neighbors = []
    for i in range(total_space):
        nbrs = [i]
        for pos in range(n):
            digit_val = (i // powers[pos]) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                nbrs.append(i + diff * powers[pos])
        neighbors.append(nbrs)

    # Also build bitmasks for greedy (needed for fast gain computation)
    masks = [0] * total_space
    for i in range(total_space):
        m = 0
        for nb in neighbors[i]:
            m |= (1 << nb)
        masks[i] = m

    # Precompute coverers for fast local search
    # coverers[x] = list of words w such that x is in neighbors[w]
    coverers = [[] for _ in range(total_space)]
    for w in range(total_space):
        for nb in neighbors[w]:
            coverers[nb].append(w)

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
        """Try to reduce the set by swapping out one word for another."""
        selected = selected[:]
        if not selected:
            return selected

        counts = np.zeros(total_space, dtype=np.int32)
        for w in selected:
            mm = masks[w]
            while mm:
                lowbit = mm & -mm
                idx = lowbit.bit_length() - 1
                counts[idx] += 1
                mm ^= lowbit

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

                unique_words = []
                mm = m
                while mm:
                    lowbit = mm & -mm
                    pos = lowbit.bit_length() - 1
                    if counts[pos] == 1:
                        unique_words.append(pos)
                    mm ^= lowbit

                if not unique_words:
                    mm = m
                    while mm:
                        lowbit = mm & -mm
                        pos = lowbit.bit_length() - 1
                        counts[pos] -= 1
                        mm ^= lowbit
                    selected.pop(idx)
                    improved = True
                    continue

                # Fast lookup for candidates covering unique_words
                # Intersection of coverers[uw] for uw in unique_words
                candidates = None
                for uw in unique_words:
                    if candidates is None:
                        candidates = set(coverers[uw])
                    else:
                        candidates.intersection_update(coverers[uw])
                    
                    if not candidates:
                        break

                if not candidates:
                    continue

                # Exclude current word w
                candidates.discard(w)
                
                best_gain = -1
                best_word = -1
                for w2 in candidates:
                    m2 = masks[w2]
                    gain = (m2 & ~m).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_word = w2

                if best_word >= 0 and best_gain > 0:
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

    def run_greedy(randomized=False, alpha=1.0):
        """Greedy set cover with optional randomization and alpha weighting.
        alpha < 1.0 adds noise to encourage diversity."""
        covered = 0
        selected = []
        remaining = full_mask

        while remaining and time.time() < deadline - 5:
            best_gain = -1
            candidates = []

            # Use numpy for speed: compute gains for all words at once
            # But since we need bit_count on big ints, we do it in a loop
            for w in range(total_space):
                gain = (masks[w] & remaining).bit_count()
                if randomized:
                    # Add noise
                    gain = gain + random.gauss(0, 1.0) * alpha
                if gain > best_gain:
                    best_gain = gain
                    candidates = [w]
                elif gain == best_gain and not randomized:
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

    def run_randomized_greedy_with_bias():
        """Greedy that prefers words that are 'central' or have more overlap with existing coverage."""
        covered = 0
        selected = []
        remaining = full_mask

        while remaining and time.time() < deadline - 5:
            best_score = -1
            best_word = -1
            # Sample a subset to speed up when remaining is large
            if total_space > 500:
                candidates = random.sample(range(total_space), min(300, total_space))
            else:
                candidates = range(total_space)

            for w in candidates:
                gain = (masks[w] & remaining).bit_count()
                # Prefer words that are not too similar to already selected
                score = gain
                best_word = w
                if score > best_score:
                    best_score = score
                    best_word = w

            if best_word < 0:
                break

            selected.append(best_word)
            covered |= masks[best_word]
            remaining = full_mask & ~covered

        return selected

    def two_phase_greedy():
        """First pass: greedy. Second pass: try to merge pairs."""
        sol = run_greedy(randomized=False)
        if time.time() > deadline - 10:
            return sol

        # Try to remove any word if its coverage is fully covered by others
        sol = polish(sol[:], deadline - 10)
        return sol

    # Strategy 1: Standard greedy + polish
    if time.time() < deadline - 10:
        sol = run_greedy(randomized=False)
        polished = polish(sol[:], deadline - 10)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    # Strategy 2: Multiple randomized greedy restarts with polish
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break

        sol = run_greedy(randomized=True, alpha=random.uniform(0.1, 0.5))
        if len(sol) < best_size + 5:  # Only polish if close
            polished = polish(sol[:], deadline - 15)
            if len(polished) < best_size:
                best_size = len(polished)
                best_selected = polished[:]

    # Strategy 3: Local search on best solution
    if time.time() < deadline - 5:
        ls = local_search(best_selected[:], deadline - 5)
        if len(ls) < best_size:
            best_size = len(ls)
            best_selected = ls[:]

    # Strategy 4: Polish + local search + polish again
    if time.time() < deadline - 3:
        polished = polish(best_selected[:], deadline - 3)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]
        if time.time() < deadline - 2:
            ls2 = local_search(best_selected[:], deadline - 2)
            if len(ls2) < best_size:
                best_size = len(ls2)
                best_selected = ls2[:]

    # Final polish
    if time.time() < deadline - 1:
        polished = polish(best_selected[:], deadline - 1)
        if len(polished) < best_size:
            best_size = len(polished)
            best_selected = polished[:]

    best_words = [int_to_tuple(v) for v in best_selected]
    return best_words
