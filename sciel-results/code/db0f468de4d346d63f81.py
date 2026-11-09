import time
import random

def construct():
    N = 18
    MIN_DIST = 6
    TOTAL_BUDGET = 55.0  # Safety margin (evaluator kills at 90s)
    t_start = time.time()
    deadline = t_start + TOTAL_BUDGET

    # Step 1: Construct the extended quadratic-residue code [18,9,6] with 512 codewords
    g_val = 313

    def poly_mod(numerator, denominator):
        rem = numerator
        deg_den = denominator.bit_length() - 1
        while rem != 0 and rem.bit_length() - 1 >= deg_den:
            deg_rem = rem.bit_length() - 1
            shift = deg_rem - deg_den
            rem ^= (denominator << shift)
        return rem

    H_val = (1 << 17) - 1
    rem = poly_mod(H_val, g_val)

    if rem != 0:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                parity = word.bit_count() % 2
                word |= (parity << 16)
                words.append(word)
        return words

    rows = []
    for i in range(9):
        vec = g_val << i
        parity = vec.bit_count() % 2
        row = vec | (parity << 17)
        rows.append(row)

    base_words = []
    for m in range(512):
        word = 0
        for j in range(9):
            if (m >> j) & 1:
                word ^= rows[j]
        base_words.append(word)

    best_words = base_words[:]
    best_len = len(base_words)

    # Pre-compute bad masks
    bad_masks = [m for m in range(1 << N) if m.bit_count() < MIN_DIST]

    base_set = set(base_words)
    remaining_candidates = [c for c in range(1 << N) if c not in base_set]

    max_restarts = 2000

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Use bytearray for fast O(1) blocked membership
        blocked = bytearray(1 << N)

        # Initialize blocked from base words
        for w in base_words:
            for m in bad_masks:
                blocked[w ^ m] = 1

        words = base_words[:]

        # Alternate between random and structured orderings
        if restart % 3 == 0:
            random.shuffle(remaining_candidates)
        elif restart % 3 == 1:
            remaining_candidates.sort(key=lambda x: x.bit_count())
        else:
            remaining_candidates.sort(key=lambda x: -x.bit_count())

        # Greedy extension
        for c in remaining_candidates:
            if time.time() > deadline:
                break

            if blocked[c]:
                continue

            words.append(c)
            blocked[c] = 1
            for m in bad_masks:
                blocked[c ^ m] = 1

        if len(words) > best_len:
            best_words = words[:]
            best_len = len(words)

        if best_len > 1800:
            break

    return best_words
