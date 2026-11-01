import time
import numpy as np
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = 145.0

    # Generate initial 81-word tetracode construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    num_cw = len(codewords)

    # Precompute ball indices for each codeword
    def compute_ball_indices(cw_row):
        indices = []
        base = 0
        for i in range(n):
            base = base * 3 + cw_row[i]
        indices.append(base)
        for pos in range(n):
            orig = cw_row[pos]
            prefix = 0
            for i in range(pos):
                prefix = prefix * 3 + cw_row[i]
            suffix_mult = 3 ** (n - 1 - pos)
            suffix_val = 0
            for i in range(pos + 1, n):
                suffix_val = suffix_val * 3 + cw_row[i]
            for v in range(3):
                if v == orig:
                    continue
                val = prefix * 3 ** (n - pos) + v * suffix_mult + suffix_val
                indices.append(val)
        return indices

    balls = [compute_ball_indices(codewords[i]) for i in range(num_cw)]

    # Coverage count
    coverage = np.zeros(729, dtype=np.int32)
    for ball in balls:
        for idx in ball:
            coverage[idx] += 1

    active = [True] * num_cw

    # Greedy removal with multiple random orderings
    rng = random.Random(12345)
    best_active = None
    best_count = num_cw

    for trial in range(200):
        if time.time() > DEADLINE - 5:
            break

        trial_active = active.copy()
        trial_coverage = coverage.copy()

        order = list(range(num_cw))
        rng.shuffle(order)

        for i in order:
            if not trial_active[i]:
                continue
            removable = True
            for idx in balls[i]:
                if trial_coverage[idx] <= 1:
                    removable = False
                    break
            if removable:
                trial_active[i] = False
                for idx in balls[i]:
                    trial_coverage[idx] -= 1

        cnt = sum(trial_active)
        if cnt < best_count:
            best_count = cnt
            best_active = trial_active
            active = trial_active
            coverage = trial_coverage

    # Final deterministic pass
    if time.time() < DEADLINE - 2:
        for _ in range(10):
            changed = True
            while changed and time.time() < DEADLINE - 2:
                changed = False
                for i in range(num_cw):
                    if not active[i]:
                        continue
                    removable = True
                    for idx in balls[i]:
                        if coverage[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        active[i] = False
                        for idx in balls[i]:
                            coverage[idx] -= 1
                        changed = True

    result = [codewords[i] for i in range(num_cw) if active[i]]

    # Verify coverage; fall back to full set if broken
    if time.time() < DEADLINE:
        final_cov = np.zeros(729, dtype=np.int32)
        for cw_row in result:
            for idx in compute_ball_indices(cw_row):
                final_cov[idx] += 1
        if np.any(final_cov == 0):
            result = codewords

    return result
