"""Improved program for the suite task; stronger search within the same family."""
# EVOLVE-BLOCK-START

import time
import random

def construct():
    """Find largest subset of {1..128} with no three distinct elements a,b,c
    such that a+b, a+c, b+c are all in the set."""
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    n = 128

    def set_to_mask(S):
        mask = 0
        for v in S:
            mask |= (1 << (v - 1))
        return mask

    def mask_to_set(mask):
        result = []
        for i in range(n):
            if (mask >> i) & 1:
                result.append(i + 1)
        return result

    def is_valid_mask(mask):
        """
        Check if the set represented by the bitmask violates the condition.
        Bit i represents integer i+1 (1..128).
        Condition: No three distinct members a, b, c in S such that a+b, a+c, b+c are also in S.
        """
        for i in range(n):
            if not (mask >> i) & 1:
                continue
            for j in range(i + 1, n):
                if not (mask >> j) & 1:
                    continue
                s_ab_idx = i + j + 1
                if s_ab_idx >= n:
                    continue
                if not (mask >> s_ab_idx) & 1:
                    continue
                candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
                if candidates & ~((1 << (j + 1)) - 1):
                    return False
        return True

    def try_add_mask(mask, x):
        """Try adding x (1..128) to set represented by mask. Returns new mask if valid, else None."""
        bit = 1 << (x - 1)
        if mask & bit:
            return mask
        new_mask = mask | bit
        if is_valid_mask(new_mask):
            return new_mask
        return None

    def find_removals_mask(mask, x, max_removals=4):
        """Find minimal removals to allow adding x. Returns set of removals or None."""
        bit = 1 << (x - 1)
        if mask & bit:
            return set()
        new_mask = mask | bit

        violations = []
        for i in range(n):
            if not (new_mask >> i) & 1:
                continue
            for j in range(i + 1, n):
                if not (new_mask >> j) & 1:
                    continue
                s_ab_idx = i + j + 1
                if s_ab_idx >= n:
                    continue
                if not (new_mask >> s_ab_idx) & 1:
                    continue
                candidates = new_mask & (new_mask >> (i + 1)) & (new_mask >> (j + 1))
                candidates &= ~((1 << (j + 1)) - 1)
                while candidates:
                    lsb = candidates & -candidates
                    k = lsb.bit_length() - 1
                    candidates ^= lsb
                    violations.append((i + 1, j + 1, k + 1))

        if not violations:
            return set()

        from collections import Counter
        elem_count = Counter()
        for v in violations:
            for e in v:
                elem_count[e] += 1

        removals = set()
        remaining = list(violations)

        while remaining and len(removals) < max_removals:
            best_elem = None
            best_cnt = 0
            for e in elem_count:
                if e in removals:
                    continue
                cnt = sum(1 for v in remaining if e in v)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_elem = e
            if best_elem is None or best_cnt == 0:
                break
            removals.add(best_elem)
            remaining = [v for v in remaining if best_elem not in v]

        if not remaining:
            test_mask = new_mask
            for r in removals:
                test_mask &= ~(1 << (r - 1))
            if is_valid_mask(test_mask):
                return removals
        return None

    # Phase 1: Structured initialization
    best_mask = 0
    best_size = 0

    # Known good baseline: {16..31} | {64..128}
    baseline = set(range(16, 32)) | set(range(64, 129))
    baseline_mask = set_to_mask(baseline)
    if is_valid_mask(baseline_mask):
        best_mask = baseline_mask
        best_size = baseline_mask.bit_count()

    # Try different K values for H = {K..128}
    for K in range(50, 80):
        if time.time() > deadline - 10:
            break
        H = set(range(K, 129))
        max_L = (K - 1) // 2
        if max_L < 1:
            continue
        start_L = max_L // 2 + 1
        L = set(range(start_L, max_L + 1))
        S = L | H
        mask = set_to_mask(S)
        if is_valid_mask(mask):
            if mask.bit_count() > best_size:
                best_size = mask.bit_count()
                best_mask = mask

    # Simple interval {63..128}
    interval_mask = set_to_mask(set(range(63, 129)))
    if is_valid_mask(interval_mask):
        if interval_mask.bit_count() > best_size:
            best_size = interval_mask.bit_count()
            best_mask = interval_mask

    # Phase 2: Greedy add - multiple passes
    for _ in range(8):
        if time.time() > deadline - 15:
            break
        improved = False
        for v in range(1, 129):
            if time.time() > deadline - 15:
                break
            bit = 1 << (v - 1)
            if best_mask & bit:
                continue
            trial_mask = best_mask | bit
            if is_valid_mask(trial_mask):
                best_mask = trial_mask
                best_size = best_mask.bit_count()
                improved = True
        if not improved:
            break

    # Phase 3: Local search with single swaps
    random.seed(42)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for rem in range(1, 129):
            if time.time() > deadline - 10:
                break
            rem_bit = 1 << (rem - 1)
            if not (best_mask & rem_bit):
                continue
            trial_base = best_mask & ~rem_bit
            for add in range(1, 129):
                if add == rem:
                    continue
                add_bit = 1 << (add - 1)
                if trial_base & add_bit:
                    continue
                trial = trial_base | add_bit
                if is_valid_mask(trial):
                    if trial.bit_count() > best_size:
                        best_mask = trial
                        best_size = trial.bit_count()
                        improved = True

    # Phase 4: Aggressive local search with random swaps and additions
    if time.time() < deadline - 5:
        current_mask = best_mask
        current_size = best_size

        for iteration in range(30000):
            if time.time() > deadline - 5:
                break

            action = random.random()

            if action < 0.3:
                candidates = [x for x in range(1, 129) if not (current_mask >> (x - 1)) & 1]
                if not candidates:
                    continue
                to_add = random.choice(candidates)
                new_mask = try_add_mask(current_mask, to_add)
                if new_mask is not None:
                    current_mask = new_mask
                    current_size = current_mask.bit_count()
                    if current_size > best_size:
                        best_mask = current_mask
                        best_size = current_size

            elif action < 0.7:
                in_set = [x for x in range(1, 129) if (current_mask >> (x - 1)) & 1]
                not_in_set = [x for x in range(1, 129) if not (current_mask >> (x - 1)) & 1]
                if not in_set or not not_in_set:
                    continue
                to_remove = random.choice(in_set)
                to_add = random.choice(not_in_set)
                rem_bit = 1 << (to_remove - 1)
                add_bit = 1 << (to_add - 1)
                new_mask = (current_mask & ~rem_bit) | add_bit
                if is_valid_mask(new_mask):
                    current_mask = new_mask
                    current_size = current_mask.bit_count()
                    if current_size > best_size:
                        best_mask = current_mask
                        best_size = current_size

            else:
                not_in_set = [x for x in range(1, 129) if not (current_mask >> (x - 1)) & 1]
                if not not_in_set:
                    continue
                to_add = random.choice(not_in_set)
                removals = find_removals_mask(current_mask, to_add, max_removals=3)
                if removals is not None:
                    new_mask = current_mask | (1 << (to_add - 1))
                    for r in removals:
                        new_mask &= ~(1 << (r - 1))
                    if is_valid_mask(new_mask):
                        new_size = new_mask.bit_count()
                        if new_size > best_size:
                            best_mask = new_mask
                            best_size = new_size
                            current_mask = new_mask
                            current_size = new_size

    # Phase 5: Final polish - try all single additions one more time
    if time.time() < deadline - 2:
        for v in range(1, 129):
            if time.time() > deadline - 2:
                break
            bit = 1 << (v - 1)
            if best_mask & bit:
                continue
            trial_mask = best_mask | bit
            if is_valid_mask(trial_mask):
                best_mask = trial_mask
                best_size = best_mask.bit_count()

    # Final verification
    if not is_valid_mask(best_mask):
        best_mask = set_to_mask(set(range(63, 129)))
        best_size = 66

    result = mask_to_set(best_mask)
    result = [int(x) for x in result]
    return result

# EVOLVE-BLOCK-END
