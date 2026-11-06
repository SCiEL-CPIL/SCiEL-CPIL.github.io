"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # LCM of 1..22 for exact integer representation
    L = reduce(lcm, range(1, 23))

    # Precompute reciprocal as integer: L // i
    recip = [L // i for i in range(1, 23)]

    best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    def is_valid(selected):
        """Check if selected forms a dissociated set."""
        sums = {0}
        for idx in selected:
            r = recip[idx - 1]
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + r)
            if len(new_sums) != len(sums) * 2:
                return False
            sums = new_sums
        return True

    def backtrack(selected, remaining, sums, diffs):
        nonlocal best_result
        if time.time() > deadline:
            return
        if len(selected) > len(best_result):
            best_result = selected[:]

        if not remaining:
            return

        n = len(remaining)
        for i in range(n):
            idx = remaining[i]
            r = recip[idx - 1]
            if r in diffs or (-r) in diffs:
                continue
            
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + r)
            
            new_diffs = set(diffs)
            for d in diffs:
                new_diffs.add(d + r)
                new_diffs.add(d - r)
            new_diffs.add(r)
            new_diffs.add(-r)

            selected.append(idx)
            backtrack(selected, remaining[i+1:], new_sums, new_diffs)
            selected.pop()
            if len(best_result) >= 16:
                return

    # Phase 1: Backtracking search
    order = list(range(1, 23))
    backtrack([], order, {0}, {0})

    # Phase 2: Local search improvement
    if time.time() < deadline - 5:
        current = best_result[:]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try adding each unselected element
            for idx in range(1, 23):
                if idx not in current:
                    trial = current + [idx]
                    if is_valid(trial):
                        current = trial
                        improved = True
                        break
            # Try swapping: remove one, add another
            for i, idx_out in enumerate(current):
                for idx_in in range(1, 23):
                    if idx_in not in current:
                        trial = [x for x in current if x != idx_out] + [idx_in]
                        if is_valid(trial):
                            current = trial
                            improved = True
                            break
                if improved:
                    break
        if len(current) > len(best_result):
            best_result = current

    return best_result
# EVOLVE-BLOCK-END
