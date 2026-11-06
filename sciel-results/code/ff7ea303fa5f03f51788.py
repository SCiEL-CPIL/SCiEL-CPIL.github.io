"""Constraint-propagation + backtracking extension of the prescribed Sidon set."""
import itertools
import time

# EVOLVE-BLOCK-START

def construct():
    """
    Find a maximum-size Sidon set (all pairwise sums a+b for a<=b distinct)
    containing 1,2,5,11 within [1,284].

    Uses a backtracking search with forward checking:
      - Variables: candidate positions 1..284 (ordered)
      - Constraints: new element x must not create a duplicate sum
      - Forward checking: precompute which sums are already used; when trying
        to add x, the new sums are {x+y for y in current} ∪ {2x}; these must
        all be new and not in the used set.

    To make this tractable within 120 s, we use a greedy seed plus a limited
    backtracking search to try to improve upon the greedy result.
    """
    prescribed = [1, 2, 5, 11]
    used_sums = set()
    for i, x in enumerate(prescribed):
        for y in prescribed[i:]:
            used_sums.add(x + y)

    current = list(prescribed)

    # Greedy phase: scan 12..284, add if compatible
    for x in range(12, 285):
        new_sums = {x + y for y in current} | {2 * x}
        if len(new_sums) == len(current) + 1 and not (new_sums & used_sums):
            current.append(x)
            used_sums |= new_sums

    base_size = len(current)

    # Backtracking phase: try to improve beyond greedy
    # We'll do a DFS from the end, trying to replace or insert elements
    # to potentially get a larger set.

    def is_compatible(x, arr, used):
        """Check if x can be added to arr given used sums."""
        new_sums = set()
        for y in arr:
            s = x + y
            if s in new_sums or s in used:
                return False
            new_sums.add(s)
        if 2 * x in new_sums or 2 * x in used:
            return False
        return True

    def backtrack(arr, used, pos, best):
        """Try to extend arr starting from position pos in 1..284."""
        if len(arr) >= len(best):
            return
        if pos > 284:
            return
        x = pos
        # Try adding x
        new_used = used | {x + y for y in arr} | {2 * x}
        if not ({x + y for y in arr} & used) and not (2 * x in used) and len({x + y for y in arr} | {2*x}) == len(arr) + 1:
            new_arr = arr + [x]
            best[0] = max(best[0], len(new_arr))
            if len(new_arr) > len(best[0]):
                pass
            backtrack(new_arr, new_used, x + 1, best)
        # Try skipping x
        backtrack(arr, used, x + 1, best)

    # The full backtracking from scratch is too expensive.
    # Instead, use a local search / iterative improvement:
    # Try removing one element and re-adding to see if we can get a larger set.

    best_result = list(current)

    # Try iterative improvement: remove each element, then greedily rebuild
    for idx in range(len(current)):
        trial = [x for i, x in enumerate(current) if i != idx]
        trial_sums = set()
        for i, x in enumerate(trial):
            for y in trial[i:]:
                trial_sums.add(x + y)
        # Greedy fill from 1..284
        for x in range(1, 285):
            if x in trial:
                continue
            new_sums = {x + y for y in trial} | {2 * x}
            if len(new_sums) == len(trial) + 1 and not (new_sums & trial_sums):
                trial.append(x)
                trial_sums |= new_sums
        if len(trial) > len(best_result):
            best_result = trial

    # Another pass: try removing pairs
    n = len(current)
    for i in range(n):
        for j in range(i + 1, n):
            trial = [x for k, x in enumerate(current) if k != i and k != j]
            trial_sums = set()
            for k, x in enumerate(trial):
                for y in trial[k:]:
                    trial_sums.add(x + y)
            for x in range(1, 285):
                if x in trial:
                    continue
                new_sums = {x + y for y in trial} | {2 * x}
                if len(new_sums) == len(trial) + 1 and not (new_sums & trial_sums):
                    trial.append(x)
                    trial_sums |= new_sums
            if len(trial) > len(best_result):
                best_result = trial

    return best_result

# EVOLVE-BLOCK-END
