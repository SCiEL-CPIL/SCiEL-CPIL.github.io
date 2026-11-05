"""Rewritten program for dissociated reciprocal subset sums."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    deadline = time.time() + 55  # Safety margin before 60s limit

    # LCM of 1..22
    D = 232792560
    vals = {x: D // x for x in range(1, 23)}

    # Start with a known good set of size 11
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19, 21, 22]
    best_size = len(best_set)

    all_elements = list(range(1, 23))

    def is_valid_addition(current_sums, v):
        """Check if adding value v to current_sums creates a collision."""
        for s in current_sums:
            if s + v in current_sums:
                return False
        return True

    def try_backtrack(perm, target_size, start_idx=0, current=None, current_sums=None):
        """
        Backtracking search to find a valid set of target_size from perm.
        Returns (found, current_list) or (False, None).
        """
        if current is None:
            current = []
        if current_sums is None:
            current_sums = {0}

        if time.time() > deadline:
            return False, None

        if len(current) == target_size:
            return True, list(current)

        if start_idx >= len(perm):
            return False, None

        # Try each element from start_idx onwards
        for i in range(start_idx, len(perm)):
            if time.time() > deadline:
                return False, None

            x = perm[i]
            v = vals[x]

            if not is_valid_addition(current_sums, v):
                continue

            # Add x
            current.append(x)
            old_sums = current_sums
            new_sums = set(old_sums)
            for s in old_sums:
                new_sums.add(s + v)
            current_sums = new_sums

            found, result = try_backtrack(perm, target_size, i + 1, current, current_sums)
            if found:
                return True, result

            # Undo
            current.pop()
            current_sums = old_sums

        return False, None

    def random_search(target_size, num_attempts=50):
        """
        Try random permutations and backtracking to find a valid set of target_size.
        """
        for attempt in range(num_attempts):
            if time.time() > deadline:
                return None

            perm = all_elements[:]
            random.shuffle(perm)

            found, result = try_backtrack(perm, target_size)
            if found:
                return result

        return None

    def greedy_search(target_size):
        """
        Greedy approach: add elements in order of increasing value (decreasing denominator).
        """
        # Sort by value ascending (denominator descending) - smaller values are "safer"
        sorted_elems = sorted(all_elements, key=lambda x: vals[x])

        current = []
        current_sums = {0}

        for x in sorted_elems:
            if len(current) >= target_size:
                break

            v = vals[x]

            if is_valid_addition(current_sums, v):
                current.append(x)
                old_sums = current_sums
                new_sums = set(old_sums)
                for s in old_sums:
                    new_sums.add(s + v)
                current_sums = new_sums

        if len(current) >= target_size:
            return current[:target_size]

        return None

    # Try to improve from current best_size
    for target_size in range(17, best_size, -1):
        if time.time() > deadline:
            break

        # First try greedy
        result = greedy_search(target_size)
        if result is not None:
            best_set = result
            best_size = target_size
            break

        # Then try random backtracking
        result = random_search(target_size, num_attempts=30)
        if result is not None:
            best_set = result
            best_size = target_size
            break

    # If we haven't improved much, try more aggressive search for best_size + 1
    if best_size < 17 and time.time() < deadline - 10:
        target = best_size + 1
        result = random_search(target, num_attempts=100)
        if result is not None:
            best_set = result
            best_size = target

    # Final verification
    def verify(subset):
        sums = {0}
        for x in subset:
            v = vals[x]
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums |= new_sums
        return len(sums) == (1 << len(subset))

    if not verify(best_set):
        # Fallback to a known valid set
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        best_size = len(best_set)

    return best_set
# EVOLVE-BLOCK-END
