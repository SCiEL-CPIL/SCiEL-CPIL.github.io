"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 256, 'maximum_size': 60}

def construct():
    start = time.time()
    deadline = start + 85  # safety margin within 90s limit

    def is_valid(s):
        """Fast DP-based check: subset sums of different cardinalities must be disjoint."""
        n = len(s)
        # reachable[k] = set of sums achievable with exactly k elements
        reachable = [set() for _ in range(n + 1)]
        reachable[0].add(0)
        for x in s:
            # Update in reverse to avoid using x multiple times
            for k in range(n, 0, -1):
                for sm in list(reachable[k - 1]):
                    reachable[k].add(sm + x)
        # Check cross-cardinality: no sum appears in two different cardinality sets
        seen = {}
        for k in range(n + 1):
            for sm in reachable[k]:
                if sm in seen:
                    return False
                seen[sm] = k
        return True

    # Start with the maximal superincreasing set: powers of 2 up to 256
    base = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best = base[:]

    if not is_valid(base):
        return base

    # Greedy extension: try adding each remaining element in increasing order
    used = set(base)
    candidates = [x for x in range(1, 257) if x not in used]

    for x in candidates:
        if time.time() > deadline - 5:
            break
        new_set = base + [x]
        if is_valid(new_set):
            best = new_set
            base = new_set
            used.add(x)

    # Local search: try swaps to enable further extension
    for _ in range(4):
        if time.time() > deadline - 5:
            break
        improved = False
        current_set = set(best)
        # Try removing one element and adding a different one
        for elem in list(best):
            if time.time() > deadline - 5:
                break
            for x in range(1, 257):
                if time.time() > deadline - 5:
                    break
                if x in current_set:
                    continue
                new_set = sorted(list(current_set - {elem}) + [x])
                if is_valid(new_set):
                    # Check if this allows further extension
                    used2 = set(new_set)
                    candidates2 = [y for y in range(1, 257) if y not in used2]
                    extended = False
                    for y in candidates2:
                        if time.time() > deadline - 5:
                            break
                        if is_valid(new_set + [y]):
                            best = new_set + [y]
                            improved = True
                            extended = True
                            break
                    if not extended and len(new_set) == len(best):
                        # Same size but different composition; might help later
                        # Only accept if it's "better" in some sense (e.g., smaller max element)
                        pass
                    if improved:
                        current_set = set(best)
                        break
            if improved:
                break
        if not improved:
            break

    # Final greedy pass after any swaps
    used = set(best)
    candidates = [x for x in range(1, 257) if x not in used]
    for x in candidates:
        if time.time() > deadline - 5:
            break
        new_set = best + [x]
        if is_valid(new_set):
            best = new_set
            used.add(x)

    # Ensure return type is exactly list of Python ints
    result = [int(x) for x in best]
    return result
# EVOLVE-BLOCK-END
