import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin within 90s limit

    def is_valid(s):
        """Check if subset sums of different cardinalities are disjoint."""
        n = len(s)
        reachable = [set() for _ in range(n + 1)]
        reachable[0].add(0)
        for x in s:
            for k in range(n, 0, -1):
                if reachable[k - 1]:
                    for sm in reachable[k - 1]:
                        reachable[k].add(sm + x)
        seen = {}
        for k in range(n + 1):
            for sm in reachable[k]:
                if sm in seen:
                    return False
                seen[sm] = k
        return True

    # Greedy: add smallest valid element each time
    best = []
    for x in range(1, 257):
        if time.time() > deadline:
            break
        test = best + [x]
        if is_valid(test):
            best = test

    best_size = len(best)

    # Random restarts with different orderings
    rng = random.Random(42)
    for trial in range(1000):
        if time.time() > deadline:
            break
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        s = []
        for x in candidates:
            test = s + [x]
            if is_valid(test):
                s = test
        if len(s) > best_size:
            best = s
            best_size = len(s)

    # Final: try to add any remaining valid elements
    for x in range(1, 257):
        if time.time() > deadline:
            break
        if x in best:
            continue
        test = best + [x]
        if is_valid(test):
            best = test
            best_size = len(best)

    # Validate
    assert is_valid(best), "Final set is not valid"
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in best)
    assert all(1 <= x <= 256 for x in best)
    assert len(set(best)) == len(best), "Elements not distinct"
    assert len(best) <= 60

    return list(best)

