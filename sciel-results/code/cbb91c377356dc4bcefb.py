"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0
    best_set = []

    def can_add_optimized(selected, sums, x):
        # Check 1: 3x
        if 3*x in sums:
            return False

        new_sums = set()
        new_sums.add(3*x)

        # Check 2: y + 2x and 2y + x
        for y in selected:
            s = y + 2*x
            if s in sums or s in new_sums:
                return False
            new_sums.add(s)

            s2 = 2*y + x
            if s2 in sums or s2 in new_sums:
                return False
            new_sums.add(s2)

        # Check 3: y + z + x
        n = len(selected)
        for i in range(n):
            y = selected[i]
            for j in range(i+1, n):
                z = selected[j]
                s = y + z + x
                if s in sums or s in new_sums:
                    return False
                new_sums.add(s)

        return True

    def greedy_asc():
        selected = []
        sums = set()
        for x in range(1, 4097):
            if time.time() > deadline:
                break
            if can_add_optimized(selected, sums, x):
                selected.append(x)
                sums.add(3*x)
                for y in selected[:-1]:
                    sums.add(y + 2*x)
                    sums.add(2*y + x)
                for i in range(len(selected)-1):
                    for j in range(i+1, len(selected)-1):
                        sums.add(selected[i] + selected[j] + x)
        return selected

    def greedy_desc():
        selected = []
        sums = set()
        for x in range(4096, 0, -1):
            if time.time() > deadline:
                break
            if can_add_optimized(selected, sums, x):
                selected.append(x)
                sums.add(3*x)
                for y in selected[:-1]:
                    sums.add(y + 2*x)
                    sums.add(2*y + x)
                for i in range(len(selected)-1):
                    for j in range(i+1, len(selected)-1):
                        sums.add(selected[i] + selected[j] + x)
        return selected

    def greedy_geometric_fill():
        # Start with a geometric-like skeleton, then fill in gaps
        selected = []
        sums = set()

        # Try to build a set with roughly geometric spacing
        # Start with small numbers, then add larger ones
        candidates = list(range(1, 4097))
        random.shuffle(candidates)

        # First, try to add some small numbers greedily
        for x in range(1, 200):
            if time.time() > deadline:
                break
            if can_add_optimized(selected, sums, x):
                selected.append(x)
                sums.add(3*x)
                for y in selected[:-1]:
                    sums.add(y + 2*x)
                    sums.add(2*y + x)
                for i in range(len(selected)-1):
                    for j in range(i+1, len(selected)-1):
                        sums.add(selected[i] + selected[j] + x)

        # Then try to add from the shuffled candidates
        for x in candidates:
            if time.time() > deadline:
                break
            if x in selected:
                continue
            if can_add_optimized(selected, sums, x):
                selected.append(x)
                sums.add(3*x)
                for y in selected[:-1]:
                    sums.add(y + 2*x)
                    sums.add(2*y + x)
                for i in range(len(selected)-1):
                    for j in range(i+1, len(selected)-1):
                        sums.add(selected[i] + selected[j] + x)

        return selected

    def greedy_randomized():
        selected = []
        sums = set()
        candidates = list(range(1, 4097))
        random.shuffle(candidates)

        for x in candidates:
            if time.time() > deadline:
                break
            if can_add_optimized(selected, sums, x):
                selected.append(x)
                sums.add(3*x)
                for y in selected[:-1]:
                    sums.add(y + 2*x)
                    sums.add(2*y + x)
                for i in range(len(selected)-1):
                    for j in range(i+1, len(selected)-1):
                        sums.add(selected[i] + selected[j] + x)

        return selected

    # Try ascending first
    res_asc = greedy_asc()
    if len(res_asc) > len(best_set):
        best_set = res_asc

    # Try descending if time allows
    if time.time() < deadline - 10.0:
        res_desc = greedy_desc()
        if len(res_desc) > len(best_set):
            best_set = res_desc

    # Try geometric fill if time allows
    if time.time() < deadline - 10.0:
        res_geo = greedy_geometric_fill()
        if len(res_geo) > len(best_set):
            best_set = res_geo

    # Try randomized if time allows
    if time.time() < deadline - 10.0:
        res_rand = greedy_randomized()
        if len(res_rand) > len(best_set):
            best_set = res_rand

    return best_set
# EVOLVE-BLOCK-END
