"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

def construct():
    # Deadline guard: keep total runtime well under 60 s.
    t_start = time.time()
    deadline = t_start + 55.0  # 5 s safety margin

    # We need to return exactly 12 distinct integers in [0, 128].
    # The evaluator computes the maximum subset of our 12 integers such that
    # all unordered pair sums (including i==j diagonals) are unique.
    # We want to MINIMIZE that maximum subset size.
    #
    # The seed program returns list(range(12)) = [0,1,2,...,11].
    # Its max Sidon size needs to be checked.
    #
    # Strategy:
    # 1. Define a fast is_sidon() test.
    # 2. Define a max_sidon_size() via backtracking with memoization/bounding.
    # 3. Try a curated set of "collision-heavy" sets:
    #    - Small consecutive ranges
    #    - Multiples of a small base with offsets
    #    - Arithmetic progressions
    #    - Sets clustered at small values
    # 4. For each candidate, compute max_sidon_size.
    # 5. Keep the best (minimum) found.
    # 6. Also do a brief local search around the best candidate.

    def is_sidon(vals):
        """Check if a sorted list of distinct ints has all unordered pair sums unique (incl diagonals)."""
        n = len(vals)
        sums = set()
        for i in range(n):
            for j in range(i, n):
                s = vals[i] + vals[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def max_sidon_size(vals):
        """Compute the maximum size of a Sidon subset of vals (sorted list of distinct ints)."""
        n = len(vals)
        # Sort for consistent ordering
        vals = sorted(vals)
        best = 0
        # Backtracking: try to build largest Sidon subset
        # Use pruning: if remaining elements can't beat best, stop
        def bt(idx, current, current_sums):
            nonlocal best
            if len(current) > best:
                best = len(current)
            if idx >= n:
                return
            if len(current) + (n - idx) <= best:
                return
            # Try including vals[idx]
            ok = True
            new_sums = set(current_sums)
            for j in range(len(current)):
                s = current[j] + vals[idx]
                if s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
            # Also check diagonal: vals[idx] + vals[idx]
            diag = vals[idx] + vals[idx]
            if ok and diag in new_sums:
                ok = False
            if ok:
                bt(idx + 1, current + [vals[idx]], new_sums)
            # Try excluding vals[idx]
            bt(idx + 1, current, current_sums)
        
        bt(0, [], set())
        return best

    # Candidate sets to try
    candidates = []

    # 1. Consecutive small ranges
    for start in range(0, 5):
        candidates.append(list(range(start, start + 12)))

    # 2. Multiples of small bases with small offsets
    for base in [2, 3, 4, 5, 6, 7, 8]:
        for offset in range(0, min(base, 4)):
            vals = []
            for i in range(12):
                v = i * base + offset
                if v <= 128:
                    vals.append(v)
            if len(vals) >= 12:
                candidates.append(sorted(vals[:12]))

    # 3. Arithmetic progressions with small step
    for step in range(1, 11):
        for start in range(0, min(step * 2, 10)):
            vals = [start + i * step for i in range(12)]
            if max(vals) <= 128:
                candidates.append(sorted(vals))

    # 4. Clustered sets: many small values
    #    e.g., [0,0,0,0,0,0,0,0,0,0,0,0] but must be distinct, so [0,1,2,3,4,5,6,7,8,9,10,11]
    #    Already covered by consecutive.
    #    Try sets with many small pairs: [0,1,2,3,4,5,6,7,8,9,10,12]
    candidates.append([0,1,2,3,4,5,6,7,8,9,10,12])
    candidates.append([0,1,2,3,4,5,6,7,8,9,11,12])
    candidates.append([0,1,2,3,4,5,6,7,8,10,11,12])

    # 5. "Dense lattice" sets: multiples of 2 with offsets to create collisions
    #    e.g., [0,1,2,3,4,6,7,8,9,10,12,13] - designed to have many sum collisions
    candidates.append([0,1,2,3,4,6,7,8,9,10,12,13])
    candidates.append([0,1,2,4,5,6,8,9,10,12,13,14])
    candidates.append([0,1,3,4,5,7,8,9,11,12,13,14])

    # 6. Sets based on binary representations (many collisions in sums)
    #    [0, 1, 2, 3, 4, 8, 16, 32, 64, 5, 12, 20]
    candidates.append([0,1,2,3,4,5,12,20,32,48,64,80])
    candidates.append([0,1,2,3,4,5,6,12,14,16,24,32])

    # 7. Quadratic-like sets: i^2 mod something
    for mod in [13, 17, 19, 23]:
        vals = []
        i = 0
        while len(vals) < 12:
            v = (i * i) % mod
            if v not in vals and v <= 128:
                vals.append(v)
            i += 1
        if len(vals) == 12:
            candidates.append(sorted(vals))

    # Remove duplicates
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen and len(set(c)) == 12 and all(0 <= x <= 128 for x in c):
            seen.add(key)
            unique_candidates.append(list(c))

    # Evaluate candidates
    best_set = list(range(12))  # fallback: the seed
    best_size = 999

    for c in unique_candidates:
        if time.time() > deadline:
            break
        sz = max_sidon_size(c)
        if sz < best_size:
            best_size = sz
            best_set = c

    # Brief local search around best_set: try swapping one element
    if time.time() < deadline - 5.0:
        for i in range(12):
            for new_val in range(0, 129):
                if new_val in best_set and new_val != best_set[i]:
                    continue
                candidate = best_set[:]
                candidate[i] = new_val
                if len(set(candidate)) != 12:
                    continue
                if any(not (0 <= x <= 128) for x in candidate):
                    continue
                if time.time() > deadline:
                    break
                sz = max_sidon_size(sorted(candidate))
                if sz < best_size:
                    best_size = sz
                    best_set = candidate
            if time.time() > deadline:
                break

    return sorted(best_set)
# EVOLVE-BLOCK-END
