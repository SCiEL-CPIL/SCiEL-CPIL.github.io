# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    elements = list(range(1, n + 1))

    def _check_sidon(s):
        """Check if set s is a valid Sidon set (pair sums unique including a+a)."""
        seen = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in seen:
                    return False
                seen.add(sm)
        return True

    def _check_diffs_disjoint(A, B):
        """Check if nonzero difference sets of A and B are disjoint."""
        diffs_a = set()
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_a.add(A[j] - A[i])
        diffs_b = set()
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    diffs_b.add(B[j] - B[i])
        return not (diffs_a & diffs_b)

    def _greedy_one_ordering(order, deadline):
        """
        Single forward pass over the given ordering.
        Uses in-place tracking for O(n^2) per pass.
        Returns (setA, setB) as lists.
        """
        setA = []
        setB = []
        sumsA = set()
        sumsB = set()
        diffsA = set()
        diffsB = set()

        for x in order:
            if time.time() > deadline:
                break
            for side in (0, 1):
                if side == 0:
                    s = setA
                    sums = sumsA
                    diffs_own = diffsA
                    diffs_other = diffsB
                else:
                    s = setB
                    sums = sumsB
                    diffs_own = diffsB
                    diffs_other = diffsA

                # Check Sidon: all pair sums with x must be unique and not in existing sums
                ok = True
                new_sums = []
                for a in s:
                    sm = a + x
                    if sm in sums:
                        ok = False
                        break
                    new_sums.append(sm)
                # Also x + x
                sm = x + x
                if sm in sums:
                    ok = False
                else:
                    new_sums.append(sm)

                if not ok:
                    continue

                # Check differences: new diffs from x with existing elements must not conflict
                new_diffs = set()
                for a in s:
                    new_diffs.add(x - a)
                    new_diffs.add(a - x)

                # Check against own diffs (should be empty since we track all)
                if new_diffs & diffs_own:
                    continue

                # Check against other set's diffs
                if new_diffs & diffs_other:
                    continue

                # Add x to this set
                s.append(x)
                sums.update(new_sums)
                diffs_own.update(new_diffs)

        return setA, setB

    def _local_search(setA, setB, deadline):
        """
        Try to improve by swapping elements.
        Returns (setA, setB).
        """
        setA = list(setA)
        setB = list(setB)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding a random element to either set
            for side in (0, 1):
                if time.time() > deadline:
                    return setA, setB
                s = setA if side == 0 else setB
                other = setB if side == 0 else setA

                # Compute current sums and diffs
                sums = set()
                for i in range(len(s)):
                    for j in range(i, len(s)):
                        sums.add(s[i] + s[j])
                diffs_own = set()
                for i in range(len(s)):
                    for j in range(len(s)):
                        if i != j:
                            diffs_own.add(s[j] - s[i])
                diffs_other = set()
                for i in range(len(other)):
                    for j in range(len(other)):
                        if i != j:
                            diffs_other.add(other[j] - other[i])

                # Try to add a candidate
                candidates = [x for x in elements if x not in s]
                random.shuffle(candidates)
                for x in candidates:
                    if time.time() > deadline:
                        break
                    # Check Sidon
                    ok = True
                    new_sums = set()
                    for a in s:
                        sm = a + x
                        if sm in sums or sm in new_sums:
                            ok = False
                            break
                        new_sums.add(sm)
                    sm = x + x
                    if sm in sums or sm in new_sums:
                        ok = False
                    else:
                        new_sums.add(sm)
                    if not ok:
                        continue

                    # Check diffs
                    new_diffs = set()
                    for a in s:
                        new_diffs.add(x - a)
                        new_diffs.add(a - x)
                    if new_diffs & diffs_own:
                        continue
                    if new_diffs & diffs_other:
                        continue

                    # Add it
                    s.append(x)
                    improved = True
                    break

        return setA, setB

    best_total = -1
    best_result = ([1], [2])

    # Try many random orderings
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        order = elements[:]
        random.shuffle(order)

        setA, setB = _greedy_one_ordering(order, deadline)
        if time.time() > deadline:
            break

        # Local search to try to improve
        setA, setB = _local_search(setA, setB, deadline)

        total = len(setA) + len(setB)
        if total > best_total:
            best_total = total
            best_result = (sorted(setA), sorted(setB))

        # Early stop if we hit the max possible
        if best_total >= 128:
            break

    return list(best_result)

# EVOLVE-BLOCK-END
