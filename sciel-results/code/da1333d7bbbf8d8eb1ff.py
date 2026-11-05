import time

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    target_diffs = set(range(1, 31))
    n_target = 30
    min_size = 9  # k=9 gives 36 pairs >= 30 required; k=8 gives 28 < 30

    # For a Sidon set of size k in [0,M], M >= k*(k-1) = 72 for k=9
    # Lower bound: with k=9, max difference M must be at least 72
    # Actually, the tightest: sum of smallest 36 distinct positive diffs >= 1+2+...+36 = 666
    # and sum of all pair diffs = sum_{i<j} (a_j - a_i) = sum_i (2i - k + 1) a_i
    # Minimum when a_i = i: sum = sum_{i<j} (j-i) = k(k^2-1)/12 = 9*80/12 = 60 < 666? No.
    # Let's just use M >= 72 as a safe lower bound and verify.

    def is_valid_sidon_covering(A):
        """Verify A is a Sidon set containing 0, covering 1..30."""
        if 0 not in A:
            return False
        diffs = set()
        A_sorted = sorted(A)
        for i in range(len(A_sorted)):
            for j in range(i + 1, len(A_sorted)):
                d = A_sorted[j] - A_sorted[i]
                if d in diffs:
                    return False
                diffs.add(d)
        for d in range(1, 31):
            if d not in diffs:
                return False
        return True

    def search_for_M(M, k):
        """
        Find a Sidon set A of size k in [0, M] containing 0 and M,
        covering all differences 1..30.
        Uses backtracking with MRV heuristic.
        """
        # Force 0 and M in the set
        A = [0, M]
        used_diffs = {M}
        missing = set(target_diffs)
        # M > 30 so M not in missing (for M >= 72)

        # We need to add (k - 2) more elements from {1, ..., M-1}
        remaining_slots = k - 2

        # Precompute: for each candidate x, which existing elements it conflicts with
        # We'll use a recursive search with forward checking

        def backtrack(A, used_diffs, missing, slots_left):
            if time.time() > deadline:
                return None

            if not missing:
                return A[:]

            if slots_left == 0:
                return None

            # Pruning: max new differences we can generate
            # Each new element x creates |A| new diffs (with each existing element)
            # Plus diffs among new elements
            # With slots_left new elements and current size len(A):
            # Max new diffs = slots_left * len(A) + slots_left * (slots_left - 1) // 2
            max_new_diffs = slots_left * len(A) + slots_left * (slots_left - 1) // 2
            if len(missing) > max_new_diffs:
                return None

            # MRV: For each candidate x in {1..M-1} \ A, compute:
            # 1. Is it valid (no diff conflicts with used_diffs or internal)?
            # 2. How many missing diffs does it cover?
            # Pick the candidate with max coverage, try it first.

            # Build candidate list
            candidates = []
            A_set = set(A)

            for x in range(1, M):
                if x in A_set:
                    continue

                # Check validity
                new_diffs = []
                valid = True
                for a in A:
                    d = abs(x - a)
                    if d in used_diffs:
                        valid = False
                        break
                    # Check for duplicate within new_diffs (Sidon: a1+a2 = 2x case)
                    if d in new_diffs:
                        valid = False
                        break
                    new_diffs.append(d)

                if not valid:
                    continue

                # Also check: new_diffs among themselves are all distinct
                # (already checked above via "d in new_diffs")

                coverage = sum(1 for d in new_diffs if d in missing)

                # Score: prefer higher coverage; tie-break by smaller x
                candidates.append((coverage, x, new_diffs))

            if not candidates:
                return None

            # Sort by coverage desc, then x asc
            candidates.sort(key=lambda c: (-c[0], c[1]))

            # Try top candidates (limit to avoid explosion)
            # Try all, but in order of preference
            for coverage, x, new_diffs in candidates:
                # Additional pruning: if coverage is 0 and we have limited slots,
                # it might still be useful for future coverage, so don't skip.
                # But if coverage == 0, deprioritize.

                new_used = used_diffs | set(new_diffs)
                new_missing = missing - set(new_diffs)
                new_A = A + [x]

                res = backtrack(new_A, new_used, new_missing, slots_left - 1)
                if res is not None:
                    return res

            return None

        result = backtrack(A, used_diffs, missing, remaining_slots)
        return result

    # Iterate M from lower bound upward
    # For k=9, M must be at least 72 (Sidon bound: M >= k(k-1) = 72)
    # Actually, the known optimal for covering 1..30 with a Sidon set:
    # Let's search M from 72 to 200, k from 9 to 12

    for M in range(72, 250):
        if time.time() > deadline:
            break

        for k in range(min_size, min(13, M + 1)):
            if time.time() > deadline:
                break

            # Quick feasibility: k elements in [0,M] Sidon requires M >= k(k-1)
            if M < k * (k - 1):
                continue

            res = search_for_M(M, k)
            if res is not None:
                # Verify
                if is_valid_sidon_covering(res):
                    return res

    # Fallback: greedy construction (should not be reached if search works)
    a = [0]
    def all_diffs(s):
        s = sorted(s)
        return [s[j] - s[i] for i in range(len(s)) for j in range(i + 1, len(s))]
    for d in range(1, 31):
        if d in set(all_diffs(a)):
            continue
        found = False
        for x in range(1, 2000):
            if x in a:
                continue
            trial = sorted(a + [x])
            diffs = all_diffs(trial)
            if len(set(diffs)) == len(diffs):
                a = trial
                found = True
                break
        if not found:
            y = 2 * max(a) + 3 * d + 1
            while True:
                trial = sorted(a + [y, y + d])
                diffs = all_diffs(trial)
                if len(set(diffs)) == len(diffs):
                    a = trial
                    break
                y += 1
    return a
