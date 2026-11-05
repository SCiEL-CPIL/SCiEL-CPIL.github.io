import time
import random
import math

N = 256
MAX_EACH = 64
DEADLINE_MARGIN = 10.0  # safety margin before 90s limit


def _is_sidon(S):
    """Check if S is a Sidon set (pair sums including a+a unique)."""
    sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _diffs(S):
    """Nonzero difference set of S."""
    d = set()
    for i in range(len(S)):
        for j in range(len(S)):
            if i != j:
                d.add(S[i] - S[j])
    return d


def _valid_pair(A, B):
    """Check both Sidon and disjoint nonzero differences."""
    if not A or not B:
        return False
    if not _is_sidon(A) or not _is_sidon(B):
        return False
    dA = _diffs(A)
    dB = _diffs(B)
    return len(dA & dB) == 0


def _greedy_build(candidates_order, prefer_A_first=True, deadline=None):
    """Greedy build: for each candidate in order, try adding to A or B."""
    A = []
    B = []
    A_sums = set()
    B_sums = set()
    A_diffs = set()
    B_diffs = set()

    def can_add_to_A(x):
        for a in A:
            s = x + a
            if s in A_sums:
                return False
        if 2 * x in A_sums:
            return False
        for a in A:
            d = x - a
            if d in B_diffs:
                return False
        return True

    def can_add_to_B(x):
        for b in B:
            s = x + b
            if s in B_sums:
                return False
        if 2 * x in B_sums:
            return False
        for b in B:
            d = x - b
            if d in A_diffs:
                return False
        return True

    for x in candidates_order:
        if deadline is not None and time.time() > deadline:
            break
        if len(A) >= MAX_EACH and len(B) >= MAX_EACH:
            break
        okA = len(A) < MAX_EACH and can_add_to_A(x)
        okB = len(B) < MAX_EACH and can_add_to_B(x)
        if okA and okB:
            # Prefer to balance sizes; add to smaller
            if len(A) <= len(B) and prefer_A_first:
                A.append(x)
                for a in A[:-1]:
                    A_sums.add(x + a)
                    A_diffs.add(x - a)
                    A_diffs.add(a - x)
                A_sums.add(2 * x)
            else:
                B.append(x)
                for b in B[:-1]:
                    B_sums.add(x + b)
                    B_diffs.add(x - b)
                    B_diffs.add(b - x)
                B_sums.add(2 * x)
        elif okA:
            A.append(x)
            for a in A[:-1]:
                A_sums.add(x + a)
                A_diffs.add(x - a)
                A_diffs.add(a - x)
            A_sums.add(2 * x)
        elif okB:
            B.append(x)
            for b in B[:-1]:
                B_sums.add(x + b)
                B_diffs.add(x - b)
                B_diffs.add(b - x)
            B_sums.add(2 * x)
    return A, B


def _local_search(A, B, max_iters=2000, deadline=None):
    """Incremental local search: try adding new elements or moving existing ones."""
    A = list(A)
    B = list(B)

    if not A or not B:
        return A, B

    # Build incremental state
    A_sums = set()
    B_sums = set()
    A_diffs = set()
    B_diffs = set()

    for i in range(len(A)):
        for j in range(i, len(A)):
            A_sums.add(A[i] + A[j])
    for i in range(len(B)):
        for j in range(i, len(B)):
            B_sums.add(B[i] + B[j])
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j:
                A_diffs.add(A[i] - A[j])
    for i in range(len(B)):
        for j in range(len(B)):
            if i != j:
                B_diffs.add(B[i] - B[j])

    # Verify initial validity
    if A_diffs & B_diffs:
        return A, B

    best_A, best_B = list(A), list(B)
    best_score = len(A) + len(B)

    all_vals = set(range(1, N + 1))
    used = set(A) | set(B)
    unused_list = list(all_vals - used)

    def try_add_to_A(x):
        """Check if x can be added to A; returns True if valid."""
        if len(A) >= MAX_EACH:
            return False
        for a in A:
            if (x + a) in A_sums:
                return False
            if (x - a) in B_diffs:
                return False
        if (2 * x) in A_sums:
            return False
        return True

    def try_add_to_B(x):
        """Check if x can be added to B; returns True if valid."""
        if len(B) >= MAX_EACH:
            return False
        for b in B:
            if (x + b) in B_sums:
                return False
            if (x - b) in A_diffs:
                return False
        if (2 * x) in B_sums:
            return False
        return True

    def commit_add_to_A(x):
        A.append(x)
        for a in A[:-1]:
            A_sums.add(x + a)
            A_diffs.add(x - a)
            A_diffs.add(a - x)
        A_sums.add(2 * x)

    def commit_add_to_B(x):
        B.append(x)
        for b in B[:-1]:
            B_sums.add(x + b)
            B_diffs.add(x - b)
            B_diffs.add(b - x)
        B_sums.add(2 * x)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        # Try adding new unused elements (most impactful for score)
        if unused_list:
            # Try a few random candidates for speed
            attempts = min(5, len(unused_list))
            for _ in range(attempts):
                if not unused_list:
                    break
                idx = random.randrange(len(unused_list))
                x = unused_list[idx]

                if try_add_to_A(x):
                    commit_add_to_A(x)
                    unused_list.pop(idx)
                    s = len(A) + len(B)
                    if s > best_score:
                        best_score = s
                        best_A, best_B = list(A), list(B)
                    break
                elif try_add_to_B(x):
                    commit_add_to_B(x)
                    unused_list.pop(idx)
                    s = len(A) + len(B)
                    if s > best_score:
                        best_score = s
                        best_A, best_B = list(A), list(B)
                    break

        # If both sets are full, try swapping to free up space or improve
        if len(A) >= MAX_EACH and len(B) >= MAX_EACH:
            # Try moving an element from larger set to smaller if possible
            if len(A) > len(B) and len(B) < MAX_EACH:
                a = random.choice(A)
                # Check if removing a from A and adding to B is valid
                # This is complex; skip for now
                pass

    return best_A, best_B


def construct():
    deadline = time.time() + (90.0 - DEADLINE_MARGIN)
    best_size = 0
    best_sets = ([], [])

    # Strategy 1: Deterministic greedy with different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        [x for x in range(1, N + 1) if x % 2 == 1] + [x for x in range(1, N + 1) if x % 2 == 0],
        [x for x in range(N, 0, -1) if x % 2 == 0] + [x for x in range(N, 0, -1) if x % 2 == 1],
        list(range(1, 65)) + list(range(65, 129)) + list(range(129, 193)) + list(range(193, 257)),
        list(range(193, 257)) + list(range(129, 193)) + list(range(65, 129)) + list(range(1, 65)),
        [x for x in range(1, N + 1) if x % 3 != 0] + [x for x in range(1, N + 1) if x % 3 == 0],
        [x for x in range(N, 0, -1) if x % 3 != 0] + [x for x in range(N, 0, -1) if x % 3 == 0],
        # New: golden-ratio-ish ordering
        [int(i * N / 257) + 1 for i in range(1, 257) if int(i * N / 257) + 1 <= N],
        # New: prime-like spacing
        [min(N, int(i * 1.618 * 100) % N + 1) for i in range(1, 257)],
    ]

    for order in orderings:
        if time.time() > deadline:
            break
        A, B = _greedy_build(order, deadline=deadline)
        if _valid_pair(A, B):
            s = len(A) + len(B)
            if s > best_size:
                best_size = s
                best_sets = (list(A), list(B))
        A2, B2 = _greedy_build(order, prefer_A_first=False, deadline=deadline)
        if _valid_pair(A2, B2):
            s = len(A2) + len(B2)
            if s > best_size:
                best_size = s
                best_sets = (list(A2), list(B2))

    # Strategy 2: Randomized greedy with multiple restarts
    rng = random.Random(42)
    num_random_tries = 300
    for t in range(num_random_tries):
        if time.time() > deadline:
            break
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        A, B = _greedy_build(candidates, deadline=deadline)
        if _valid_pair(A, B):
            s = len(A) + len(B)
            if s > best_size:
                best_size = s
                best_sets = (list(A), list(B))
            # Local search on promising solutions
            if s >= best_size - 5:
                A_ls, B_ls = _local_search(A, B, max_iters=1500, deadline=deadline)
                if _valid_pair(A_ls, B_ls):
                    s2 = len(A_ls) + len(B_ls)
                    if s2 > best_size:
                        best_size = s2
                        best_sets = (list(A_ls), list(B_ls))

    # Strategy 3: Final polish - local search on the best found with more iterations
    if best_sets[0] and best_sets[1]:
        A_ls, B_ls = _local_search(best_sets[0], best_sets[1], max_iters=10000, deadline=deadline)
        if _valid_pair(A_ls, B_ls):
            s = len(A_ls) + len(B_ls)
            if s > best_size:
                best_size = s
                best_sets = (list(A_ls), list(B_ls))

    # Fallback: ensure non-empty valid pair
    if not best_sets[0] or not best_sets[1]:
        best_sets = ([1, 10], [2, 20])
        best_size = 4

    # Final validation
    if not _valid_pair(best_sets[0], best_sets[1]):
        best_sets = ([1, 10], [2, 20])

    return [list(best_sets[0]), list(best_sets[1])]
