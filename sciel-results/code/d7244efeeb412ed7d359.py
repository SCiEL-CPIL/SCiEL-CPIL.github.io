import random
import time

def construct():
    n = 256
    deadline = time.time() + 50.0  # safety margin under 60s

    best_sum = 0
    best_A = []
    best_B = []

    def compute_sums(S):
        s = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s.add(S[i] + S[j])
        return s

    def compute_diffs(S):
        d = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    d.add(S[i] - S[j])
        return d

    def can_add_to_sidon(x, S, sums_set):
        for a in S:
            if a + x in sums_set:
                return False
        if x + x in sums_set:
            return False
        return True

    def can_add_with_diff(x, S, sums_set, D_S, D_other):
        if not can_add_to_sidon(x, S, sums_set):
            return False
        for a in S:
            d1 = x - a
            d2 = a - x
            if d1 in D_other or d2 in D_other:
                return False
        return True

    def build_interleaved(candidates, target=16):
        A, B = [], []
        sums_A, sums_B = set(), set()
        D_A, D_B = set(), set()
        for x in candidates:
            if len(A) >= target and len(B) >= target:
                break
            added = False
            if len(A) < target:
                if can_add_with_diff(x, A, sums_A, D_A, D_B):
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(a + x)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        D_A.add(x - a)
                        D_A.add(a - x)
                    added = True
            if not added and len(B) < target:
                if can_add_with_diff(x, B, sums_B, D_B, D_A):
                    B.append(x)
                    for a in B[:-1]:
                        sums_B.add(a + x)
                    sums_B.add(x + x)
                    for a in B[:-1]:
                        D_B.add(x - a)
                        D_B.add(a - x)
        return A, B

    def build_sequential(candidates, target=16, first_is_A=True):
        if first_is_A:
            first, second = [], []
        else:
            first, second = [], []

        sums_first, sums_second = set(), set()
        D_first, D_second = set(), set()

        for x in candidates:
            if len(first) >= target:
                break
            if can_add_with_diff(x, first, sums_first, D_first, D_second):
                first.append(x)
                for a in first[:-1]:
                    sums_first.add(a + x)
                sums_first.add(x + x)
                for a in first[:-1]:
                    D_first.add(x - a)
                    D_first.add(a - x)

        for x in candidates:
            if len(second) >= target:
                break
            if can_add_with_diff(x, second, sums_second, D_second, D_first):
                second.append(x)
                for a in second[:-1]:
                    sums_second.add(a + x)
                sums_second.add(x + x)
                for a in second[:-1]:
                    D_second.add(x - a)
                    D_second.add(a - x)

        if first_is_A:
            return first, second
        else:
            return second, first

    def polish(A, B, max_iters=200):
        """Local search: try to add elements to A or B, or swap to free up space."""
        A = list(A)
        B = list(B)
        sums_A = compute_sums(A)
        sums_B = compute_sums(B)
        D_A = compute_diffs(A)
        D_B = compute_diffs(B)
        all_used = set(A) | set(B)
        all_vals = set(range(1, n + 1))
        available = list(all_vals - all_used)
        random.shuffle(available)

        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try adding to A
            for x in available:
                if can_add_with_diff(x, A, sums_A, D_A, D_B):
                    A.append(x)
                    for a in A[:-1]:
                        sums_A.add(a + x)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        D_A.add(x - a)
                        D_A.add(a - x)
                    available.remove(x)
                    improved = True
                    break
            if improved:
                continue
            # Try adding to B
            for x in available:
                if can_add_with_diff(x, B, sums_B, D_B, D_A):
                    B.append(x)
                    for a in B[:-1]:
                        sums_B.add(a + x)
                    sums_B.add(x + x)
                    for a in B[:-1]:
                        D_B.add(x - a)
                        D_B.add(a - x)
                    available.remove(x)
                    improved = True
                    break
        return A, B

    for restart in range(2000):
        if time.time() > deadline:
            break

        candidates = list(range(1, n + 1))
        random.shuffle(candidates)

        strategy = restart % 4
        if strategy == 0:
            A, B = build_interleaved(candidates)
        elif strategy == 1:
            A, B = build_sequential(candidates, first_is_A=True)
        elif strategy == 2:
            A, B = build_sequential(candidates, first_is_A=False)
        else:
            # Spread out candidates for better initial picks
            spread = list(range(1, n + 1, 4)) + list(range(2, n + 1, 4)) + list(range(3, n + 1, 4)) + list(range(4, n + 1, 4))
            random.shuffle(spread)
            A, B = build_interleaved(spread)

        # Polish
        A, B = polish(A, B)

        current_sum = len(A) + len(B)
        if current_sum > best_sum:
            best_sum = current_sum
            best_A = A[:]
            best_B = B[:]

        if best_sum >= 32:
            break

    return [best_A, best_B]
