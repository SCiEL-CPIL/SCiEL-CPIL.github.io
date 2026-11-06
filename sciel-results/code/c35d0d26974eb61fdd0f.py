import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    N = 256
    MAX_SUM = 2 * N
    MAX_DIFF = N - 1

    best_size = 0
    best_A = []
    best_B = []

    def build_set(order, forbidden_diffs_mask, max_size, existing_sums_mask=0, existing_set=None):
        """
        Greedily builds a Sidon set from 'order'
        """
        S = list(existing_set) if existing_set else []
        sums_mask = existing_sums_mask
        diffs_mask = 0

        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = abs(S[i] - S[j])
                diffs_mask |= (1 << d)

        for x in order:
            if len(S) >= max_size:
                break
            if x in S:
                continue

            valid = True

            if (sums_mask >> (2 * x)) & 1:
                valid = False

            if valid:
                for a in S:
                    s = x + a
                    if (sums_mask >> s) & 1:
                        valid = False
                        break

            if valid:
                for a in S:
                    d = abs(x - a)
                    if d == 0:
                        continue
                    if (forbidden_diffs_mask >> d) & 1:
                        valid = False
                        break

            if valid:
                S.append(x)
                for a in S[:-1]:
                    s = x + a
                    sums_mask |= (1 << s)
                sums_mask |= (1 << (2 * x))
                for a in S[:-1]:
                    d = abs(x - a)
                    if d > 0:
                        diffs_mask |= (1 << d)

        return S, sums_mask, diffs_mask

    def compute_diffs_mask(S):
        mask = 0
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                d = abs(S[i] - S[j])
                mask |= (1 << d)
        return mask

    def compute_sums_mask(S):
        mask = 0
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                mask |= (1 << s)
        return mask

    def polish(A, B, forbidden_A_mask, forbidden_B_mask, max_iterations=5):
        """
        Try to add elements to A and B with random orders.
        """
        improved = True
        iters = 0
        while improved and iters < max_iterations and time.time() < deadline:
            improved = False
            iters += 1

            # Try to add to A with random order
            sums_A = compute_sums_mask(A)
            diffs_A = compute_diffs_mask(A)
            candidates_A = [x for x in range(1, N + 1) if x not in A]
            random.shuffle(candidates_A)
            for x in candidates_A:
                if len(A) >= 16:
                    break
                valid = True
                if (sums_A >> (2 * x)) & 1:
                    valid = False
                if valid:
                    for a in A:
                        if (sums_A >> (x + a)) & 1:
                            valid = False
                            break
                if valid:
                    for a in A:
                        d = abs(x - a)
                        if d > 0 and (forbidden_B_mask >> d) & 1:
                            valid = False
                            break
                if valid:
                    A.append(x)
                    improved = True
                    break

            # Try to add to B with random order
            if improved:
                continue
            sums_B = compute_sums_mask(B)
            diffs_B = compute_diffs_mask(B)
            candidates_B = [x for x in range(1, N + 1) if x not in B]
            random.shuffle(candidates_B)
            for x in candidates_B:
                if len(B) >= 16:
                    break
                valid = True
                if (sums_B >> (2 * x)) & 1:
                    valid = False
                if valid:
                    for b in B:
                        if (sums_B >> (x + b)) & 1:
                            valid = False
                            break
                if valid:
                    for b in B:
                        d = abs(x - b)
                        if d > 0 and (forbidden_A_mask >> d) & 1:
                            valid = False
                            break
                if valid:
                    B.append(x)
                    improved = True
                    break

        return A, B

    while time.time() < deadline:
        order_A = list(range(1, N + 1))
        random.shuffle(order_A)
        A, sums_A, diffs_A = build_set(order_A, 0, 16)

        if len(A) == 0:
            continue

        order_B = list(range(1, N + 1))
        random.shuffle(order_B)
        B, sums_B, diffs_B = build_set(order_B, diffs_A, 16)

        current_size = len(A) + len(B)

        if current_size > best_size or (current_size >= best_size - 2 and random.random() < 0.5):
            forbidden_B = diffs_B
            forbidden_A = diffs_A
            A_pol, B_pol = polish(A, B, forbidden_A, forbidden_B)
            current_size = len(A_pol) + len(B_pol)
            if current_size > best_size:
                best_size = current_size
                best_A = A_pol
                best_B = B_pol

        if best_size >= 32:
            break

    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
