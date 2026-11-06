import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    N = 256
    best_size = 0
    best_A = []
    best_B = []

    def build_set(order, forbidden_diffs_mask, max_size=16):
        """
        Greedily builds a Sidon set from 'order'
        such that its differences do not intersect with forbidden_diffs_mask.
        Returns the set.
        """
        S = []
        sums_mask = 0
        diffs_mask = 0

        for x in order:
            if len(S) >= max_size:
                break

            # Check Sidon condition
            valid = True
            temp_sums = 0

            # Check sums
            for a in S:
                s = x + a
                if (sums_mask >> s) & 1:
                    valid = False
                    break
                temp_sums |= (1 << s)

            if valid:
                if (sums_mask >> (2*x)) & 1:
                    valid = False
                else:
                    temp_sums |= (1 << (2*x))

            # Check diff conflicts with forbidden
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
                sums_mask |= temp_sums

        return S

    def get_diffs_mask(S):
        mask = 0
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                d = abs(S[i] - S[j])
                mask |= (1 << d)
        return mask

    def can_add_to_sidon(S, x, sums_mask, forbidden_diffs_mask):
        """Check if x can be added to S maintaining Sidon and diff constraints."""
        for a in S:
            s = x + a
            if (sums_mask >> s) & 1:
                return False
        if (sums_mask >> (2*x)) & 1:
            return False
        for a in S:
            d = abs(x - a)
            if d > 0 and (forbidden_diffs_mask >> d) & 1:
                return False
        return True

    def local_search(A, B, max_iter=200):
        """Try to grow A and B by swapping elements."""
        A = list(A)
        B = list(B)
        diffs_A = get_diffs_mask(A)
        diffs_B = get_diffs_mask(B)

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            # Try to add an element to A
            added = False
            for x in range(1, N + 1):
                if x in A:
                    continue
                sums_A = 0
                for a in A:
                    sums_A |= (1 << (a + x)) | (1 << (2*a))
                # Recompute properly
                sums_A = 0
                for i in range(len(A)):
                    for j in range(i, len(A)):
                        sums_A |= (1 << (A[i] + A[j]))

                if can_add_to_sidon(A, x, sums_A, diffs_B):
                    new_diffs_A = 0
                    for a in A:
                        d = abs(x - a)
                        if d > 0:
                            new_diffs_A |= (1 << d)
                    if not (new_diffs_A & diffs_B):
                        A.append(x)
                        diffs_A = get_diffs_mask(A)
                        added = True
                        break

            # Try to add an element to B
            if not added:
                for x in range(1, N + 1):
                    if x in B:
                        continue
                    sums_B = 0
                    for i in range(len(B)):
                        for j in range(i, len(B)):
                            sums_B |= (1 << (B[i] + B[j]))

                    if can_add_to_sidon(B, x, sums_B, diffs_A):
                        new_diffs_B = 0
                        for b in B:
                            d = abs(x - b)
                            if d > 0:
                                new_diffs_B |= (1 << d)
                        if not (new_diffs_B & diffs_A):
                            B.append(x)
                            diffs_B = get_diffs_mask(B)
                            break

            # Try swapping: remove from A, add to B
            if not added:
                swapped = False
                for i, a in enumerate(A):
                    A_temp = A[:i] + A[i+1:]
                    diffs_A_temp = get_diffs_mask(A_temp)
                    for x in range(1, N + 1):
                        if x in B:
                            continue
                        sums_B = 0
                        for ii in range(len(B)):
                            for jj in range(ii, len(B)):
                                sums_B |= (1 << (B[ii] + B[jj]))
                        if can_add_to_sidon(B, x, sums_B, diffs_A_temp):
                            new_diffs_B = 0
                            for b in B:
                                d = abs(x - b)
                                if d > 0:
                                    new_diffs_B |= (1 << d)
                            if not (new_diffs_B & diffs_A_temp):
                                # Also check if removing a from A keeps it Sidon
                                sums_A_temp = 0
                                for ii in range(len(A_temp)):
                                    for jj in range(ii, len(A_temp)):
                                        sums_A_temp |= (1 << (A_temp[ii] + A_temp[jj]))
                                # A_temp is already Sidon since we removed an element
                                B.append(x)
                                A = A_temp
                                diffs_A = diffs_A_temp
                                diffs_B = get_diffs_mask(B)
                                swapped = True
                                break
                        if swapped:
                            break
                if swapped:
                    continue

            # If we couldn't add or swap, try a random restart of local search
            if not added and not swapped:
                break

        return A, B

    # We want to maximize |A|+|B|.
    # Max possible is 32.

    # Strategy:
    # 1. Try to find A of size 16.
    # 2. Then find B of size 16 avoiding D_A.
    # 3. Apply local search to try to grow further.

    while time.time() < deadline:
        # Generate random A
        order_A = list(range(1, N + 1))
        random.shuffle(order_A)
        A = build_set(order_A, 0, 16)

        if len(A) == 0:
            continue

        # Compute D_A mask
        diffs_A_mask = get_diffs_mask(A)

        # Generate random B avoiding D_A
        order_B = list(range(1, N + 1))
        random.shuffle(order_B)
        B = build_set(order_B, diffs_A_mask, 16)

        current_size = len(A) + len(B)

        # Try local search to improve
        if time.time() < deadline - 5.0:
            A_ls, B_ls = local_search(A, B, 100)
            ls_size = len(A_ls) + len(B_ls)
            if ls_size > current_size:
                A, B = A_ls, B_ls
                current_size = ls_size

        if current_size > best_size:
            best_size = current_size
            best_A = A
            best_B = B

        # If we hit 32, we can break?
        # 32 is the theoretical max.
        if best_size == 32:
            break

    # Fallback if nothing found (should not happen)
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
