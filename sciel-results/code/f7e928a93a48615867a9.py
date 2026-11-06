import random
import time

def construct():
    best_A = []
    best_B = []
    best_size = 0

    def greedy_construct():
        A = []
        B = []
        sums_A_mask = 0
        sums_B_mask = 0
        A_mask = 0
        B_mask = 0
        diffs_A_mask = 0
        diffs_B_mask = 0

        candidates = list(range(1, 257))
        random.shuffle(candidates)

        for x in candidates:
            can_add_A = True
            if (sums_A_mask & (A_mask << x)) != 0:
                can_add_A = False
            elif (sums_A_mask & (1 << (2 * x))) != 0:
                can_add_A = False
            else:
                if A:
                    for a in A:
                        d = x - a
                        if (diffs_B_mask >> (d + 256)) & 1:
                            can_add_A = False
                            break

            can_add_B = True
            if (sums_B_mask & (B_mask << x)) != 0:
                can_add_B = False
            elif (sums_B_mask & (1 << (2 * x))) != 0:
                can_add_B = False
            else:
                if B:
                    for b in B:
                        d = x - b
                        if (diffs_A_mask >> (d + 256)) & 1:
                            can_add_B = False
                            break

            if can_add_A and can_add_B:
                if len(A) <= len(B):
                    A.append(x)
                    A_mask |= (1 << x)
                    sums_A_mask |= (A_mask << x) | (1 << (2 * x))
                    for a in A[:-1]:
                        d1 = x - a
                        d2 = a - x
                        diffs_A_mask |= (1 << (d1 + 256)) | (1 << (d2 + 256))
                else:
                    B.append(x)
                    B_mask |= (1 << x)
                    sums_B_mask |= (B_mask << x) | (1 << (2 * x))
                    for b in B[:-1]:
                        d1 = x - b
                        d2 = b - x
                        diffs_B_mask |= (1 << (d1 + 256)) | (1 << (d2 + 256))
            elif can_add_A:
                A.append(x)
                A_mask |= (1 << x)
                sums_A_mask |= (A_mask << x) | (1 << (2 * x))
                for a in A[:-1]:
                    d1 = x - a
                    d2 = a - x
                    diffs_A_mask |= (1 << (d1 + 256)) | (1 << (d2 + 256))
            elif can_add_B:
                B.append(x)
                B_mask |= (1 << x)
                sums_B_mask |= (B_mask << x) | (1 << (2 * x))
                for b in B[:-1]:
                    d1 = x - b
                    d2 = b - x
                    diffs_B_mask |= (1 << (d1 + 256)) | (1 << (d2 + 256))

        return A, B, sums_A_mask, sums_B_mask, A_mask, B_mask, diffs_A_mask, diffs_B_mask

    def local_search(A, B, sums_A, sums_B, A_mask, B_mask, diffs_A, diffs_B, time_limit=5):
        start_time = time.time()
        
        def can_add_to_A(x):
            if (sums_A & (A_mask << x)) != 0: return False
            if (sums_A & (1 << (2 * x))) != 0: return False
            for a in A:
                d = x - a
                if (diffs_B >> (d + 256)) & 1: return False
            return True

        def can_add_to_B(x):
            if (sums_B & (B_mask << x)) != 0: return False
            if (sums_B & (1 << (2 * x))) != 0: return False
            for b in B:
                d = x - b
                if (diffs_A >> (d + 256)) & 1: return False
            return True

        def add_to_A(x):
            nonlocal A, sums_A, A_mask, diffs_A
            A.append(x)
            A_mask |= (1 << x)
            sums_A |= (A_mask << x) | (1 << (2 * x))
            for a in A[:-1]:
                d1 = x - a
                d2 = a - x
                diffs_A |= (1 << (d1 + 256)) | (1 << (d2 + 256))

        def add_to_B(x):
            nonlocal B, sums_B, B_mask, diffs_B
            B.append(x)
            B_mask |= (1 << x)
            sums_B |= (B_mask << x) | (1 << (2 * x))
            for b in B[:-1]:
                d1 = x - b
                d2 = b - x
                diffs_B |= (1 << (d1 + 256)) | (1 << (d2 + 256))

        def remove_from_A(x):
            nonlocal A, sums_A, A_mask, diffs_A
            A.remove(x)
            A_mask = 0
            sums_A = 0
            diffs_A = 0
            for a in A:
                A_mask |= (1 << a)
                for b in A:
                    if a <= b:
                        sums_A |= (1 << (a + b))
                    d = a - b
                    if d != 0:
                        diffs_A |= (1 << (d + 256))

        def remove_from_B(x):
            nonlocal B, sums_B, B_mask, diffs_B
            B.remove(x)
            B_mask = 0
            sums_B = 0
            diffs_B = 0
            for b in B:
                B_mask |= (1 << b)
                for c in B:
                    if b <= c:
                        sums_B |= (1 << (b + c))
                    d = b - c
                    if d != 0:
                        diffs_B |= (1 << (d + 256))

        candidates = list(range(1, 257))
        
        while time.time() - start_time < time_limit:
            improved = False
            random.shuffle(candidates)
            for x in candidates:
                if x in A or x in B: continue
                if can_add_to_A(x):
                    add_to_A(x)
                    improved = True
                    break
                elif can_add_to_B(x):
                    add_to_B(x)
                    improved = True
                    break
            
            if improved: continue

            if A:
                idx = random.randint(0, len(A) - 1)
                x_rm = A[idx]
                remove_from_A(x_rm)
                random.shuffle(candidates)
                added = False
                for x in candidates:
                    if x == x_rm or x in A: continue
                    if can_add_to_A(x):
                        add_to_A(x)
                        added = True
                        break
                    elif can_add_to_B(x):
                        add_to_B(x)
                        added = True
                        break
                if not added:
                    add_to_A(x_rm)

        return len(A) + len(B), A, B

    A, B, sA, sB, mA, mB, dA, dB = greedy_construct()
    best_size = len(A) + len(B)
    best_A, best_B = A, B

    size, A_ls, B_ls = local_search(A, B, sA, sB, mA, mB, dA, dB, 5)
    if size > best_size:
        best_size = size
        best_A, best_B = A_ls, B_ls

    for _ in range(5):
        A, B, sA, sB, mA, mB, dA, dB = greedy_construct()
        if len(A) + len(B) > best_size:
            best_size = len(A) + len(B)
            best_A, best_B = A, B
        size, A_ls, B_ls = local_search(A, B, sA, sB, mA, mB, dA, dB, 2)
        if size > best_size:
            best_size = size
            best_A, best_B = A_ls, B_ls

    if not best_A: best_A = [1]
    if not best_B: best_B = [2]
    return [best_A, best_B]
