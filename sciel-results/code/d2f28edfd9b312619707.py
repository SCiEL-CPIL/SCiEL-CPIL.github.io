"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 78  # 78s safety margin for 90s limit

    n = 256
    max_size = 64

    # ---------- Fast incremental state ----------
    def rebuild(A, B):
        """Build sums and diffs sets from A and B."""
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_A.add(A[i] + A[j])
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_A.add(A[j] - A[i])
        for i in range(len(B)):
            for j in range(i, len(B)):
                sums_B.add(B[i] + B[j])
        for i in range(len(B)):
            for j in range(len(B)):
                if i != j:
                    diffs_B.add(B[j] - B[i])
        return sums_A, sums_B, diffs_A, diffs_B

    def can_add(x, A, sums_A, diffs_A, diffs_other):
        """Check if x can be added to A maintaining Sidon + diff disjointness."""
        for elem in A:
            if (x + elem) in sums_A:
                return False
            if (x - elem) in diffs_other or (elem - x) in diffs_other:
                return False
        if (x + x) in sums_A:
            return False
        return True

    def add_to(A, x, sums_set, diffs_set):
        """Add x to A, updating sums and diffs incrementally."""
        A.append(x)
        for elem in A[:-1]:
            sums_set.add(x + elem)
            diffs_set.add(x - elem)
            diffs_set.add(elem - x)
        sums_set.add(x + x)

    def remove_from(A, x, sums_set, diffs_set):
        """Remove x from A, rebuilding sums and diffs."""
        A.remove(x)
        sums_set.clear()
        diffs_set.clear()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums_set.add(A[i] + A[j])
        for i in range(len(A)):
            for j in range(len(A)):
                if i != j:
                    diffs_set.add(A[j] - A[i])

    # ---------- Greedy construction with conflict scoring ----------
    def greedy_construct(order, max_size):
        """Greedy: for each x in order, pick the side (A or B) with fewer conflicts."""
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()

        for x in order:
            if len(A) >= max_size and len(B) >= max_size:
                break

            best_side = None
            best_conflicts = float('inf')

            # Try adding to A
            if len(A) < max_size and can_add(x, A, sums_A, diffs_A, diffs_B):
                # Count how many future conflicts this creates (heuristic)
                conflicts = 0
                for elem in A:
                    # x-elem and elem-x create new diffs; count overlap with diffs_B
                    if (x - elem) in diffs_B:
                        conflicts += 1
                    if (elem - x) in diffs_B:
                        conflicts += 1
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_side = 0

            # Try adding to B
            if len(B) < max_size and can_add(x, B, sums_B, diffs_B, diffs_A):
                conflicts = 0
                for elem in B:
                    if (x - elem) in diffs_A:
                        conflicts += 1
                    if (elem - x) in diffs_A:
                        conflicts += 1
                if conflicts < best_conflicts:
                    best_conflicts = conflicts
                    best_side = 1

            if best_side == 0:
                add_to(A, x, sums_A, diffs_A)
            elif best_side == 1:
                add_to(B, x, sums_B, diffs_B)

        return A, B

    # ---------- Polish: try to add more elements ----------
    def polish(A, B, deadline):
        """Try adding remaining elements in random orders."""
        A = A[:]
        B = B[:]
        sums_A, sums_B, diffs_A, diffs_B = rebuild(A, B)
        in_A = set(A)
        in_B = set(B)
        remaining = [x for x in range(1, n + 1) if x not in in_A and x not in in_B]

        for _ in range(8):
            if time.time() > deadline:
                break
            random.shuffle(remaining)
            for x in remaining:
                if time.time() > deadline:
                    break
                if len(A) >= max_size and len(B) >= max_size:
                    break
                added = False
                if len(A) < max_size and can_add(x, A, sums_A, diffs_A, diffs_B):
                    add_to(A, x, sums_A, diffs_A)
                    in_A.add(x)
                    added = True
                if not added and len(B) < max_size and can_add(x, B, sums_B, diffs_B, diffs_A):
                    add_to(B, x, sums_B, diffs_B)
                    in_B.add(x)
                    added = True

        return A, B

    # ---------- Local search: swap and transfer moves ----------
    def local_search(A, B, deadline):
        """Improve by swapping elements between A and B, or transferring single elements."""
        A = A[:]
        B = B[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Swap move: A[i] <-> B[j]
            for i in range(len(A)):
                if time.time() > deadline:
                    break
                for j in range(len(B)):
                    if time.time() > deadline:
                        break
                    a_val = A[i]
                    b_val = B[j]
                    A_new = A[:i] + A[i+1:] + [b_val]
                    B_new = B[:j] + B[j+1:] + [a_val]
                    # Check Sidon for A_new
                    ok = True
                    sums = set()
                    for ii in range(len(A_new)):
                        for jj in range(ii, len(A_new)):
                            sv = A_new[ii] + A_new[jj]
                            if sv in sums:
                                ok = False
                                break
                            sums.add(sv)
                        if not ok:
                            break
                    if ok:
                        dA = set()
                        for ii in range(len(A_new)):
                            for jj in range(len(A_new)):
                                if ii != jj:
                                    dA.add(A_new[jj] - A_new[ii])
                        # Check Sidon for B_new
                        sums2 = set()
                        for ii in range(len(B_new)):
                            for jj in range(ii, len(B_new)):
                                sv = B_new[ii] + B_new[jj]
                                if sv in sums2:
                                    ok = False
                                    break
                                sums2.add(sv)
                            if not ok:
                                break
                    if ok:
                        dB = set()
                        for ii in range(len(B_new)):
                            for jj in range(len(B_new)):
                                if ii != jj:
                                    dB.add(B_new[jj] - B_new[ii])
                        if not dA & dB:
                            A, B = A_new, B_new
                            improved = True
                            break
                if improved:
                    break

            # Transfer move: move A[i] to B (if B has room and it fits)
            if not improved and time.time() < deadline:
                for i in range(len(A)):
                    if time.time() > deadline:
                        break
                    if len(B) >= max_size:
                        break
                    a_val = A[i]
                    A_new = A[:i] + A[i+1:]
                    B_new = B + [a_val]
                    # Check Sidon for A_new
                    ok = True
                    sums = set()
                    for ii in range(len(A_new)):
                        for jj in range(ii, len(A_new)):
                            sv = A_new[ii] + A_new[jj]
                            if sv in sums:
                                ok = False
                                break
                            sums.add(sv)
                        if not ok:
                            break
                    if ok:
                        dA = set()
                        for ii in range(len(A_new)):
                            for jj in range(len(A_new)):
                                if ii != jj:
                                    dA.add(A_new[jj] - A_new[ii])
                        # Check Sidon for B_new
                        sums2 = set()
                        for ii in range(len(B_new)):
                            for jj in range(ii, len(B_new)):
                                sv = B_new[ii] + B_new[jj]
                                if sv in sums2:
                                    ok = False
                                    break
                                sums2.add(sv)
                            if not ok:
                                break
                    if ok:
                        dB = set()
                        for ii in range(len(B_new)):
                            for jj in range(len(B_new)):
                                if ii != jj:
                                    dB.add(B_new[jj] - B_new[ii])
                        if not dA & dB:
                            A, B = A_new, B_new
                            improved = True
                            break

            # Transfer move: move B[j] to A (if A has room and it fits)
            if not improved and time.time() < deadline:
                for j in range(len(B)):
                    if time.time() > deadline:
                        break
                    if len(A) >= max_size:
                        break
                    b_val = B[j]
                    B_new = B[:j] + B[j+1:]
                    A_new = A + [b_val]
                    ok = True
                    sums = set()
                    for ii in range(len(A_new)):
                        for jj in range(ii, len(A_new)):
                            sv = A_new[ii] + A_new[jj]
                            if sv in sums:
                                ok = False
                                break
                            sums.add(sv)
                        if not ok:
                            break
                    if ok:
                        dA = set()
                        for ii in range(len(A_new)):
                            for jj in range(len(A_new)):
                                if ii != jj:
                                    dA.add(A_new[jj] - A_new[ii])
                        sums2 = set()
                        for ii in range(len(B_new)):
                            for jj in range(ii, len(B_new)):
                                sv = B_new[ii] + B_new[jj]
                                if sv in sums2:
                                    ok = False
                                    break
                                sums2.add(sv)
                            if not ok:
                                break
                    if ok:
                        dB = set()
                        for ii in range(len(B_new)):
                            for jj in range(len(B_new)):
                                if ii != jj:
                                    dB.add(B_new[jj] - B_new[ii])
                        if not dA & dB:
                            A, B = A_new, B_new
                            improved = True
                            break

        return A, B

    # ---------- Main: multiple restarts ----------
    best_A, best_B = [], []
    best_score = 0

    num_trials = 40
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break

        order = list(range(1, n + 1))
        random.shuffle(order)

        A, B = greedy_construct(order, max_size)

        if time.time() < deadline - 10:
            A, B = polish(A, B, deadline - 8)

        if time.time() < deadline - 3:
            A, B = local_search(A, B, deadline - 2)

        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A[:], B[:]

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
