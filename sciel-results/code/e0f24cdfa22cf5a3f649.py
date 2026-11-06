"""Two Sidon sets with disjoint difference sets – bitset-accelerated greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin before 60s kill


def _build_sum_diff_bits(elems):
    """Build bitsets for pair sums (with replacement) and nonzero differences."""
    sum_bits = 0
    for a, b in combinations_with_replacement(elems, 2):
        s = a + b
        if s <= 512:
            sum_bits |= (1 << s)
    diff_bits = 0
    for a, b in combinations(elems, 2):
        d = abs(a - b)
        diff_bits |= (1 << d)
    return sum_bits, diff_bits


def _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """Check if elem can be added to the current set without violating constraints."""
    # Check sum collisions: elem+elem and elem+x for all x in elems
    if (sum_bits >> (elem + elem)) & 1:
        return False
    for x in elems:
        s = elem + x
        if (sum_bits >> s) & 1:
            return False
    # Check difference collisions: |elem - x| for all x in elems
    for x in elems:
        d = abs(elem - x)
        if d == 0:
            continue
        if (diff_bits >> d) & 1:
            return False
        if (other_diff_bits >> d) & 1:
            return False
    return True


def _greedy_build(order_a, order_b, deadline):
    """Greedy build using given element orders for A and B."""
    A = []
    B = []
    sumA = 0
    diffA = 0
    sumB = 0
    diffB = 0

    # Interleave: try to add to A or B in the given order
    ia = 0
    ib = 0
    len_a = len(order_a)
    len_b = len(order_b)

    while ia < len_a or ib < len_b:
        if time.time() > deadline:
            break

        # Try to add next element to A
        if ia < len_a and len(A) < MAX_EACH:
            elem = order_a[ia]
            if _can_add(elem, A, sumA, diffA, diffB):
                # Update bits
                sumA |= (1 << (elem + elem))
                for x in A:
                    sumA |= (1 << (elem + x))
                    d = abs(elem - x)
                    diffA |= (1 << d)
                A.append(elem)
                ia += 1
                continue

        # Try to add next element to B
        if ib < len_b and len(B) < MAX_EACH:
            elem = order_b[ib]
            if _can_add(elem, B, sumB, diffB, diffA):
                # Update bits
                sumB |= (1 << (elem + elem))
                for x in B:
                    sumB |= (1 << (elem + x))
                    d = abs(elem - x)
                    diffB |= (1 << d)
                B.append(elem)
                ib += 1
                continue

        # Neither can be added; advance the pointer that failed
        if ia < len_a:
            ia += 1
        if ib < len_b:
            ib += 1

    return A, B


def _local_search(A, B, sumA, diffA, sumB, diffB, deadline):
    """Try to improve by removing and re-adding elements."""
    best_A = list(A)
    best_B = list(B)
    best_size = len(A) + len(B)
    best_sumA = sumA
    best_diffA = diffA
    best_sumB = sumB
    best_diffB = diffB

    cur_A = list(A)
    cur_B = list(B)
    cur_sumA = sumA
    cur_diffA = diffA
    cur_sumB = sumB
    cur_diffB = diffB
    cur_size = len(A) + len(B)

    # Try swapping: remove an element from one set, try to add a new one
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)

    for _ in range(200):
        if time.time() > deadline:
            break

        # Try removing an element from A and adding a different one
        if len(cur_A) > 0:
            idx = random.randint(0, len(cur_A) - 1)
            removed = cur_A[idx]
            new_A = cur_A[:idx] + cur_A[idx+1:]
            # Rebuild bits for new_A
            nsA, ndA = _build_sum_diff_bits(new_A)

            # Try to add a candidate to new_A
            for cand in candidates:
                if cand == removed or cand in new_A:
                    continue
                if len(new_A) >= MAX_EACH:
                    break
                if _can_add(cand, new_A, nsA, ndA, cur_diffB):
                    # Add it
                    nsA |= (1 << (cand + cand))
                    for x in new_A:
                        nsA |= (1 << (cand + x))
                        d = abs(cand - x)
                        ndA |= (1 << d)
                    new_A.append(cand)
                    new_size = len(new_A) + len(cur_B)
                    if new_size > cur_size:
                        cur_A = new_A
                        cur_sumA = nsA
                        cur_diffA = ndA
                        cur_size = new_size
                        if cur_size > best_size:
                            best_A = list(cur_A)
                            best_B = list(cur_B)
                            best_size = cur_size
                            best_sumA = cur_sumA
                            best_diffA = cur_diffA
                            best_sumB = cur_sumB
                            best_diffB = cur_diffB
                    break

        # Try removing an element from B and adding a different one
        if len(cur_B) > 0:
            idx = random.randint(0, len(cur_B) - 1)
            removed = cur_B[idx]
            new_B = cur_B[:idx] + cur_B[idx+1:]
            nsB, ndB = _build_sum_diff_bits(new_B)

            for cand in candidates:
                if cand == removed or cand in new_B:
                    continue
                if len(new_B) >= MAX_EACH:
                    break
                if _can_add(cand, new_B, nsB, ndB, cur_diffA):
                    nsB |= (1 << (cand + cand))
                    for x in new_B:
                        nsB |= (1 << (cand + x))
                        d = abs(cand - x)
                        ndB |= (1 << d)
                    new_B.append(cand)
                    new_size = len(cur_A) + len(new_B)
                    if new_size > cur_size:
                        cur_B = new_B
                        cur_sumB = nsB
                        cur_diffB = ndB
                        cur_size = new_size
                        if cur_size > best_size:
                            best_A = list(cur_A)
                            best_B = list(cur_B)
                            best_size = cur_size
                            best_sumA = cur_sumA
                            best_diffA = cur_diffA
                            best_sumB = cur_sumB
                            best_diffB = cur_diffB
                    break

    return best_A, best_B, best_sumA, best_diffA, best_sumB, best_diffB


def _validate(A, B):
    """Validate that A and B are valid Sidon sets with disjoint nonzero differences."""
    if not A or not B:
        return False
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return False
    for s in (A, B):
        if len(s) != len(set(s)):
            return False
        sums = [a + b for a, b in combinations_with_replacement(s, 2)]
        if len(sums) != len(set(sums)):
            return False
    diffA = {abs(a - b) for a, b in combinations(A, 2)}
    diffB = {abs(a - b) for a, b in combinations(B, 2)}
    if diffA & diffB:
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    best_A = []
    best_B = []
    best_size = 0

    # Multiple random restarts with different element orderings
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break

        # Create random orderings
        order_a = list(range(1, N + 1))
        order_b = list(range(1, N + 1))
        random.shuffle(order_a)
        random.shuffle(order_b)

        A, B = _greedy_build(order_a, order_b, deadline)
        size = len(A) + len(B)

        if size > best_size:
            best_A = A
            best_B = B
            best_size = size

            # Do local search on the best solution
            if time.time() < deadline - 3.0:
                sumA, diffA = _build_sum_diff_bits(A)
                sumB, diffB = _build_sum_diff_bits(B)
                A_ls, B_ls, _, _, _, _ = _local_search(A, B, sumA, diffA, sumB, diffB, deadline)
                ls_size = len(A_ls) + len(B_ls)
                if ls_size > best_size:
                    best_A = A_ls
                    best_B = B_ls
                    best_size = ls_size

    # Validate final result
    if not _validate(best_A, best_B):
        # Fallback: simple greedy
        best_A, best_B = _greedy_build(list(range(1, N + 1)), list(range(1, N + 1)), deadline)

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
