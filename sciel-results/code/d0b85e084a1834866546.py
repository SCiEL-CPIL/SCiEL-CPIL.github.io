"""Two Sidon sets with disjoint difference sets – randomized greedy with bitsets + local search."""
# EVOLVE-BLOCK-START
import time
import random
import sys

N = 256
MAX_EACH = 64
DEADLINE = 52.0  # Safety margin (90s limit, leave room for overhead)


def _is_valid_sidon(elems):
    """Check if elems is a Sidon set (all pair sums unique)."""
    if not elems:
        return True
    sums = set()
    for i in range(len(elems)):
        for j in range(i, len(elems)):
            s = elems[i] + elems[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _diff_set(elems):
    """Return set of absolute differences between distinct pairs."""
    diffs = set()
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            diffs.add(abs(elems[i] - elems[j]))
    return diffs


def _sum_bitset(elems):
    """Return bitmask where bit s is set if s is a pair sum (with replacement)."""
    bits = 0
    for i in range(len(elems)):
        for j in range(i, len(elems)):
            bits |= (1 << (elems[i] + elems[j]))
    return bits


def _diff_bitset(elems):
    """Return bitmask where bit d is set if d is an absolute difference."""
    bits = 0
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            bits |= (1 << abs(elems[i] - elems[j]))
    return bits


def _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """Check if elem can be added to the set with given bitsets."""
    # Check sum collisions: elem+elem and elem+a for all a in elems
    if (sum_bits >> (elem + elem)) & 1:
        return False
    for a in elems:
        if (sum_bits >> (elem + a)) & 1:
            return False
    # Check diff collisions with other set
    for a in elems:
        d = abs(elem - a)
        if d > 0 and (other_diff_bits >> d) & 1:
            return False
    # Check diff collisions within own set (new differences must not already exist)
    for a in elems:
        d = abs(elem - a)
        if d > 0 and (diff_bits >> d) & 1:
            return False
    return True


def _greedy_build(order, time_limit):
    """Greedy build following a given element order. Returns (A, B, score)."""
    start = time.time()
    A = []
    B = []
    sum_bits_A = 0
    sum_bits_B = 0
    diff_bits_A = 0
    diff_bits_B = 0

    for x in order:
        if time.time() - start > time_limit:
            break
        # Try adding to A
        if len(A) < MAX_EACH and _can_add(x, A, sum_bits_A, diff_bits_A, diff_bits_B):
            # Update bitsets
            for a in A:
                sum_bits_A |= (1 << (x + a))
                diff_bits_A |= (1 << abs(x - a))
            sum_bits_A |= (1 << (x + x))
            A.append(x)
        elif len(B) < MAX_EACH and _can_add(x, B, sum_bits_B, diff_bits_B, diff_bits_A):
            for b in B:
                sum_bits_B |= (1 << (x + b))
                diff_bits_B |= (1 << abs(x - b))
            sum_bits_B |= (1 << (x + x))
            B.append(x)

    if not A or not B:
        return None, None, 0
    score = len(A) + len(B)
    return A, B, score


def _local_search(A, B, time_limit):
    """Try to improve via swaps and insertions. Returns (A, B, score)."""
    start = time.time()
    A = list(A)
    B = list(B)
    score = len(A) + len(B)
    improved = True

    while improved and time.time() - start < time_limit:
        improved = False
        sum_bits_A = _sum_bitset(A)
        sum_bits_B = _sum_bitset(B)
        diff_bits_A = _diff_bitset(A)
        diff_bits_B = _diff_bitset(B)

        # Try adding random elements
        for _ in range(200):
            if time.time() - start > time_limit:
                break
            x = random.randint(1, N)
            if x not in A and x not in B:
                if len(A) < MAX_EACH and _can_add(x, A, sum_bits_A, diff_bits_A, diff_bits_B):
                    for a in A:
                        sum_bits_A |= (1 << (x + a))
                        diff_bits_A |= (1 << abs(x - a))
                    sum_bits_A |= (1 << (x + x))
                    A.append(x)
                    score += 1
                    improved = True
                    break
                elif len(B) < MAX_EACH and _can_add(x, B, sum_bits_B, diff_bits_B, diff_bits_A):
                    for b in B:
                        sum_bits_B |= (1 << (x + b))
                        diff_bits_B |= (1 << abs(x - b))
                    sum_bits_B |= (1 << (x + x))
                    B.append(x)
                    score += 1
                    improved = True
                    break
            elif x in A and x not in B:
                if len(B) < MAX_EACH and _can_add(x, B, sum_bits_B, diff_bits_B, diff_bits_A):
                    # Move from A to B
                    A.remove(x)
                    sum_bits_A = _sum_bitset(A)
                    diff_bits_A = _diff_bitset(A)
                    for b in B:
                        sum_bits_B |= (1 << (x + b))
                        diff_bits_B |= (1 << abs(x - b))
                    sum_bits_B |= (1 << (x + x))
                    B.append(x)
                    # score unchanged
                    improved = True
                    break
            elif x in B and x not in A:
                if len(A) < MAX_EACH and _can_add(x, A, sum_bits_A, diff_bits_A, diff_bits_B):
                    B.remove(x)
                    sum_bits_B = _sum_bitset(B)
                    diff_bits_B = _diff_bitset(B)
                    for a in A:
                        sum_bits_A |= (1 << (x + a))
                        diff_bits_A |= (1 << abs(x - a))
                    sum_bits_A |= (1 << (x + x))
                    A.append(x)
                    improved = True
                    break

        # Try swap: remove from A, add to B or vice versa
        if len(A) > 1 and len(B) < MAX_EACH:
            for i in range(len(A)):
                if time.time() - start > time_limit:
                    break
                x = A[i]
                A2 = A[:i] + A[i+1:]
                sum_bits_A2 = _sum_bitset(A2)
                diff_bits_A2 = _diff_bitset(A2)
                if _can_add(x, B, sum_bits_B, diff_bits_B, diff_bits_A2):
                    # Check A2 is still valid (it is, subset)
                    A = A2
                    sum_bits_A = sum_bits_A2
                    diff_bits_A = diff_bits_A2
                    for b in B:
                        sum_bits_B |= (1 << (x + b))
                        diff_bits_B |= (1 << abs(x - b))
                    sum_bits_B |= (1 << (x + x))
                    B.append(x)
                    improved = True
                    break

        if len(B) > 1 and len(A) < MAX_EACH:
            for i in range(len(B)):
                if time.time() - start > time_limit:
                    break
                x = B[i]
                B2 = B[:i] + B[i+1:]
                sum_bits_B2 = _sum_bitset(B2)
                diff_bits_B2 = _diff_bitset(B2)
                if _can_add(x, A, sum_bits_A, diff_bits_A, diff_bits_B2):
                    B = B2
                    sum_bits_B = sum_bits_B2
                    diff_bits_B = diff_bits_B2
                    for a in A:
                        sum_bits_A |= (1 << (x + a))
                        diff_bits_A |= (1 << abs(x - a))
                    sum_bits_A |= (1 << (x + x))
                    A.append(x)
                    improved = True
                    break

    return A, B, len(A) + len(B)


def construct():
    start = time.time()
    best_A, best_B, best_score = None, None, 0

    # Strategy 1: Multiple randomized greedy restarts
    for trial in range(2000):
        if time.time() - start > DEADLINE - 10:
            break

        # Create random order with some bias
        elems = list(range(1, N + 1))
        random.shuffle(elems)

        # Try different biases
        bias = random.random()
        if bias < 0.3:
            # Favor small numbers
            elems.sort(key=lambda x: x + random.random() * 50)
        elif bias < 0.6:
            # Favor large numbers
            elems.sort(key=lambda x: -x + random.random() * 50)
        elif bias < 0.8:
            # Random
            pass
        else:
            # Spread out
            elems.sort(key=lambda x: x % 16 + random.random() * 16)

        A, B, score = _greedy_build(elems, 0.005)
        if score > best_score:
            best_score = score
            best_A, best_B = A[:], B[:]

        # If we have a good result, do local search
        if score >= best_score - 2 and score > 0 and time.time() - start < DEADLINE - 5:
            A2, B2, score2 = _local_search(A, B, min(0.5, DEADLINE - (time.time() - start) - 2))
            if score2 > best_score:
                best_score = score2
                best_A, best_B = A2[:], B2[:]

    # Strategy 2: Try structured constructions
    if time.time() - start < DEADLINE - 5:
        # Try arithmetic progressions with gaps
        for gap_a in [2, 3, 4, 5]:
            for gap_b in [2, 3, 4, 5]:
                if gap_a == gap_b:
                    continue
                A = list(range(1, N + 1, gap_a))[:MAX_EACH]
                B = list(range(1, N + 1, gap_b))[:MAX_EACH]
                # Filter to maintain Sidon property
                A_filt = []
                sum_bits_A_filt = 0
                diff_bits_A_filt = 0
                B_filt = []
                sum_bits_B_filt = 0
                diff_bits_B_filt = 0
                
                # Build A_filt first
                for x in A:
                    if len(A_filt) < MAX_EACH and _can_add(x, A_filt, sum_bits_A_filt, diff_bits_A_filt, diff_bits_B_filt):
                        for a in A_filt:
                            sum_bits_A_filt |= (1 << (x + a))
                            diff_bits_A_filt |= (1 << abs(x - a))
                        sum_bits_A_filt |= (1 << (x + x))
                        A_filt.append(x)
                
                # Build B_filt next
                for x in B:
                    if len(B_filt) < MAX_EACH and _can_add(x, B_filt, sum_bits_B_filt, diff_bits_B_filt, diff_bits_A_filt):
                        for b in B_filt:
                            sum_bits_B_filt |= (1 << (x + b))
                            diff_bits_B_filt |= (1 << abs(x - b))
                        sum_bits_B_filt |= (1 << (x + x))
                        B_filt.append(x)
                
                if A_filt and B_filt:
                    s = len(A_filt) + len(B_filt)
                    if s > best_score:
                        best_score = s
                        best_A, best_B = A_filt[:], B_filt[:]

    # Final local search on best
    if best_A and best_B and time.time() - start < DEADLINE - 1:
        A2, B2, score2 = _local_search(best_A, best_B, min(3.0, DEADLINE - (time.time() - start)))
        if score2 > best_score:
            best_score = score2
            best_A, best_B = A2[:], B2[:]

    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]

    return [best_A, best_B]
# EVOLVE-BLOCK-END
