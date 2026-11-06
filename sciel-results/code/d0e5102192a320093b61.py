"""Two Sidon sets with disjoint difference sets – ILS with Bitsets."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 50.0  # Safety margin (90s limit, leave 40s buffer)

def _rebuild_bits(elems):
    """Rebuild sum_bits and diff_bits for a list of elements."""
    sum_bits = 0
    for i in range(len(elems)):
        for j in range(i, len(elems)):
            a, b = elems[i], elems[j]
            sum_bits |= (1 << (a + b))
    diff_bits = 0
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            a, b = elems[i], elems[j]
            diff_bits |= (1 << abs(a - b))
    return sum_bits, diff_bits

def _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """Check if elem can be added to the set defined by elems, sum_bits, diff_bits."""
    # Check sum collisions: elem+elem and elem+existing
    if (sum_bits >> (elem + elem)) & 1:
        return False
    for a in elems:
        if (sum_bits >> (elem + a)) & 1:
            return False
    # Check difference collisions: elem - existing and existing - elem
    for a in elems:
        d = abs(elem - a)
        if (diff_bits >> d) & 1:
            return False
        if (other_diff_bits >> d) & 1:
            return False
    return True

def _add_elem(elem, elems, sum_bits, diff_bits):
    """Add elem to elems, update sum_bits and diff_bits."""
    new_sum = sum_bits | (1 << (elem + elem))
    new_diff = diff_bits
    for a in elems:
        new_sum |= (1 << (elem + a))
        new_diff |= (1 << abs(elem - a))
    elems.append(elem)
    return new_sum, new_diff

def _remove_elem(elem, elems, sum_bits, diff_bits):
    """Remove elem from elems, rebuild bits."""
    elems.remove(elem)
    return _rebuild_bits(elems)

def _greedy_build(side, other_diff_bits, rng, start=1, end=N+1):
    """Greedy build for one side, trying to add elements."""
    elems = []
    sum_bits = 0
    diff_bits = 0
    for x in range(start, end):
        if len(elems) >= MAX_EACH:
            break
        if _can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
            sum_bits, diff_bits = _add_elem(x, elems, sum_bits, diff_bits)
    return elems, sum_bits, diff_bits

def _greedy_build_random(side, other_diff_bits, rng, start=1, end=N+1):
    """Greedy build with random ordering."""
    elems = []
    sum_bits = 0
    diff_bits = 0
    candidates = list(range(start, end))
    rng.shuffle(candidates)
    for x in candidates:
        if len(elems) >= MAX_EACH:
            break
        if _can_add(x, elems, sum_bits, diff_bits, other_diff_bits):
            sum_bits, diff_bits = _add_elem(x, elems, sum_bits, diff_bits)
    return elems, sum_bits, diff_bits

def _ilimprove(A, B, A_sum, A_diff, B_sum, B_diff, rng, time_limit):
    """Iterated local search: try swaps and additions to improve total size."""
    best_total = len(A) + len(B)
    best_A, best_B = list(A), list(B)
    best_A_sum, best_A_diff = A_sum, A_diff
    best_B_sum, best_B_diff = B_sum, B_diff
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        # Try to add a random element to A or B
        improved = False
        side = rng.randint(0, 1)
        if side == 0:
            if len(A) < MAX_EACH:
                # Try adding a random element
                for attempt in range(50):
                    x = rng.randint(1, N)
                    if x not in A:
                        if _can_add(x, A, A_sum, A_diff, B_diff):
                            A_sum, A_diff = _add_elem(x, A, A_sum, A_diff)
                            improved = True
                            break
                if not improved:
                    # Try removing an element from A and adding two
                    if len(A) > 1:
                        remove_idx = rng.randint(0, len(A)-1)
                        old_elem = A[remove_idx]
                        old_sum, old_diff = A_sum, A_diff
                        A_sum, A_diff = _remove_elem(old_elem, A, A_sum, A_diff)
                        added = 0
                        for attempt in range(100):
                            x = rng.randint(1, N)
                            if x not in A and x != old_elem:
                                if _can_add(x, A, A_sum, A_diff, B_diff):
                                    A_sum, A_diff = _add_elem(x, A, A_sum, A_diff)
                                    added += 1
                                    if added >= 2:
                                        break
                        if added < 2:
                            # Revert
                            A_sum, A_diff = _remove_elem(old_elem, A, A_sum, A_diff)
                            A_sum, A_diff = old_sum, old_diff
                            A.append(old_elem)
                            A.sort()
                            # Rebuild properly
                            A_sum, A_diff = _rebuild_bits(A)
                        else:
                            improved = True
            if len(A) + len(B) > best_total:
                best_total = len(A) + len(B)
                best_A, best_B = list(A), list(B)
                best_A_sum, best_A_diff = A_sum, A_diff
                best_B_sum, best_B_diff = B_sum, B_diff
        else:
            if len(B) < MAX_EACH:
                for attempt in range(50):
                    x = rng.randint(1, N)
                    if x not in B:
                        if _can_add(x, B, B_sum, B_diff, A_diff):
                            B_sum, B_diff = _add_elem(x, B, B_sum, B_diff)
                            improved = True
                            break
                if not improved:
                    if len(B) > 1:
                        remove_idx = rng.randint(0, len(B)-1)
                        old_elem = B[remove_idx]
                        old_sum, old_diff = B_sum, B_diff
                        B_sum, B_diff = _remove_elem(old_elem, B, B_sum, B_diff)
                        added = 0
                        for attempt in range(100):
                            x = rng.randint(1, N)
                            if x not in B and x != old_elem:
                                if _can_add(x, B, B_sum, B_diff, A_diff):
                                    B_sum, B_diff = _add_elem(x, B, B_sum, B_diff)
                                    added += 1
                                    if added >= 2:
                                        break
                        if added < 2:
                            B_sum, B_diff = _remove_elem(old_elem, B, B_sum, B_diff)
                            B_sum, B_diff = old_sum, old_diff
                            B.append(old_elem)
                            B.sort()
                            B_sum, B_diff = _rebuild_bits(B)
                        else:
                            improved = True
            if len(A) + len(B) > best_total:
                best_total = len(A) + len(B)
                best_A, best_B = list(A), list(B)
                best_A_sum, best_A_diff = A_sum, A_diff
                best_B_sum, best_B_diff = B_sum, B_diff
    
    return best_A, best_B, best_A_sum, best_A_diff, best_B_sum, best_B_diff

def construct():
    rng = random.Random(42)
    start_time = time.time()
    
    best_total = 0
    best_A, best_B = [], []
    best_A_sum, best_A_diff = 0, 0
    best_B_sum, best_B_diff = 0, 0
    
    # Initial greedy builds with different strategies
    for trial in range(20):
        if time.time() - start_time > DEADLINE:
            break
        
        # Try different initial orderings
        if trial % 3 == 0:
            # Standard ascending
            A, A_sum, A_diff = _greedy_build(0, 0, rng)
            B, B_sum, B_diff = _greedy_build(1, A_diff, rng)
        elif trial % 3 == 1:
            # Random order for A, then greedy for B
            A, A_sum, A_diff = _greedy_build_random(0, 0, rng)
            B, B_sum, B_diff = _greedy_build(1, A_diff, rng)
        else:
            # Greedy for A, random for B
            A, A_sum, A_diff = _greedy_build(0, 0, rng)
            B, B_sum, B_diff = _greedy_build_random(1, A_diff, rng)
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A, best_B = list(A), list(B)
            best_A_sum, best_A_diff = A_sum, A_diff
            best_B_sum, best_B_diff = B_sum, B_diff
        
        # Local improvement
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 5:
            A, B, A_sum, A_diff, B_sum, B_diff = _ilimprove(
                A, B, A_sum, A_diff, B_sum, B_diff, rng, min(remaining - 2, 10)
            )
            total = len(A) + len(B)
            if total > best_total:
                best_total = total
                best_A, best_B = list(A), list(B)
                best_A_sum, best_A_diff = A_sum, A_diff
                best_B_sum, best_B_diff = B_sum, B_diff
    
    # Final polish: try to add more elements
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 2:
        best_A, best_B, best_A_sum, best_A_diff, best_B_sum, best_B_diff = _ilimprove(
            best_A, best_B, best_A_sum, best_A_diff, best_B_sum, best_B_diff, rng, remaining - 1
        )
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
