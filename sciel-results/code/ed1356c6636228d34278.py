import time
import random

N = 256
DEADLINE = 55.0  # Safety margin before 60s limit


def _count_3ap(selected_set):
    """Count 3-APs in a set using bitmask-style check."""
    count = 0
    sel = selected_set
    for a in sorted(sel):
        for b in sel:
            if b > a:
                c = 2 * b - a
                if c in sel and c > b:
                    count += 1
    return count


def _has_4ap(selected_set, new_elem):
    """Check if adding new_elem to selected_set creates a 4-AP."""
    sel = selected_set
    d_max = (N - 1) // 3
    
    # new_elem as first: a, a+d, a+2d, a+3d
    for d in range(1, d_max + 1):
        if new_elem + d in sel and new_elem + 2*d in sel and new_elem + 3*d in sel:
            return True
    # new_elem as second: a, a+d, a+2d, a+3d where a+d=new_elem
    for d in range(1, d_max + 1):
        a = new_elem - d
        if a >= 1 and a in sel and new_elem + d in sel and new_elem + 2*d in sel:
            return True
    # new_elem as third: a, a+d, a+2d, a+3d where a+2d=new_elem
    for d in range(1, d_max + 1):
        a = new_elem - 2*d
        if a >= 1 and a in sel and new_elem - d in sel and new_elem + d in sel:
            return True
    # new_elem as fourth: a, a+d, a+2d, a+3d where a+3d=new_elem
    for d in range(1, d_max + 1):
        a = new_elem - 3*d
        if a >= 1 and a in sel and a + d in sel and a + 2*d in sel:
            return True
    return False


def _greedy_with_order(order, start_time):
    """Greedy construction following a given order."""
    selected = set()
    for x in order:
        if time.time() - start_time > DEADLINE - 5:
            break
        if not _has_4ap(selected, x):
            selected.add(x)
    return selected


def _local_search(selected, start_time, max_iters=5000):
    """Try to improve by swapping elements out and in."""
    best_score = _count_3ap(selected)
    current = set(selected)
    current_score = best_score
    
    for _ in range(max_iters):
        if time.time() - start_time > DEADLINE - 3:
            break
        # Try removing one element
        for x in list(current):
            if time.time() - start_time > DEADLINE - 3:
                break
            new_set = current - {x}
            new_score = _count_3ap(new_set)
            if new_score > current_score:
                current = new_set
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                break
        else:
            # Try adding one element
            added = False
            for x in range(1, N + 1):
                if time.time() - start_time > DEADLINE - 3:
                    break
                if x not in current and not _has_4ap(current, x):
                    new_set = current | {x}
                    new_score = _count_3ap(new_set)
                    if new_score > current_score:
                        current = new_set
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                        added = True
                        break
            if not added:
                # Try swap: remove one, add another
                swapped = False
                for x in list(current):
                    if time.time() - start_time > DEADLINE - 3:
                        break
                    for y in range(1, N + 1):
                        if time.time() - start_time > DEADLINE - 3:
                            break
                        if y == x or y in current:
                            continue
                        if _has_4ap(current - {x}, y):
                            continue
                        new_set = (current - {x}) | {y}
                        new_score = _count_3ap(new_set)
                        if new_score > current_score:
                            current = new_set
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                            swapped = True
                            break
                    if swapped:
                        break
                if not swapped:
                    break
    
    return current, best_score


def construct():
    start_time = time.time()
    
    # Strategy 1: Greedy with natural order
    best_set = _greedy_with_order(range(1, N + 1), start_time)
    best_score = _count_3ap(best_set)
    
    # Strategy 2: Greedy with random orders
    random.seed(42)
    for trial in range(20):
        if time.time() - start_time > DEADLINE - 10:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        candidate = _greedy_with_order(order, start_time)
        score = _count_3ap(candidate)
        if score > best_score:
            best_score = score
            best_set = candidate
    
    # Strategy 3: Local search from best
    if time.time() - start_time < DEADLINE - 5:
        best_set, best_score = _local_search(best_set, start_time, max_iters=3000)
    
    # Final local search pass
    if time.time() - start_time < DEADLINE - 2:
        best_set, best_score = _local_search(best_set, start_time, max_iters=2000)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
