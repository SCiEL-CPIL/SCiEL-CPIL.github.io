"""Two Sidon sets with disjoint difference sets – ILS with Bitsets and time guard."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin before 60s kill

def _rebuild_bits(elems):
    """Rebuild sum_bits and diff_bits for a list of elements."""
    sum_bits = 0
    for a, b in combinations_with_replacement(elems, 2):
        sum_bits |= (1 << (a + b))
    diff_bits = 0
    for a, b in combinations(elems, 2):
        diff_bits |= (1 << abs(a - b))
    return sum_bits, diff_bits

def _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """Check if elem can be added to the set defined by elems, sum_bits, diff_bits."""
    # Check sum collisions
    if (sum_bits >> (elem + elem)) & 1:
        return False
    for a in elems:
        if (sum_bits >> (elem + a)) & 1:
            return False
    # Check difference collisions within this set
    for a in elems:
        d = abs(elem - a)
        if (diff_bits >> d) & 1:
            return False
    # Check difference collisions with other set
    if (other_diff_bits >> abs(elem - elem)) & 1:
        pass  # 0 difference, skip
    for a in elems:
        d = abs(elem - a)
        if (other_diff_bits >> d) & 1:
            return False
    return True

def _add_elem(elem, elems, sum_bits, diff_bits):
    """Add elem and update bits. Returns new sum_bits, diff_bits."""
    new_sum_bits = sum_bits
    new_diff_bits = diff_bits
    for a in elems:
        new_sum_bits |= (1 << (elem + a))
        new_diff_bits |= (1 << abs(elem - a))
    new_sum_bits |= (1 << (elem + elem))
    elems.append(elem)
    return new_sum_bits, new_diff_bits

def _remove_elem(elem, elems, sum_bits, diff_bits):
    """Remove elem and rebuild bits."""
    elems.remove(elem)
    return _rebuild_bits(elems)

def _greedy_build(start_time):
    """Greedy construction of two Sidon sets with disjoint differences."""
    sets = [[], []]
    sum_bits = [0, 0]
    diff_bits = [0, 0]
    
    # Try elements in order, assign to set 0 or 1
    for x in range(1, N + 1):
        # Try adding to set 0
        if _can_add(x, sets[0], sum_bits[0], diff_bits[0], diff_bits[1]):
            sum_bits[0], diff_bits[0] = _add_elem(x, sets[0], sum_bits[0], diff_bits[0])
            continue
        # Try adding to set 1
        if _can_add(x, sets[1], sum_bits[1], diff_bits[1], diff_bits[0]):
            sum_bits[1], diff_bits[1] = _add_elem(x, sets[1], sum_bits[1], diff_bits[1])
            continue
    return sets, sum_bits, diff_bits

def _ilimprove(sets, sum_bits, diff_bits, start_time, max_iters=2000):
    """Iterated local search with perturbation."""
    best_sets = [list(sets[0]), list(sets[1])]
    best_score = len(sets[0]) + len(sets[1])
    
    cur_sets = [list(sets[0]), list(sets[1])]
    cur_sum_bits = [sum_bits[0], sum_bits[1]]
    cur_diff_bits = [diff_bits[0], diff_bits[1]]
    
    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Try to add a random element to a random set
        side = random.randint(0, 1)
        # Find a candidate that can be added
        candidates = [x for x in range(1, N + 1) if x not in cur_sets[side]]
        if not candidates:
            continue
        random.shuffle(candidates)
        
        added = False
        for x in candidates[:20]:  # Limit search
            if _can_add(x, cur_sets[side], cur_sum_bits[side], cur_diff_bits[side], cur_diff_bits[1 - side]):
                cur_sum_bits[side], cur_diff_bits[side] = _add_elem(x, cur_sets[side], cur_sum_bits[side], cur_diff_bits[side])
                added = True
                break
        
        if added:
            new_score = len(cur_sets[0]) + len(cur_sets[1])
            if new_score > best_score:
                best_score = new_score
                best_sets = [list(cur_sets[0]), list(cur_sets[1])]
        else:
            # Perturbation: remove a random element from a random set
            if cur_sets[side]:
                idx = random.randint(0, len(cur_sets[side]) - 1)
                elem = cur_sets[side][idx]
                cur_sum_bits[side], cur_diff_bits[side] = _remove_elem(elem, cur_sets[side], cur_sum_bits[side], cur_diff_bits[side])
    
    return best_sets, best_score

def construct():
    start_time = time.time()
    
    # Initial greedy build
    sets, sum_bits, diff_bits = _greedy_build(start_time)
    best_sets = [list(sets[0]), list(sets[1])]
    best_score = len(sets[0]) + len(sets[1])
    
    # Run ILS if time permits
    if time.time() - start_time < DEADLINE - 10:
        best_sets, best_score = _ilimprove(sets, sum_bits, diff_bits, start_time, max_iters=5000)
    
    # Verify and return
    if not best_sets[0]:
        best_sets[0] = [1]
    if not best_sets[1]:
        best_sets[1] = [2]
    
    return best_sets
# EVOLVE-BLOCK-END
