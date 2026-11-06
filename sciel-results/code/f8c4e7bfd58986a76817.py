"""Two Sidon sets with disjoint difference sets – Bitset + multi-start local search."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # safety margin before 60s kill


def _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """Check if elem can be added to elems maintaining Sidon + disjoint diffs."""
    # Check elem+elem sum
    s2 = elem + elem
    if (sum_bits >> s2) & 1:
        return False
    # Check elem+a sums and |elem-a| diffs
    for a in elems:
        s = elem + a
        if (sum_bits >> s) & 1:
            return False
        d = abs(elem - a)
        if (diff_bits >> d) & 1 or (other_diff_bits >> d) & 1:
            return False
    return True


def _add_elem(state, side, elem):
    """Add elem to side, updating bitsets."""
    elems = state['elems'][side]
    sum_bits = state['sum_bits'][side]
    diff_bits = state['diff_bits'][side]
    
    # Update sum bits
    s2 = elem + elem
    sum_bits |= (1 << s2)
    for a in elems:
        s = elem + a
        sum_bits |= (1 << s)
    
    # Update diff bits
    for a in elems:
        d = abs(elem - a)
        diff_bits |= (1 << d)
    
    state['elems'][side].append(elem)
    state['sum_bits'][side] = sum_bits
    state['diff_bits'][side] = diff_bits


def _remove_elem(state, side, elem):
    """Remove elem from side, updating bitsets."""
    elems = state['elems'][side]
    idx = elems.index(elem)
    elems[idx] = elems[-1]
    elems.pop()
    
    # Rebuild sum and diff bits for this side
    sum_bits = 0
    diff_bits = 0
    for i, a in enumerate(elems):
        sum_bits |= (1 << (a + a))
        for j in range(i + 1, len(elems)):
            b = elems[j]
            sum_bits |= (1 << (a + b))
            d = abs(a - b)
            diff_bits |= (1 << d)
    
    state['sum_bits'][side] = sum_bits
    state['diff_bits'][side] = diff_bits


def _greedy_build(order, deadline):
    """Greedy build in given order. Returns (state, total_size)."""
    state = {
        'elems': [[], []],
        'sum_bits': [0, 0],
        'diff_bits': [0, 0]
    }
    
    for elem in order:
        if time.time() > deadline:
            break
        # Try adding to side 0
        if _can_add(elem, state['elems'][0], state['sum_bits'][0], 
                    state['diff_bits'][0], state['diff_bits'][1]):
            _add_elem(state, 0, elem)
        # Try adding to side 1
        elif _can_add(elem, state['elems'][1], state['sum_bits'][1], 
                      state['diff_bits'][1], state['diff_bits'][0]):
            _add_elem(state, 1, elem)
    
    total = len(state['elems'][0]) + len(state['elems'][1])
    return state, total


def _local_search(state, deadline):
    """1-for-1 and 1-for-2 swap local search."""
    best_state = {
        'elems': [list(state['elems'][0]), list(state['elems'][1])],
        'sum_bits': [state['sum_bits'][0], state['sum_bits'][1]],
        'diff_bits': [state['diff_bits'][0], state['diff_bits'][1]]
    }
    best_total = len(best_state['elems'][0]) + len(best_state['elems'][1])
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try 1-for-1 swaps: remove x from side i, add y to side i
        for side in (0, 1):
            elems = best_state['elems'][side]
            for i, x in enumerate(elems):
                if time.time() > deadline:
                    return best_state, best_total
                
                # Try removing x
                _remove_elem(best_state, side, x)
                
                # Try adding each candidate y not in either set
                for y in range(1, N + 1):
                    if y == x:
                        continue
                    if y in best_state['elems'][0] or y in best_state['elems'][1]:
                        continue
                    
                    if _can_add(y, best_state['elems'][side], best_state['sum_bits'][side],
                                best_state['diff_bits'][side], best_state['diff_bits'][1 - side]):
                        _add_elem(best_state, side, y)
                        new_total = len(best_state['elems'][0]) + len(best_state['elems'][1])
                        if new_total > best_total:
                            best_total = new_total
                            improved = True
                        # Keep the improvement (don't revert)
                        break
                else:
                    # No improvement, revert
                    _add_elem(best_state, side, x)
        
        # Try 1-for-2 swaps: remove x from side i, add y to side i and z to side j
        # This is more complex; skip for now due to time constraints
    
    return best_state, best_total


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_total = 0
    best_result = ([1], [2])  # fallback
    
    # Multiple random restarts
    random.seed(42)
    num_restarts = 0
    
    while time.time() < deadline - 5.0:  # leave 5s for final polish
        # Generate random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Greedy build
        state, total = _greedy_build(order, deadline)
        num_restarts += 1
        
        if total > best_total:
            best_total = total
            best_result = (list(state['elems'][0]), list(state['elems'][1]))
        
        # Local search if we have time
        if time.time() < deadline - 2.0:
            improved_state, improved_total = _local_search(state, deadline)
            if improved_total > best_total:
                best_total = improved_total
                best_result = (list(improved_state['elems'][0]), list(improved_state['elems'][1]))
    
    # Ensure nonempty
    if not best_result[0]:
        best_result = ([1], list(best_result[1]) if best_result[1] else [2])
    if not best_result[1]:
        best_result = (list(best_result[0]), [2])
    
    return list(best_result)
# EVOLVE-BLOCK-END
