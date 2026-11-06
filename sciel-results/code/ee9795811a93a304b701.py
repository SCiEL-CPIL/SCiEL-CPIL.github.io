"""Two Sidon sets with disjoint difference sets – ILS with Bitsets."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations, combinations_with_replacement

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # Safety margin

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
    # elem + elem
    if (sum_bits >> (elem + elem)) & 1:
        return False
    # elem + a
    for a in elems:
        if (sum_bits >> (elem + a)) & 1:
            return False
        # Check diff collisions
        d = abs(elem - a)
        if (diff_bits >> d) & 1:
            return False
        if (other_diff_bits >> d) & 1:
            return False
    return True

def _add_elem(state, side, elem):
    """Add elem to side in state. Returns True if added, False otherwise."""
    elems = state['sets'][side]
    sum_bits = state['sum_bits'][side]
    diff_bits = state['diff_bits'][side]
    other_diff = state['diff_bits'][1 - side]
    
    if not _can_add(elem, elems, sum_bits, diff_bits, other_diff):
        return False
        
    # Update
    sum_bits |= (1 << (elem + elem))
    for a in elems:
        sum_bits |= (1 << (elem + a))
        d = abs(elem - a)
        diff_bits |= (1 << d)
        
    state['sets'][side].append(elem)
    state['sum_bits'][side] = sum_bits
    state['diff_bits'][side] = diff_bits
    return True

def _remove_elem(state, side, idx):
    """Remove element at index idx from side in state."""
    elems = state['sets'][side]
    elems.pop(idx)
    state['sum_bits'][side], state['diff_bits'][side] = _rebuild_bits(elems)

def _greedy_init(order):
    """Greedy construction using a specific order."""
    state = {
        'sets': [[], []],
        'sum_bits': [0, 0],
        'diff_bits': [0, 0]
    }
    for elem in order:
        if len(state['sets'][0]) < MAX_EACH:
            if _add_elem(state, 0, elem):
                continue
        if len(state['sets'][1]) < MAX_EACH:
            if _add_elem(state, 1, elem):
                continue
    return state

def _local_search(state, max_time):
    """
    Local search:
    1. Try to add any available element.
    2. Try 1-for-1 swaps.
    3. Try 1-for-2 swaps.
    """
    improved = True
    while improved and time.time() < max_time:
        improved = False
        all_used = set(state['sets'][0]) | set(state['sets'][1])
        
        # 1. Try adding new elements
        for side in (0, 1):
            if len(state['sets'][side]) >= MAX_EACH:
                continue
            # Randomize candidates to escape deterministic traps
            candidates = [x for x in range(1, N + 1) if x not in all_used]
            random.shuffle(candidates)
            for elem in candidates:
                if time.time() > max_time:
                    return state
                if _add_elem(state, side, elem):
                    improved = True
                    all_used.add(elem)
                    break # Restart loop to re-evaluate all_used and other side
            
        if improved:
            continue

        # 2. Try 1-for-1 swaps
        for side in (0, 1):
            elems = state['sets'][side]
            if not elems:
                continue
            # Randomize indices
            indices = list(range(len(elems)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > max_time:
                    return state
                removed = elems[i]
                # Temporarily remove
                _remove_elem(state, side, i)
                all_used.discard(removed)
                
                # Try to add a new element
                candidates = [x for x in range(1, N + 1) if x not in all_used]
                random.shuffle(candidates)
                added = False
                for elem in candidates:
                    if _add_elem(state, side, elem):
                        added = True
                        all_used.add(elem)
                        break
                
                if not added:
                    # Restore
                    _add_elem(state, side, removed)
                    all_used.add(removed)
                else:
                    improved = True
                    break # Restart loop

        if improved:
            continue

        # 3. Try 1-for-2 swaps
        # This is expensive, so we limit attempts
        swap_attempts = 0
        max_swap_attempts = 50
        for side in (0, 1):
            if len(state['sets'][side]) >= MAX_EACH:
                continue
            elems = state['sets'][side]
            if not elems:
                continue
            indices = list(range(len(elems)))
            random.shuffle(indices)
            
            for i in indices:
                if swap_attempts >= max_swap_attempts:
                    break
                if time.time() > max_time:
                    return state
                
                removed = elems[i]
                _remove_elem(state, side, i)
                all_used.discard(removed)
                
                # Try to add two elements
                candidates = [x for x in range(1, N + 1) if x not in all_used]
                random.shuffle(candidates)
                
                added_count = 0
                added_elems = []
                for elem in candidates:
                    if added_count == 2:
                        break
                    if _add_elem(state, side, elem):
                        added_count += 1
                        added_elems.append(elem)
                        all_used.add(elem)
                    if time.time() > max_time:
                        break
                
                if added_count == 2:
                    improved = True
                    break
                else:
                    # Restore
                    _add_elem(state, side, removed)
                    all_used.add(removed)
                    # Remove the ones we added
                    for ae in added_elems:
                        # Find index
                        for j, e in enumerate(state['sets'][side]):
                            if e == ae:
                                _remove_elem(state, side, j)
                                break
                        all_used.discard(ae)
                swap_attempts += 1
            
            if improved:
                break

    return state

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    best_state = None
    best_size = 0
    
    random.seed(42)
    
    # Multi-start
    while time.time() < deadline - 5.0:
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        state = _greedy_init(order)
        state = _local_search(state, deadline - 2.0)
        
        size = len(state['sets'][0]) + len(state['sets'][1])
        if size > best_size:
            best_size = size
            best_state = {
                'sets': [list(state['sets'][0]), list(state['sets'][1])],
            }
            
        if best_size >= 2 * MAX_EACH:
            break
            
    if best_state is None:
        # Fallback
        state = _greedy_init(list(range(1, N + 1)))
        best_state = {
            'sets': [list(state['sets'][0]), list(state['sets'][1])],
        }
        
    return best_state['sets']
# EVOLVE-BLOCK-END
