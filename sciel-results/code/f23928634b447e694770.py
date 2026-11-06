"""Two Sidon sets with disjoint difference sets – Bitmask + local search."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 52.0  # safety margin before 60s kill


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
    """Add elem to side. state = [elems0, elems1, sum_bits0, sum_bits1, diff_bits0, diff_bits1]"""
    elems = state[side]
    sum_bits = state[2 + side]
    diff_bits = state[4 + side]
    other_diff_bits = state[4 + (1 - side)]
    
    if not _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
        return False
    
    # Update sum_bits: add elem+elem and elem+a for all a in elems
    sum_bits |= (1 << (elem + elem))
    for a in elems:
        sum_bits |= (1 << (elem + a))
    
    # Update diff_bits: add |elem-a| for all a in elems
    for a in elems:
        diff_bits |= (1 << abs(elem - a))
    
    state[side] = elems + [elem]
    state[2 + side] = sum_bits
    state[4 + side] = diff_bits
    return True


def _remove_elem(state, side, elem):
    """Remove elem from side. Returns True if removed successfully."""
    elems = state[side]
    if elem not in elems:
        return False
    
    # Rebuild sum_bits and diff_bits from scratch (simpler and safe)
    new_elems = [x for x in elems if x != elem]
    
    sum_bits = 0
    diff_bits = 0
    for i in range(len(new_elems)):
        for j in range(i, len(new_elems)):
            sum_bits |= (1 << (new_elems[i] + new_elems[j]))
        for j in range(len(new_elems)):
            if i != j:
                diff_bits |= (1 << abs(new_elems[i] - new_elems[j]))
    
    state[side] = new_elems
    state[2 + side] = sum_bits
    state[4 + side] = diff_bits
    return True


def _validate(state):
    """Validate the state is a valid solution. Returns (valid, total_size)."""
    total = 0
    for side in range(2):
        elems = state[side]
        total += len(elems)
        
        # Check Sidon: all pair sums (with replacement) must be unique
        sums = set()
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                s = elems[i] + elems[j]
                if s in sums:
                    return False, 0
                sums.add(s)
        
        # Check differences are disjoint with other set
        diffs = set()
        for i in range(len(elems)):
            for j in range(len(elems)):
                if i != j:
                    d = abs(elems[i] - elems[j])
                    if d not in diffs:
                        diffs.add(d)
                    else:
                        return False, 0
        
        # Check disjointness with other set's diffs
        other_diffs = set()
        other_elems = state[1 - side]
        for i in range(len(other_elems)):
            for j in range(len(other_elems)):
                if i != j:
                    other_diffs.add(abs(other_elems[i] - other_elems[j]))
        
        if diffs & other_diffs:
            return False, 0
    
    return True, total


def _greedy_build():
    """Build initial solution using greedy approach."""
    state = [[], [], 0, 0, 0, 0]
    
    for x in range(1, N + 1):
        for side in (0, 1):
            if len(state[side]) < MAX_EACH:
                if _can_add(x, state[side], state[2 + side], state[4 + side], state[4 + (1 - side)]):
                    _add_elem(state, side, x)
    
    return state


def _local_search(state, deadline):
    """Perform 1-for-1 and 1-for-2 swap local search."""
    best_state = [list(state[0]), list(state[1]), state[2], state[3], state[4], state[5]]
    best_size = len(best_state[0]) + len(best_state[1])
    
    while time.time() < deadline:
        improved = False
        
        # Try 1-for-1 swaps: remove one elem from a side, add another
        for side in range(2):
            if len(state[side]) == 0:
                continue
            
            # Try removing each element and adding a different one
            for remove_idx in range(len(state[side])):
                remove_elem = state[side][remove_idx]
                
                # Try adding each candidate not already in the set
                for add_elem in range(1, N + 1):
                    if add_elem in state[side]:
                        continue
                    
                    # Create a copy of the state
                    new_state = [list(state[0]), list(state[1]), state[2], state[3], state[4], state[5]]
                    
                    # Remove the element
                    _remove_elem(new_state, side, remove_elem)
                    
                    # Try to add the new element
                    if len(new_state[side]) < MAX_EACH:
                        if _can_add(add_elem, new_state[side], new_state[2 + side], new_state[4 + side], new_state[4 + (1 - side)]):
                            _add_elem(new_state, side, add_elem)
                            new_size = len(new_state[0]) + len(new_state[1])
                            if new_size > best_size:
                                best_state = [list(new_state[0]), list(new_state[1]), new_state[2], new_state[3], new_state[4], new_state[5]]
                                best_size = new_size
                                state = [list(new_state[0]), list(new_state[1]), new_state[2], new_state[3], new_state[4], new_state[5]]
                                improved = True
        
        if not improved:
            break
    
    return best_state, best_size


def construct():
    start_time = time.time()
    
    # Try multiple random restarts with different orderings
    best_state = None
    best_size = 0
    
    # First, greedy build
    state = _greedy_build()
    valid, size = _validate(state)
    if valid and size > best_size:
        best_state = [list(state[0]), list(state[1]), state[2], state[3], state[4], state[5]]
        best_size = size
    
    # Local search from greedy
    if time.time() < DEADLINE:
        improved_state, improved_size = _local_search(state, DEADLINE - 5)
        if improved_size > best_size:
            best_state = improved_state
            best_size = improved_size
    
    # Try a few more random initializations
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 10:
        for restart in range(3):
            if time.time() > DEADLINE:
                break
            
            # Random permutation of elements
            order = list(range(1, N + 1))
            random.shuffle(order)
            
            state = [[], [], 0, 0, 0, 0]
            for x in order:
                for side in (0, 1):
                    if len(state[side]) < MAX_EACH:
                        if _can_add(x, state[side], state[2 + side], state[4 + side], state[4 + (1 - side)]):
                            _add_elem(state, side, x)
            
            valid, size = _validate(state)
            if valid and size > best_size:
                best_state = [list(state[0]), list(state[1]), state[2], state[3], state[4], state[5]]
                best_size = size
            
            # Local search if time permits
            if time.time() < DEADLINE - 10:
                improved_state, improved_size = _local_search(state, DEADLINE - 5)
                if improved_size > best_size:
                    best_state = improved_state
                    best_size = improved_size
    
    # Ensure nonempty
    if best_state is None or len(best_state[0]) == 0 or len(best_state[1]) == 0:
        # Fallback: simple valid solution
        A = [1, 2, 4, 8]
        B = [3, 5, 9, 16]
        best_state = [A, B, 0, 0, 0, 0]
    
    return [best_state[0], best_state[1]]
# EVOLVE-BLOCK-END
