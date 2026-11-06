"""Two Sidon sets with disjoint difference sets – Bitset + multi-start local search."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
MAX_EACH = 64
DEADLINE = 55.0  # safety margin before 60s kill


def _can_add(elem, elems, sum_bits, diff_bits, other_diff_bits):
    """Check if elem can be added to elems maintaining Sidon + disjoint diffs."""
    s2 = elem + elem
    if s2 <= 512 and (sum_bits >> s2) & 1:
        return False
    for a in elems:
        s = elem + a
        if (sum_bits >> s) & 1:
            return False
        d = abs(elem - a)
        if (diff_bits >> d) & 1 or (other_diff_bits >> d) & 1:
            return False
    return True


def _add_elem(state, side, elem):
    """Add elem to side; update bitsets. Returns None if invalid."""
    elems = state['sets'][side]
    sum_bits = state['sum_bits'][side]
    diff_bits = state['diff_bits'][side]
    other_diff = state['diff_bits'][1 - side]

    if not _can_add(elem, elems, sum_bits, diff_bits, other_diff):
        return None

    s2 = elem + elem
    sum_bits |= (1 << s2)
    for a in elems:
        sum_bits |= (1 << (elem + a))

    for a in elems:
        d = abs(elem - a)
        diff_bits |= (1 << d)

    state['sets'][side].append(elem)
    state['sum_bits'][side] = sum_bits
    state['diff_bits'][side] = diff_bits
    state['all_used'].add(elem)
    return state


def _rebuild_bits(state, side):
    """Rebuild sum and diff bitsets for side from current set."""
    from itertools import combinations_with_replacement, combinations
    elems = state['sets'][side]
    new_sum = 0
    for a, b in combinations_with_replacement(elems, 2):
        new_sum |= (1 << (a + b))
    state['sum_bits'][side] = new_sum

    new_diff = 0
    for a, b in combinations(elems, 2):
        new_diff |= (1 << abs(a - b))
    state['diff_bits'][side] = new_diff


def _remove_elem(state, side, idx):
    """Remove element at index idx from side; rebuild bitsets."""
    elems = state['sets'][side]
    removed = elems[idx]
    elems.pop(idx)
    state['all_used'].discard(removed)
    _rebuild_bits(state, side)


def _greedy_build(order, max_time, balance=True):
    """Greedy build using given element order. Optionally balance set sizes."""
    state = {
        'sets': [[], []],
        'sum_bits': [0, 0],
        'diff_bits': [0, 0],
        'all_used': set()
    }
    for elem in order:
        if time.time() > max_time:
            break
        if balance:
            if len(state['sets'][0]) > len(state['sets'][1]):
                sides = [1, 0]
            else:
                sides = [0, 1]
        else:
            sides = [0, 1]
        
        for side in sides:
            if len(state['sets'][side]) < MAX_EACH:
                r = _add_elem(state, side, elem)
                if r is not None:
                    break
    return state


def _try_additions(state, max_time):
    """Try adding any unused element to either side. Returns count added."""
    added = 0
    for side in (0, 1):
        if len(state['sets'][side]) >= MAX_EACH:
            continue
        all_used = state['all_used']
        for candidate in range(1, N + 1):
            if candidate in all_used:
                continue
            if time.time() > max_time:
                return added
            r = _add_elem(state, side, candidate)
            if r is not None:
                added += 1
    return added


def _try_swaps(state, max_time):
    """Try 1-for-1 swaps to enable further additions. Returns True if any swap made."""
    improved = False
    for side in (0, 1):
        n = len(state['sets'][side])
        if n == 0:
            continue
        for i in range(n):
            if time.time() > max_time:
                return improved
            removed = state['sets'][side][i]
            _remove_elem(state, side, i)
            all_used = state['all_used']
            found = False
            for candidate in range(1, N + 1):
                if candidate in all_used or candidate == removed:
                    continue
                if time.time() > max_time:
                    _add_elem(state, side, removed)
                    return improved
                r = _add_elem(state, side, candidate)
                if r is not None:
                    found = True
                    break
            if not found:
                _add_elem(state, side, removed)
            else:
                improved = True
    return improved


def _local_search(state, max_time):
    """Iterate: add elements, then try swaps to unlock more additions."""
    while time.time() < max_time:
        added = _try_additions(state, max_time)
        if added > 0:
            continue
        swapped = _try_swaps(state, max_time)
        if not swapped:
            break
    return state


def _polish(state, max_time):
    """Final polish: exhaustive addition + targeted swaps."""
    _try_additions(state, max_time)
    for _ in range(3):
        if time.time() > max_time:
            break
        if not _try_swaps(state, max_time):
            break
        _try_additions(state, max_time)
    return state


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    best_state = None
    best_size = 0

    random.seed(42)
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        order = list(range(1, N + 1))
        random.shuffle(order)

        greedy_budget = min(deadline - 8.0, start_time + 0.5)
        state = _greedy_build(order, greedy_budget, balance=True)
        
        ls_budget = deadline - 5.0
        state = _local_search(state, ls_budget)
        state = _polish(state, deadline - 1.0)

        size = len(state['sets'][0]) + len(state['sets'][1])
        if size > best_size:
            best_size = size
            best_state = {
                'sets': [list(state['sets'][0]), list(state['sets'][1])],
                'sum_bits': [state['sum_bits'][0], state['sum_bits'][1]],
                'diff_bits': [state['diff_bits'][0], state['diff_bits'][1]],
                'all_used': set(state['all_used'])
            }

        if best_size >= 2 * MAX_EACH:
            break

    if best_state is None or best_size < 20:
        order = list(range(1, N + 1))
        state = _greedy_build(order, deadline - 2.0, balance=True)
        state = _local_search(state, deadline - 1.0)
        state = _polish(state, deadline - 0.5)
        size = len(state['sets'][0]) + len(state['sets'][1])
        if size > best_size:
            best_state = {
                'sets': [list(state['sets'][0]), list(state['sets'][1])],
                'sum_bits': [state['sum_bits'][0], state['sum_bits'][1]],
                'diff_bits': [state['diff_bits'][0], state['diff_bits'][1]],
                'all_used': set(state['all_used'])
            }

    return [best_state['sets'][0], best_state['sets'][1]]
# EVOLVE-BLOCK-END
