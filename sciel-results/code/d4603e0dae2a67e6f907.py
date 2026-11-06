import time
import random

def _precompute_aps_by_pos(n=128):
    aps_by_pos = [[] for _ in range(n)]
    all_aps = []
    for a in range(n):
        for d in range(1, (n - a - 1) // 3 + 1):
            if a + 3 * d < n:
                ap = (a, a + d, a + 2 * d, a + 3 * d)
                all_aps.append(ap)
                for idx, pos in enumerate(ap):
                    aps_by_pos[pos].append((ap, idx))
    return all_aps, aps_by_pos

_ALL_APS, _APS_BY_POS = _precompute_aps_by_pos()

def _check_valid_at(colors, pos, color, n=128):
    for (ap, idx) in _APS_BY_POS[pos]:
        i0, i1, i2, i3 = ap
        c0 = color if idx == 0 else colors[i0]
        c1 = color if idx == 1 else colors[i1]
        c2 = color if idx == 2 else colors[i2]
        c3 = color if idx == 3 else colors[i3]
        if -1 in (c0, c1, c2, c3):
            continue
        if len({c0, c1, c2, c3}) < 3:
            return False
    return True

def _is_valid_full(colors, n=128):
    for ap in _ALL_APS:
        i0, i1, i2, i3 = ap
        if len({colors[i0], colors[i1], colors[i2], colors[i3]}) < 3:
            return False
    return True

def _greedy_construct(num_colors, n=128, color_order=None):
    colors = [-1] * n
    if color_order is None:
        color_order = list(range(num_colors))
    for pos in range(n):
        placed = False
        for color in color_order:
            if _check_valid_at(colors, pos, color, n):
                colors[pos] = color
                placed = True
                break
        if not placed:
            return None
    return colors

def _randomized_greedy(num_colors, n=128, deadline=None, max_attempts=500):
    attempts = 0
    while attempts < max_attempts:
        if deadline is not None and time.time() > deadline:
            break
        attempts += 1
        color_order = list(range(num_colors))
        random.shuffle(color_order)
        result = _greedy_construct(num_colors, n, color_order)
        if result is not None:
            return result
    return None

def _backtrack_construct(num_colors, n=128, max_nodes=1000000, deadline=None):
    colors = [-1] * n
    color_order = list(range(num_colors))
    random.shuffle(color_order)
    node_count = [0]
    
    def backtrack(pos):
        if deadline is not None and time.time() > deadline:
            return False
        node_count[0] += 1
        if node_count[0] > max_nodes:
            return False
        if pos == n:
            return True
        for color in color_order:
            if _check_valid_at(colors, pos, color, n):
                colors[pos] = color
                if backtrack(pos + 1):
                    return True
                colors[pos] = -1
        return False
    
    if backtrack(0):
        return colors
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    
    # Try 3 colors with greedy
    result = _greedy_construct(3, n)
    if result is not None:
        return result
    
    # Try randomized greedy with 3 colors
    if time.time() < deadline - 15:
        result = _randomized_greedy(3, n, deadline - 15, max_attempts=500)
        if result is not None:
            return result
    
    # Try backtracking with 3 colors
    if time.time() < deadline - 10:
        result = _backtrack_construct(3, n, max_nodes=500000, deadline=deadline - 10)
        if result is not None:
            return result
    
    # Try 4 colors
    result = _greedy_construct(4, n)
    if result is not None:
        return result
    
    if time.time() < deadline - 10:
        result = _randomized_greedy(4, n, deadline - 10, max_attempts=200)
        if result is not None:
            return result
    
    if time.time() < deadline - 5:
        result = _backtrack_construct(4, n, max_nodes=1000000, deadline=deadline - 5)
        if result is not None:
            return result
    
    # Try 5 colors
    result = _greedy_construct(5, n)
    if result is not None:
        return result
    
    # Fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                if len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            colors.append(color)
            break
        else:
            colors.append(0)
    return colors
