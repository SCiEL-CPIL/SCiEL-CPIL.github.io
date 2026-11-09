"""Bitmask-accelerated greedy cost-minimization for 5-colourings of K_n without monochromatic C4."""
import time
import random

# Deadline guard
_START_TIME = time.time()
DEADLINE = 118.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _get_idx(n, a, b):
    """Get edge index for pair (a,b) where a < b in lexicographic order."""
    return a * n - a * (a + 1) // 2 + (b - a - 1)

def _validate_coloring(n, colors):
    """Check if coloring is valid (no monochromatic C4). Returns True if valid."""
    if len(colors) != n * (n - 1) // 2:
        return False
    # Build adjacency bitmasks per color
    adj = [0] * (5 * n)  # adj[color * n + vertex] = bitmask of neighbors in that color
    i = 0
    for a in range(n):
        for b in range(a + 1, n):
            c = colors[i]
            adj[c * n + a] |= (1 << b)
            adj[c * n + b] |= (1 << a)
            i += 1
    # Check for C4: two vertices with >=2 common neighbors in same color
    for c in range(5):
        base = c * n
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[base + u] & adj[base + v]
                if common & (common - 1):  # >=2 bits set
                    return False
    return True

def _count_c4_violations(n, colors):
    """Count total C4 violations across all colors using bitmasks."""
    adj = [0] * (5 * n)
    i = 0
    for a in range(n):
        for b in range(a + 1, n):
            c = colors[i]
            adj[c * n + a] |= (1 << b)
            adj[c * n + b] |= (1 << a)
            i += 1
    total = 0
    for c in range(5):
        base = c * n
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[base + u] & adj[base + v]
                if common & (common - 1):
                    total += 1
    return total

def _greedy_cost_minimize(n, colors, timeout=8.0):
    """Greedy local search using bitmask-accelerated C4 counting.
    Tries recoloring edges to reduce total C4 violations."""
    start_t = time.time()
    colors = list(colors)
    current_viol = _count_c4_violations(n, colors)
    
    if current_viol == 0:
        return colors
    
    best_viol = current_viol
    best_colors = colors[:]
    n_edges = n * (n - 1) // 2
    
    iters = 0
    while True:
        if time.time() - start_t > timeout:
            break
        if _time_left() < 5.0:
            break
        iters += 1
        
        # Pick random edge
        eidx = random.randint(0, n_edges - 1)
        old_color = colors[eidx]
        new_color = (old_color + 1 + random.randint(0, 3)) % 5
        
        colors[eidx] = new_color
        new_viol = _count_c4_violations(n, colors)
        
        if new_viol <= current_viol:
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_colors = colors[:]
            if current_viol == 0:
                break
        else:
            colors[eidx] = old_color
    
    return best_colors

def _try_construct(n, timeout=8.0):
    """Try to find a valid 5-coloring of K_n. Returns colors list or None."""
    n_edges = n * (n - 1) // 2
    colors = [random.randint(0, 4) for _ in range(n_edges)]
    colors = _greedy_cost_minimize(n, colors, timeout=timeout)
    if _count_c4_violations(n, colors) == 0:
        return colors
    return None

def _construct_cyclic(n):
    """Try a cyclic/symmetric construction as initial guess."""
    n_edges = n * (n - 1) // 2
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            # Use a simple modular pattern
            c = (d * 7 + i * 3) % 5
            colors.append(c)
    colors = _greedy_cost_minimize(n, colors, timeout=4.0)
    if _count_c4_violations(n, colors) == 0:
        return colors
    return None

def construct():
    random.seed(42)
    
    # Fallback: small valid coloring
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try from high n down, with multiple restarts
    for target_n in [29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11]:
        if _time_left() < 10.0:
            break
        if target_n <= best_n:
            break
        
        # Try cyclic construction first
        if _time_left() > 12.0:
            result = _construct_cyclic(target_n)
            if result is not None:
                best_n = target_n
                best_colors = result
                continue
        
        # Try multiple random restarts
        max_restarts = 15
        for attempt in range(max_restarts):
            if _time_left() < 8.0:
                break
            result = _try_construct(target_n, timeout=5.0)
            if result is not None:
                best_n = target_n
                best_colors = result
                break
    
    return {'n': best_n, 'colors': best_colors}
