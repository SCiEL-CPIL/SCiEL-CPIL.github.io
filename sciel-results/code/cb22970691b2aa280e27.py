import time
import random

_START = time.time()
_LIMIT = 140.0  # safety margin before 150s kill

def _time_left():
    return _LIMIT - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4).
    Equivalent: for some colour c, two vertices have >=2 common neighbours in c.
    """
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
    # Build neighbour sets per colour
    nbrs = [[set() for _ in range(n)] for _ in range(5)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            nbrs[c][i].add(j)
            nbrs[c][j].add(i)
            idx += 1
    for c in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                if len(nbrs[c][i] & nbrs[c][j]) >= 2:
                    return False
    return True

def _seed_construction():
    """Original K11 seed: split into five Hamiltonian cycles by circular distance 1..5."""
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

def _lazebnik_woldar():
    """
    Lazebnik-Woldar algebraic construction for R_5(C4) >= 26.
    Uses GF(25) = GF(5^2) with alpha^2 = 3 (irreducible x^2+2 over GF(5)).
    Vertices: 25 elements of GF(25) + infinity = 26 vertices.
    """
    p = 5
    
    def gf25_mul(x, y):
        a1, b1 = x
        a2, b2 = y
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)
    
    def gf25_pow(x, n):
        result = (1, 0)
        base = x
        while n > 0:
            if n % 2 == 1:
                result = gf25_mul(result, base)
            base = gf25_mul(base, base)
            n //= 2
        return result
    
    def gf25_order(x):
        for d in [1, 2, 3, 4, 6, 8, 12, 24]:
            if gf25_pow(x, d) == (1, 0):
                return d
        return -1
    
    gf25_elements = [(a, b) for a in range(p) for b in range(p)]
    
    # Find primitive element
    primitive = None
    for elem in gf25_elements:
        if elem == (0, 0):
            continue
        if gf25_order(elem) == 24:
            primitive = elem
            break
    
    if primitive is None:
        return None
    
    # Generate all powers
    powers = []
    current = (1, 0)
    for i in range(24):
        powers.append(current)
        current = gf25_mul(current, primitive)
    
    log_map = {}
    for i, elem in enumerate(powers):
        log_map[elem] = i
    
    n = 26
    colors = []
    
    def get_color(i, j):
        if i == 25 or j == 25:
            other = j if i == 25 else i
            if other == 0:
                return 0
            else:
                return (log_map[gf25_elements[other]] % 4) + 1
        else:
            elem_i = gf25_elements[i]
            elem_j = gf25_elements[j]
            if elem_i == (0, 0) or elem_j == (0, 0):
                return 0
            else:
                log_i = log_map[elem_i]
                log_j = log_map[elem_j]
                return ((log_i - log_j) % 4) + 1
    
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(get_color(i, j))
    
    if _validate(n, colors):
        return {'n': n, 'colors': colors}
    return None

def _greedy_with_repair(n, max_iter=20000):
    """
    Greedy construction with local search to find a valid 5-colouring of K_n.
    Start with cyclic initialisation, then repair C4 violations.
    """
    random.seed(42)
    
    m = n * (n - 1) // 2
    # Start with cyclic-like initialisation
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = min(j - i, n - j + i)
            colors.append((d - 1) % 5)
    
    def edge_idx(a, b):
        if a > b:
            a, b = b, a
        return a * (n - 1) - a * (a + 1) // 2 + (b - a - 1)
    
    for _ in range(max_iter):
        if _time_left() < 15:
            break
        
        # Find a violation
        nbrs = [[set() for _ in range(n)] for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                nbrs[c][i].add(j)
                nbrs[c][j].add(i)
                idx += 1
        
        violation = None
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = nbrs[c][i] & nbrs[c][j]
                    if len(common) >= 2:
                        violation = (c, i, j, list(common)[:2])
                        break
                if violation:
                    break
            if violation:
                break
        
        if violation is None:
            return {'n': n, 'colors': colors}
        
        c, i, j, common = violation
        # Try to recolor one of the edges in the C4 to break it
        edges_in_c4 = [(i, common[0]), (i, common[1]), (j, common[0]), (j, common[1])]
        random.shuffle(edges_in_c4)
        for a, b in edges_in_c4:
            idx = edge_idx(a, b)
            new_c = (colors[idx] + 1 + random.randint(0, 3)) % 5
            colors[idx] = new_c
            break
    
    if _validate(n, colors):
        return {'n': n, 'colors': colors}
    return None

def construct():
    """Entry point: return the best 5-colouring found."""
    if _time_left() <= 0:
        return _seed_construction()
    
    best = _seed_construction()
    
    # Try Lazebnik-Woldar construction for n=26
    lw = _lazebnik_woldar()
    if lw is not None:
        best = lw
        return best
    
    # Try greedy with repair for n=26
    if _time_left() > 20:
        res = _greedy_with_repair(26, max_iter=10000)
        if res is not None:
            best = res
            return best
    
    # Try smaller n values
    for n in [25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]:
        if _time_left() < 10:
            break
        if n <= best['n']:
            break
        res = _greedy_with_repair(n, max_iter=5000)
        if res is not None and res['n'] > best['n']:
            best = res
    
    return best
