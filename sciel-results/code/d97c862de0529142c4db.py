"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _build_index(n):
    """Build mapping from triple (a,b,c) to index in lexicographic order."""
    triples = list(combinations(range(n), 3))
    idx = {}
    for i, t in enumerate(triples):
        idx[t] = i
    return triples, idx

def _is_valid_fast(colors, n):
    """Check if coloring is valid (no red K4, no blue K5) - optimized."""
    triples, idx = _build_index(n)
    
    # Check red K4: no 4 vertices with all 4 triples red
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        if (colors[idx[(a,b,c)]] == 0 and 
            colors[idx[(a,b,d)]] == 0 and 
            colors[idx[(a,c,d)]] == 0 and 
            colors[idx[(b,c,d)]] == 0):
            return False
    
    # Check blue K5: no 5 vertices with all 10 triples blue
    for quint in combinations(range(n), 5):
        all_blue = True
        for tri in combinations(quint, 3):
            if colors[idx[tri]] != 1:
                all_blue = False
                break
        if all_blue:
            return False
    
    return True

def _try_construction_z34(n):
    """Construction using Z_34 structure."""
    if n != 34:
        return None
    triples, idx = _build_index(n)
    colors = bytearray(len(triples))
    
    # Color based on sum mod 34: red if sum in a specific set
    # Try: red if (a+b+c) % 34 is even, blue otherwise
    for a, b, c in triples:
        s = (a + b + c) % 34
        colors[idx[(a,b,c)]] = 0 if s % 2 == 0 else 1
    
    if _is_valid_fast(colors, n):
        return list(colors)
    return None

def _try_construction_z17x2(n):
    """Construction using Z_17 x {0,1}."""
    if n != 34:
        return None
    triples, idx = _build_index(n)
    colors = bytearray(len(triples))
    
    # Map vertex v to (v % 17, v // 17)
    for a, b, c in triples:
        a1, a2 = a % 17, a // 17
        b1, b2 = b % 17, b // 17
        c1, c2 = c % 17, c // 17
        
        # Color red if the Z_17 projections sum to 0 mod 17
        s = (a1 + b1 + c1) % 17
        colors[idx[(a,b,c)]] = 0 if s == 0 else 1
    
    if _is_valid_fast(colors, n):
        return list(colors)
    return None

def _try_construction_quadratic(n):
    """Construction using quadratic form."""
    if n != 34:
        return None
    triples, idx = _build_index(n)
    colors = bytearray(len(triples))
    
    # Use: red if (a-b)^2 + (b-c)^2 + (c-a)^2 is a quadratic residue mod 34
    # Or simpler: red if (a-b)*(b-c)*(c-a) % 34 == 0
    for a, b, c in triples:
        d1 = (b - a) % 34
        d2 = (c - b) % 34
        d3 = (a - c) % 34
        # Color red if any difference is 0 mod 17
        red = (d1 % 17 == 0) or (d2 % 17 == 0) or (d3 % 17 == 0)
        colors[idx[(a,b,c)]] = 0 if red else 1
    
    if _is_valid_fast(colors, n):
        return list(colors)
    return None

def _try_construction_ap(n):
    """Construction: red if triple is an arithmetic progression mod 17."""
    if n != 34:
        return None
    triples, idx = _build_index(n)
    colors = bytearray(len(triples))
    
    for a, b, c in triples:
        a1, b1, c1 = a % 17, b % 17, c % 17
        # Check if {a1, b1, c1} forms an AP mod 17
        # Sort them
        s = sorted([a1, b1, c1])
        x, y, z = s[0], s[1], s[2]
        # AP if (y-x) == (z-y) mod 17, or (y-x)+(z-y) == (z-x) which is always true
        # AP means: 2*y ≡ x+z (mod 17)
        is_ap = (2 * y - x - z) % 17 == 0
        colors[idx[(a,b,c)]] = 0 if is_ap else 1
    
    if _is_valid_fast(colors, n):
        return list(colors)
    return None

def _try_construction_sum_range(n):
    """Construction: red if sum mod 34 is in [0, 16], blue otherwise."""
    if n != 34:
        return None
    triples, idx = _build_index(n)
    colors = bytearray(len(triples))
    
    for a, b, c in triples:
        s = (a + b + c) % 34
        colors[idx[(a,b,c)]] = 0 if s <= 16 else 1
    
    if _is_valid_fast(colors, n):
        return list(colors)
    return None

def _try_construction_product(n):
    """Construction: red if product of differences is 0 mod 17."""
    if n != 34:
        return None
    triples, idx = _build_index(n)
    colors = bytearray(len(triples))
    
    for a, b, c in triples:
        a1, b1, c1 = a % 17, b % 17, c % 17
        prod = ((b1 - a1) * (c1 - b1) * (a1 - c1)) % 17
        colors[idx[(a,b,c)]] = 0 if prod == 0 else 1
    
    if _is_valid_fast(colors, n):
        return list(colors)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Try known constructions for n=34
    for n in range(34, 8, -1):
        if time.time() > deadline:
            break
        
        if n == 34:
            # Try multiple algebraic constructions
            for func in [_try_construction_z34, _try_construction_z17x2, 
                        _try_construction_quadratic, _try_construction_ap,
                        _try_construction_sum_range, _try_construction_product]:
                if time.time() > deadline:
                    break
                result = func(n)
                if result is not None:
                    return {'n': n, 'colors': result}
        
        # Fallback for smaller n: use modular coloring
        if n <= 10:
            triples = list(combinations(range(n), 3))
            colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
            if _is_valid_fast(bytearray(colors), n):
                return {'n': n, 'colors': colors}
    
    # Final fallback: n=9
    n = 9
    triples = list(combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
