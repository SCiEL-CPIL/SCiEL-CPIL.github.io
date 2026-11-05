"""5-colourings of K_n without monochromatic C4.
Uses algebraic constructions over GF(25) and Z_2 x Z_13, with validation.
"""
import time
import random

_START = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _time_left():
    return _LIMIT - _MARGIN - (time.time() - _START)

def _validate(n, colors):
    """Check that no colour class contains a 4-cycle (C4)."""
    if len(colors) != n * (n - 1) // 2:
        return False
    if any(c < 0 or c > 4 for c in colors):
        return False
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

def _cyclic_construction(n):
    """Cyclic 5-colouring using circular distance mapped to colour (d-1)%5."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            d = min(d, n - d)
            c = (d - 1) % 5
            colors.append(c)
    return {'n': n, 'colors': colors}

def _gf25_construction():
    """
    Construction using GF(25) = GF(5^2).
    Vertices: 25 elements of GF(25).
    GF(25)* is cyclic of order 24. We partition into cosets.
    The subgroup of 6th powers has order 4 (since 24/6=4), giving 6 cosets of size 4.
    Wait, let me reconsider: 24 = 5*4 + 4, no. 24/5 is not integer.
    
    Let's use: GF(25)* has order 24. We want to split into 5 groups.
    24 = 5*4 + 4, so we can have 4 groups of 5 and 1 group of 4, or similar.
    
    Actually, a better approach: use the fact that we need the colour classes to be
    C4-free. A sufficient condition is that each colour class is a sum-free set in some sense.
    
    Let me try: pick a primitive element g in GF(25). The powers g^0, g^1, ..., g^23
    are all non-zero elements. Assign colour based on floor(k/5) or similar.
    
    But this is hard to verify without implementing GF(25). Let me do it.
    """
    # GF(25) = GF(5)[x]/(x^2 + 1) since x^2+1 is irreducible over GF(5)
    # Check: x^2+1 mod 5: x=0->1, x=1->2, x=2->0 (4+1=5=0), so x=2 is a root? 4+1=5=0 mod 5. Yes!
    # So x^2+1 = (x-2)(x+2) = (x-2)(x-3) mod 5. Not irreducible!
    # Try x^2+2: x=0->2, x=1->3, x=2->1, x=3->11=1, x=4->18=3. No roots. Irreducible!
    # So GF(25) = GF(5)[x]/(x^2+2)
    
    # Elements are a + b*alpha where alpha^2 = -2 = 3 mod 5
    # a, b in {0,1,2,3,4}
    
    def add(p, q):
        return ((p[0]+q[0])%5, (p[1]+q[1])%5)
    
    def sub(p, q):
        return ((p[0]-q[0])%5, (p[1]-q[1])%5)
    
    def mul(p, q):
        # (a+b*alpha)(c+d*alpha) = ac + (ad+bc)*alpha + bd*alpha^2 = ac + 3*bd + (ad+bc)*alpha
        a, b = p
        c, d = q
        real = (a*c + 3*b*d) % 5
        imag = (a*d + b*c) % 5
        return (real, imag)
    
    def neg(p):
        return ((-p[0])%5, (-p[1])%5)
    
    # Find primitive element: an element whose order is 24
    # Try alpha = (0,1)
    def pow_elem(e, n):
        result = (1, 0)
        base = e
        while n > 0:
            if n % 2 == 1:
                result = mul(result, base)
            base = mul(base, base)
            n //= 2
        return result
    
    # Check if alpha = (0,1) is primitive
    alpha = (0, 1)
    # alpha^2 = (3,0), alpha^4 = mul((3,0),(3,0)) = (9%5, 0) = (4,0)
    # alpha^12 = alpha^4 cubed = (4,0)^3 = (64%5, 0) = (4,0). Not 1, so order doesn't divide 12.
    # alpha^24 should be 1 if alpha is in the group.
    # Let's just enumerate powers
    powers = []
    cur = (1, 0)
    for i in range(24):
        powers.append(cur)
        cur = mul(cur, alpha)
    # Check if all 24 are distinct
    if len(set(powers)) == 24:
        # alpha is primitive
        pass
    
    # Now assign colours: partition the 24 non-zero elements into 5 groups
    # 24 = 5*4 + 4, so 4 groups of 5 and 1 group of 4, or 5 groups with sizes summing to 24
    # Let's try: colours 0-3 get 5 elements each, colour 4 gets 4 elements
    # Or: use the cyclic structure. Colour k gets elements g^(5k), g^(5k+1), g^(5k+2), g^(5k+3) for k=0,1,2,3
    # and colour 4 gets g^20, g^21, g^22, g^23
    
    # Actually, let's try a different partition that's more likely to be C4-free.
    # Use: colour based on the "quadratic character" and further splitting.
    
    # For now, let's try: colour = (exponent // 5) % 5, but 24/5 = 4.8
    # exponents 0-23: colour = exponent // 5 gives 0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4
    # That's 5,5,5,5,4. 
    
    # Let me just try this and validate
    colors = []
    # Map each field element to its exponent (or -1 for zero)
    elem_to_exp = {}
    for i, p in enumerate(powers):
        elem_to_exp[p] = i
    
    # Vertices: all 25 elements of GF(25)
    # Index them: 0=(0,0), 1..24 = powers[0]..powers[23]
    # Actually, let's list all elements
    all_elems = [(a, b) for a in range(5) for b in range(5)]
    # Create a mapping from element to index
    elem_to_idx = {e: i for i, e in enumerate(all_elems)}
    
    n = 25
    for i in range(n):
        ei = all_elems[i]
        for j in range(i+1, n):
            ej = all_elems[j]
            diff = sub(ej, ei)
            if diff == (0, 0):
                continue  # shouldn't happen
            exp = elem_to_exp.get(diff, None)
            if exp is None:
                # diff is not in powers, should not happen for non-zero
                c = 0
            else:
                c = exp // 5
            colors.append(c)
    
    return {'n': n, 'colors': colors}

def _z2_z13_construction():
    """
    Construction using Z_2 x Z_13.
    Vertices: (x, y) where x in {0,1}, y in Z_13. Total 26 vertices.
    
    Use quadratic residues in Z_13 to define colouring.
    QR = {1, 3, 4, 9, 10, 12}, QNR = {2, 5, 6, 7, 8, 11}
    
    For edge between (x1,y1) and (x2,y2):
      dx = (x2-x1) mod 2, dy = (y2-y1) mod 13
      If dx == 0:
        if dy in QR: colour based on dy value
        else: colour based on dy value
      If dx == 1:
        different mapping
    """
    n = 26
    qr = {1, 3, 4, 9, 10, 12}
    qnr = {2, 5, 6, 7, 8, 11}
    
    # Map each non-zero element of Z_13 to a colour in 0..4
    # 12 non-zero elements, 5 colours: 3 colours get 2-3 elements
    # Let's try: split QR into 3 groups of 2, QNR into 3 groups of 2
    # QR: {1,3}, {4,9}, {10,12} -> colours 0,1,2
    # QNR: {2,5}, {6,7}, {8,11} -> colours 3,4,? 
    # That's only 5 colours for 12 elements: 3*2 + 2*2 = 10, need 2 more.
    # Let's do: QR -> colours 0,1,2 (2 each), QNR -> colours 3,4 (3 each)
    # {1,3}->0, {4,9}->1, {10,12}->2, {2,5,6}->3, {7,8,11}->4
    
    color_map_same = {}
    for d in [1, 3]:
        color_map_same[d] = 0
    for d in [4, 9]:
        color_map_same[d] = 1
    for d in [10, 12]:
        color_map_same[d] = 2
    for d in [2, 5, 6]:
        color_map_same[d] = 3
    for d in [7, 8, 11]:
        color_map_same[d] = 4
    
    # For dx=1, use a shifted mapping
    color_map_diff = {}
    for d in [1, 2]:
        color_map_diff[d] = 0
    for d in [3, 5]:
        color_map_diff[d] = 1
    for d in [4, 6]:
        color_map_diff[d] = 2
    for d in [7, 10]:
        color_map_diff[d] = 3
    for d in [8, 9, 11, 12]:
        color_map_diff[d] = 4
    
    colors = []
    for i in range(n):
        x1 = i % 2
        y1 = i // 2
        for j in range(i + 1, n):
            x2 = j % 2
            y2 = j // 2
            dx = (x2 - x1) % 2
            dy = (y2 - y1) % 13
            if dx == 0:
                if dy == 0:
                    c = 0  # shouldn't happen for distinct vertices with same x
                else:
                    c = color_map_same.get(dy, 0)
            else:
                if dy == 0:
                    c = 4
                else:
                    c = color_map_diff.get(dy, 0)
            colors.append(c)
    
    return {'n': n, 'colors': colors}

def _greedy_with_repair(n, max_iter=50000, seed=42):
    """
    Greedy construction with local search to find a valid 5-colouring of K_n.
    """
    random.seed(seed)
    
    m = n * (n - 1) // 2
    # Start with cyclic-like initialisation
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = min(j - i, n - j + i)
            colors.append((d - 1) % 5)
    
    def _edge_idx(a, b):
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
        # Try to recolor one of the edges in the C4
        edges_in_c4 = [(i, common[0]), (i, common[1]), (j, common[0]), (j, common[1])]
        random.shuffle(edges_in_c4)
        for a, b in edges_in_c4:
            if a > b:
                a, b = b, a
            idx = _edge_idx(a, b)
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
    
    # Try cyclic constructions for various n
    for n in [25, 20, 15, 10, 5]:
        if _time_left() < 5:
            break
        res = _cyclic_construction(n)
        if _validate(n, res['colors']):
            if n > best['n']:
                best = res
    
    # Try GF(25) construction
    if _time_left() > 10:
        res = _gf25_construction()
        if _validate(res['n'], res['colors']):
            if res['n'] > best['n']:
                best = res
    
    # Try Z_2 x Z_13 construction
    if _time_left() > 10:
        res = _z2_z13_construction()
        if _validate(res['n'], res['colors']):
            if res['n'] > best['n']:
                best = res
    
    # Try greedy with repair for larger n
    if best['n'] < 26 and _time_left() > 30:
        res = _greedy_with_repair(26, max_iter=30000)
        if res and res['n'] > best['n']:
            best = res
    
    if best['n'] < 25 and _time_left() > 20:
        res = _greedy_with_repair(25, max_iter=30000, seed=123)
        if res and res['n'] > best['n']:
            best = res
    
    if best['n'] < 24 and _time_left() > 15:
        res = _greedy_with_repair(24, max_iter=20000, seed=456)
        if res and res['n'] > best['n']:
            best = res
    
    return best

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, valid={_validate(res['n'], res['colors'])}")
    print(res['colors'][:20])
