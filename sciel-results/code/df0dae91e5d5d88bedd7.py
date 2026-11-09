import time
import random

def _verify(n, colors):
    N = n
    col = [[0] * N for _ in range(N)]
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            col[i][j] = col[j][i] = colors[k]
            k += 1
    for i in range(N):
        for j in range(i + 1, N):
            for l in range(j + 1, N):
                if col[i][j] == col[j][l] == col[i][l]:
                    return False
    return True

def _gf49_4coloring():
    """Construct 4-coloring of K_49 using GF(49) structure."""
    # GF(49) = GF(7)[x]/(x^2 - 3) where 3 is not a square mod 7
    def add(e1, e2):
        return ((e1[0] + e2[0]) % 7, (e1[1] + e2[1]) % 7)
    def mul(e1, e2):
        a = (e1[0]*e2[0] + 3*e1[1]*e2[1]) % 7
        b = (e1[0]*e2[1] + e2[0]*e1[1]) % 7
        return (a, b)
    
    elements = [(a, b) for a in range(7) for b in range(7)]
    idx = {e: i for i, e in enumerate(elements)}
    
    add_table = [[0]*49 for _ in range(49)]
    for i, e1 in enumerate(elements):
        for j, e2 in enumerate(elements):
            add_table[i][j] = idx[add(e1, e2)]
    mul_table = [[0]*49 for _ in range(49)]
    for i, e1 in enumerate(elements):
        for j, e2 in enumerate(elements):
            mul_table[i][j] = idx[mul(e1, e2)]
    
    zero_idx = idx[(0, 0)]
    one_idx = idx[(1, 0)]
    nonzero = [i for i in range(49) if i != zero_idx]
    
    # Find generator of GF(49)* (order 48)
    gen = None
    for g in nonzero:
        seen = set()
        cur = g
        for _ in range(48):
            seen.add(cur)
            cur = mul_table[cur][g]
        if len(seen) == 48:
            gen = g
            break
    if gen is None:
        return None
    
    powers = [one_idx]
    cur = one_idx
    for _ in range(47):
        cur = mul_table[cur][gen]
        powers.append(cur)
    
    # Subgroup of order 16: gen^3
    h = powers[3]
    subgroup = [one_idx]
    cur = one_idx
    for _ in range(15):
        cur = mul_table[cur][h]
        subgroup.append(cur)
    
    cosets = [set(subgroup)]
    for k in [1, 2]:
        coset = set()
        for s in subgroup:
            coset.add(mul_table[powers[k]][s])
        cosets.append(coset)
    
    if (cosets[0] | cosets[1] | cosets[2]) != set(nonzero):
        return None
    
    color_of_nonzero = {}
    for c_idx, coset in enumerate(cosets):
        for elem in coset:
            color_of_nonzero[elem] = c_idx
    
    neg_table = [0]*49
    for i in range(49):
        for j in range(49):
            if add_table[i][j] == zero_idx:
                neg_table[i] = j
                break
    
    N = 49
    colors = []
    for i in range(N):
        for j in range(i+1, N):
            if i == zero_idx or j == zero_idx:
                colors.append(3)
            else:
                diff = add_table[i][neg_table[j]]
                colors.append(color_of_nonzero[diff])
    
    return N, colors

def _seed32():
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    c = []
    for i in range(32):
        for j in range(i + 1, 32):
            c.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': c}

def construct():
    deadline = time.time() + 140.0
    result = _gf49_4coloring()
    if result is not None:
        n, colors = result
        if _verify(n, colors):
            return {'n': n, 'colors': colors}
    return _seed32()
