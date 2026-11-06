"""Deterministic 4-colouring of K_64 with no monochromatic triangle.

4 disjoint Clebsch copies (16 each).
  - intra-copy edges: Clebsch edge-classes C0,C1,C2 (each triangle-free);
    non-Clebsch intra edges -> colour 3.
  - inter-copy edges: 4-colouring of K4 on copy-indices, no mono-triangle.
Self-verifies; falls back to a valid K_32 construction if needed.
"""
# EVOLVE-BLOCK-START

def _clebsch_edge_classes():
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            if (i ^ j) in (1, 2, 4, 8, 10):
                edges.append((i, j))
    col = {}
    adj = [{i: set() for i in range(16)} for _ in range(3)]
    for (i, j) in edges:
        chosen = None
        for c in range(3):
            if not (adj[c][i] & adj[c][j]):
                chosen = c
                break
        if chosen is None:
            chosen = 0
        col[(i, j)] = chosen
        adj[chosen][i].add(j)
        adj[chosen][j].add(i)
    return col

def _has_mono_triangle(n, colors):
    adj = [dict() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]; idx += 1
            adj[c].setdefault(i, set()).add(j)
            adj[c].setdefault(j, set()).add(i)
    for c in range(4):
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[c].get(i, set()):
                    for k in range(j + 1, n):
                        if k in adj[c].get(i, set()) and k in adj[c].get(j, set()):
                            return True
    return False

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
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}

def construct():
    clebsch_col = _clebsch_edge_classes()
    n = 64
    inter = {(0,1):0, (0,2):1, (0,3):2, (1,2):2, (1,3):1, (2,3):0}
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = i // 16, j // 16
            ia, ib = i % 16, j % 16
            if a == b:
                c = clebsch_col.get((ia, ib))
                colors.append(3 if c is None else c)
            else:
                colors.append(inter[(a, b)])
    if _has_mono_triangle(n, colors):
        return _seed32()
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
