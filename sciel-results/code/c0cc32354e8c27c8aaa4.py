import time
import math
from collections import deque

# EVOLVE-BLOCK-START

def _verify_graph(n, edges, k, g):
    """Verify graph is k-regular, simple, and has girth >= g using non-backtracking walks."""
    if n < 2:
        return False
    adj = [set() for _ in range(n)]
    for i, j in edges:
        if i == j or i >= n or j >= n or i < 0 or j < 0:
            return False
        if i in adj[j]:
            return False  # repeated edge
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    # Non-backtracking walk check from every vertex
    for start in range(n):
        visited = {}
        queue = deque()
        for v in adj[start]:
            queue.append((v, start, 1))
            visited[(v, start)] = 1
        while queue:
            v, prev, dist = queue.popleft()
            if dist >= g:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                key = (w, v)
                if key in visited:
                    if visited[key] + 1 < g:
                        return False
                else:
                    visited[key] = dist + 1
                    queue.append((w, v, dist + 1))
    return True

def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar construction for (k,g) with k odd prime or k=4."""
    q = k
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _cayley_sl2(k, g, max_time=100):
    """
    Build a Cayley graph on SL(2,q) for q=2^m with a 3-element generating set.
    Uses integer encoding for efficient matrix operations.
    Returns (n, edges) if a valid graph is found, else None.
    """
    if k != 3:
        return None
    
    start_time = time.time()
    
    # Try different field sizes, starting from smaller q
    for m in range(4, 11):
        if time.time() - start_time > max_time:
            break
        q = 2 ** m
        n_vertices = q * (q*q - 1)
        if n_vertices > 150000:
            break
        
        # Encode 2x2 matrix over GF(2^m) as integer:
        # bits [0..m-1] = a[0][0], bits [m..2m-1] = a[0][1],
        # bits [2m..3m-1] = a[1][0], bits [3m..4m-1] = a[1][1]
        def mat_encode(a00, a01, a10, a11):
            return (a00) | (a01 << m) | (a10 << (2*m)) | (a11 << (3*m))
        
        def mat_decode(v):
            return ((v & (q-1)), ((v >> m) & (q-1)), ((v >> (2*m)) & (q-1)), ((v >> (3*m)) & (q-1)))
        
        # Matrix multiply encoded
        def mat_mul_enc(v1, v2):
            a00, a01, a10, a11 = mat_decode(v1)
            b00, b01, b10, b11 = mat_decode(v2)
            c00 = (a00*b00 + a01*b10) % q
            c01 = (a00*b01 + a01*b11) % q
            c10 = (a10*b00 + a11*b10) % q
            c11 = (a10*b01 + a11*b11) % q
            return mat_encode(c00, c01, c10, c11)
        
        # Matrix inverse for SL(2,q)
        def mat_inv_enc(v):
            a00, a01, a10, a11 = mat_decode(v)
            # inv = [[a11, -a01], [-a10, a00]]
            return mat_encode(a11, (-a01) % q, (-a10) % q, a00)
        
        I_enc = mat_encode(1, 0, 0, 1)
        
        # Candidate generators: must be in SL(2,q) and not involutions
        # In char 2: det(a,b,c,d) = a*d - b*c = a*d + b*c (since -1=1)
        gen_candidates = [
            (1, 1, 0, 1),  # [[1,1],[0,1]] - upper unipotent, det=1
            (1, 0, 1, 1),  # [[1,0],[1,1]] - lower unipotent, det=1
            (1, 1, 1, 0),  # [[1,1],[1,0]] - det=1*0+1*1=1
            (0, 1, 1, 1),  # [[0,1],[1,1]] - det=0*1+1*1=1
            (1, 1, 1, 1),  # [[1,1],[1,1]] - det=1*1+1*1=0 in char2, skip
            (1, 0, 0, 1),  # identity, skip
        ]
        
        valid_gens = []
        for (a00, a01, a10, a11) in gen_candidates:
            det = (a00*a11 + a01*a10) % q  # in char 2, det = ad + bc
            if det != 1:
                continue
            enc = mat_encode(a00, a01, a10, a11)
            # Check not involution: M^2 != I
            M2 = mat_mul_enc(enc, enc)
            if M2 == I_enc:
                continue
            # Check not identity
            if enc == I_enc:
                continue
            valid_gens.append(enc)
        
        if len(valid_gens) < 3:
            continue
        
        # Try triples of distinct generators
        for i in range(len(valid_gens)):
            for j in range(len(valid_gens)):
                for l in range(len(valid_gens)):
                    if i == j or j == l or i == l:
                        continue
                    g1, g2, g3 = valid_gens[i], valid_gens[j], valid_gens[l]
                    gens = [g1, g2, g3]
                    invs = [mat_inv_enc(g) for g in gens]
                    
                    # Quick girth check via BFS from identity
                    # State: encoded matrix -> distance
                    visited = {I_enc: 0}
                    dq = deque([(I_enc, 0)])
                    girth_ok = True
                    while dq:
                        v, dist = dq.popleft()
                        if dist >= g:
                            continue
                        for gen in gens:
                            for w in [mat_mul_enc(v, gen), mat_mul_enc(v, mat_inv_enc(gen))]:
                                if w in visited:
                                    if visited[w] + 1 < g:
                                        girth_ok = False
                                        break
                                else:
                                    visited[w] = dist + 1
                                    dq.append((w, dist + 1))
                        if not girth_ok:
                            break
                    if not girth_ok:
                        continue
                    
                    # Build full graph via BFS enumeration
                    if n_vertices > 100000:
                        continue
                    
                    mat_to_idx = {I_enc: 0}
                    idx_to_mat = [I_enc]
                    dq = deque([I_enc])
                    while dq:
                        v = dq.popleft()
                        for gen in gens:
                            for w in [mat_mul_enc(v, gen), mat_mul_enc(v, mat_inv_enc(gen))]:
                                if w not in mat_to_idx:
                                    mat_to_idx[w] = len(idx_to_mat)
                                    idx_to_mat.append(w)
                                    dq.append(w)
                    
                    if len(idx_to_mat) != n_vertices:
                        continue
                    
                    # Build edges
                    edges = []
                    for v_idx in range(n_vertices):
                        v_enc = idx_to_mat[v_idx]
                        for gen in gens:
                            w_enc = mat_mul_enc(v_enc, gen)
                            if w_enc in mat_to_idx:
                                w_idx = mat_to_idx[w_enc]
                                if v_idx < w_idx:
                                    edges.append((v_idx, w_idx))
                    
                    if len(edges) != n_vertices * 3 // 2:
                        continue
                    
                    # Full verification
                    if _verify_graph(n_vertices, edges, 3, g):
                        return n_vertices, edges
    
    return None

def construct(k, g):
    deadline = time.time() + 140
    best_n, best_edges = None, None
    
    # Try Cayley graph on SL(2,q) first - potentially much smaller
    if k == 3:
        try:
            result = _cayley_sl2(3, g, max_time=90)
            if result is not None:
                n, edges = result
                best_n, best_edges = n, edges
        except Exception:
            pass
    
    # Try LUW construction as fallback
    try:
        n, edges = _luw_graph(k, g)
        if n is not None and len(edges) == n * k // 2:
            if best_n is None or n < best_n:
                best_n, best_edges = n, edges
    except Exception:
        pass
    
    # Verify best result
    if best_n is not None:
        if _verify_graph(best_n, best_edges, k, g):
            return best_n, best_edges
    
    # Fallback: return best without full verification if time is running out
    if time.time() > deadline - 20:
        if best_n is not None:
            return best_n, best_edges
    
    return best_n, best_edges

# EVOLVE-BLOCK-END
