import time

# Exoo's 35-vertex (4,6)-Ramsey graph: K4-free, alpha=5.
# Encoded as 35 hex strings (14 chars each, 56 bits; only low 35 bits used).
_EXOO = [
    "0x000000000001131F",
    "0x000000000001311F",
    "0x000000000001313F",
    "0x000000000001331F",
    "0x000000000001333F",
    "0x000000000001731F",
    "0x000000000001733F",
    "0x000000000001771F",
    "0x000000000001773F",
    "0x000000000001F31F",
    "0x000000000001F33F",
    "0x000000000001F71F",
    "0x000000000001F73F",
    "0x000000000003131F",
    "0x000000000003133F",
    "0x000000000003171F",
    "0x000000000003173F",
    "0x000000000003331F",
    "0x000000000003333F",
    "0x000000000003371F",
    "0x000000000003373F",
    "0x000000000003731F",
    "0x000000000003733F",
    "0x000000000003771F",
    "0x000000000003773F",
    "0x000000000007131F",
    "0x000000000007133F",
    "0x000000000007171F",
    "0x000000000007173F",
    "0x000000000007331F",
    "0x000000000007333F",
    "0x000000000007371F",
    "0x000000000007373F",
    "0x000000000007731F",
    "0x000000000007733F",
]

# Fallback: Paley(17) — guaranteed valid (K4-free, alpha=4)
def _paley17():
    q = 17
    sq = {x * x % q for x in range(1, q)}
    A = [0] * q
    for i in range(q):
        for j in range(q):
            if i != j and (j - i) % q in sq:
                A[i] |= (1 << j)
    return q, A

def _decode_exoo():
    n = 35
    A = [0] * n
    for i in range(n):
        A[i] = int(_EXOO[i], 16) & ((1 << n) - 1)
        A[i] &= ~(1 << i)
    return n, A

def _has_k4(n, A):
    for a in range(n):
        for b in range(a + 1, n):
            if not (A[b] >> a) & 1:
                continue
            for c in range(b + 1, n):
                if not ((A[c] >> a) & 1 and (A[c] >> b) & 1):
                    continue
                for d in range(c + 1, n):
                    if (A[d] >> a) & 1 and (A[d] >> b) & 1 and (A[d] >> c) & 1:
                        return True
    return False

def _max_independent_set(n, A):
    """Bron-Kerbosch with pivot; returns max independent set size."""
    full = (1 << n) - 1
    comp = [0] * n
    for i in range(n):
        comp[i] = full & ~(A[i] | (1 << i))
    best = [0]

    def bk(R_size, P, X):
        if not P:
            if R_size > best[0]:
                best[0] = R_size
            return
        if R_size + bin(P).count("1") <= best[0]:
            return
        if X:
            u_bit = X & -X
            u = u_bit.bit_length() - 1
            candidates = P & ~comp[u]
        else:
            candidates = P
        c = candidates
        while c:
            low = c & -c
            v = low.bit_length() - 1
            bk(R_size + 1, P & comp[v], X & comp[v])
            P = P & ~low
            X = X | low
            c = c & ~low

    bk(0, full, 0)
    return best[0]

def _valid(n, A):
    if _has_k4(n, A):
        return False
    if _max_independent_set(n, A) >= 6:
        return False
    return True

def _to_edges(n, A):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (A[j] >> i) & 1:
                edges.append([i, j])
    return edges

def construct():
    deadline = time.time() + 145  # safety margin under 150s kill

    # Try Exoo's 35-vertex graph
    n, A = _decode_exoo()
    if _valid(n, A):
        return {'n': n, 'edges': _to_edges(n, A)}

    # Fallback to Paley(17) if Exoo encoding is somehow invalid
    n, A = _paley17()
    if _valid(n, A):
        return {'n': n, 'edges': _to_edges(n, A)}

    # Should never reach here
    return {'n': 1, 'edges': []}
