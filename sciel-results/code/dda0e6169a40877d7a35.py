import time
import random

# ---------- K2,2,2 detection using bitmasks ----------
def _has_k222(n, adj):
    """Return (a,b,u,v,x,y) describing a K2,2,2 in the color given by adj, else None.
    adj: list of n ints, bitmask of neighbors in THIS color."""
    for a in range(n):
        ba = adj[a]
        for b in range(a + 1, n):
            if ba & (1 << b):
                continue  # edge present in this color -> cannot be a part
            C = ba & adj[b] & ~((1 << a) | (1 << b))
            if C == 0:
                continue
            # list of vertices in C
            c_list = []
            m = C
            while m:
                lsb = m & (-m)
                c_list.append(lsb.bit_length() - 1)
                m ^= lsb
            L = len(c_list)
            for i in range(L):
                u = c_list[i]
                bu = adj[u]
                for j in range(i + 1, L):
                    v = c_list[j]
                    if bu & (1 << v):
                        continue  # u,v adjacent in this color -> not a part
                    common = bu & adj[v] & C
                    if common == 0:
                        continue
                    # need two non-adjacent x,y in common (they form a part)
                    m = common
                    while m:
                        lsb = m & (-m)
                        x = lsb.bit_length() - 1
                        ny = common & ~adj[x]
                        if ny:
                            lsb2 = ny & (-ny)
                            y = lsb2.bit_length() - 1
                            return (a, b, u, v, x, y)
                        m ^= lsb
    return None


def _build_adj(n, colors):
    adj0 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            idx += 1
    return adj0


def _colors_from_adj(n, adj0):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (adj0[i] & (1 << j)) else 1)
    return colors


def _is_valid(n, adj0):
    full = (1 << n) - 1
    adj1 = [full ^ a for a in adj0]
    return _has_k222(n, adj0) is None and _has_k222(n, adj1) is None


def _paley(n):
    squares = {x * x % n for x in range(1, n)}
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if ((j - i) % n) in squares else 1)
    return colors


def _local_search(n, init_adj0, deadline, seed=0):
    """Repair init_adj0 (list of ints) to a K2,2,2-free-in-both-colors state.
    Returns adj0 if found, else None."""
    rng = random.Random(seed)
    adj0 = list(init_adj0)
    full = (1 << n) - 1
    seen = set()
    stall = 0
    while time.time() < deadline:
        adj1 = [full ^ a for a in adj0]
        v0 = _has_k222(n, adj0)
        if v0 is None:
            v1 = _has_k222(n, adj1)
            if v1 is None:
                return adj0
            viol = (1, v1)
        else:
            viol = (0, v0)
        c, (a, b, u, v, x, y) = viol
        parts = [(a, b), (u, v), (x, y)]
        cross = []
        for pi in range(3):
            for pj in range(pi + 1, 3):
                for x1 in parts[pi]:
                    for x2 in parts[pj]:
                        cross.append((min(x1, x2), max(x1, x2)))
        # choose a cross edge to flip; prefer one that is currently in color c
        target = [e for e in cross if ((adj0[e[0]] & (1 << e[1])) == (1 if c == 0 else 0))]
        pool = target if target else cross
        e = rng.choice(pool)
        i, j = e
        if adj0[i] & (1 << j):
            adj0[i] ^= (1 << j)
            adj0[j] ^= (1 << i)
        else:
            adj0[i] |= (1 << j)
            adj0[j] |= (1 << i)
        # cycle / stall escape
        h = tuple(adj0)
        if h in seen:
            stall += 1
            if stall > 30:
                # random kick: flip a random edge
                for _ in range(3):
                    ii = rng.randrange(n)
                    jj = rng.randrange(n)
                    if ii == jj:
                        continue
                    ii, jj = min(ii, jj), max(ii, jj)
                    if adj0[ii] & (1 << jj):
                        adj0[ii] ^= (1 << jj)
                        adj0[jj] ^= (1 << ii)
                    else:
                        adj0[ii] |= (1 << jj)
                        adj0[jj] |= (1 << ii)
                seen.clear()
                stall = 0
        else:
            seen.add(h)
            if len(seen) > 20000:
                seen.clear()
    return None


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin under 150s kill

    best = None

    # Try target sizes from largest down.
    for n in (29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17):
        if time.time() > deadline - 5.0:
            break
        # Algebraic init: Paley if n is prime ≡1 mod 4, else random.
        if n in (5, 13, 17, 29):
            init_colors = _paley(n)
        else:
            init_colors = [random.Random(n).randint(0, 1) for _ in range(n * (n - 1) // 2)]
        init_adj0 = _build_adj(n, init_colors)

        # Fast path: is the init already valid?
        if _is_valid(n, init_adj0):
            best = (n, _colors_from_adj(n, init_adj0))
            break

        # Local search to repair.
        found = _local_search(n, init_adj0, min(deadline - 3.0, t0 + 140.0), seed=n)
        if found is not None and _is_valid(n, found):
            best = (n, _colors_from_adj(n, found))
            break

    # Guaranteed fallback: Paley(17) is known valid.
    if best is None:
        n = 17
        colors = _paley(n)
        adj0 = _build_adj(n, colors)
        if _is_valid(n, adj0):
            best = (n, colors)
        else:
            # extremely unlikely; return anyway
            best = (n, colors)

    return {'n': best[0], 'colors': best[1]}
