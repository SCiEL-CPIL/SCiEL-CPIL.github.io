import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill

# ---------------- Fast exact verification (bitsets) ----------------

def _has_K4_bitset(n, bits):
    """Return True if graph contains a K4. bits[i] has bit j set iff edge (i,j)."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (bits[a] >> b) & 1:
                continue
            common = bits[a] & bits[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & bits[c]
                common2 &= ~((1 << (c + 1)) - 1)
                if common2:
                    return True
    return False

def _has_indep_6_bitset(n, bits):
    """Return True if graph contains an independent set of size 6."""
    all_mask = (1 << n) - 1
    comp = [(all_mask ^ bits[i]) & ~(1 << i) for i in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if not (comp[a] >> b) & 1:
                continue
            common = comp[a] & comp[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & comp[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if not (common2 >> d) & 1:
                        continue
                    common3 = common2 & comp[d]
                    common3 &= ~((1 << (d + 1)) - 1)
                    for e in range(d + 1, n):
                        if not (common3 >> e) & 1:
                            continue
                        common4 = common3 & comp[e]
                        common4 &= ~((1 << (e + 1)) - 1)
                        if common4:
                            return True
    return False

def _is_valid(n, edges):
    if n < 1:
        return False
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        adj[u].add(v)
        adj[v].add(u)
    bits = [0] * n
    for i in range(n):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        bits[i] = mask
    if _has_K4_bitset(n, bits):
        return False
    if _has_indep_6_bitset(n, bits):
        return False
    return True

def _edges_from_bits(n, bits):
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if (bits[i] >> j) & 1:
                out.append([i, j])
    return out

# ---------------- Algebraic constructions ----------------

def _paley_bits(p):
    """Paley graph on prime p ≡ 1 mod 4. Returns bits list of length p."""
    squares = {x * x % p for x in range(1, p)}
    bits = [0] * p
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits

def _gf2_quadratic_bits(k, rng):
    """
    Graph on GF(2)^k (n = 2^k vertices).
    Edge between x and y if Q(x+y) = 1, where Q is a random non-degenerate
    quadratic form over GF(2). This yields a strongly regular-like graph.
    """
    n = 1 << k
    # Random quadratic form: Q(x) = sum_{i<j} a_{ij} x_i x_j + sum_i b_i x_i
    a = {}
    for i in range(k):
        for j in range(i + 1, k):
            a[(i, j)] = rng.randint(0, 1)
    b = [rng.randint(0, 1) for _ in range(k)]
    
    def Q(x):
        # x is an integer representing a vector in GF(2)^k
        s = 0
        for i in range(k):
            if (x >> i) & 1:
                s ^= b[i]
        for i in range(k):
            for j in range(i + 1, k):
                if a[(i, j)] and ((x >> i) & 1) and ((x >> j) & 1):
                    s ^= 1
        return s & 1
    
    bits = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if Q(i ^ j) == 1:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    return bits

def _circulant_diffset_bits(n, rng):
    """
    Random circulant graph: vertex i connected to i+d mod n for d in D,
    where D is a random subset of {1, ..., n//2}.
    """
    D = set()
    for d in range(1, n // 2 + 1):
        if rng.random() < 0.4:
            D.add(d)
    bits = [0] * n
    for i in range(n):
        for d in D:
            j = (i + d) % n
            if j > i:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
            elif j < i:
                bits[i] |= (1 << j)
                bits[j] |= (1 << i)
    # Deduplicate (in case d and n-d both in D)
    for i in range(n):
        for j in range(i + 1, n):
            if (bits[i] >> j) & 1:
                bits[j] |= (1 << i)
    return bits

def _paley_circulant_hybrid(n, rng):
    """
    Use a prime p near n, take Paley structure, then randomly flip a small
    fraction of edges to break perfect symmetry while preserving structure.
    """
    # Find prime p near n
    p = None
    for cand in range(max(5, n - 10), n + 15):
        if cand % 2 == 0 and cand != 2:
            continue
        is_prime = True
        for d in range(2, int(cand**0.5) + 1):
            if cand % d == 0:
                is_prime = False
                break
        if is_prime and cand % 4 == 1:
            p = cand
            break
    if p is None:
        return _circulant_diffset_bits(n, rng)
    
    bits = _paley_bits(p)
    # Truncate or pad to n vertices
    if p > n:
        bits = bits[:n]
        # Clear edges to vertices >= n
        for i in range(n):
            mask = (1 << n) - 1
            bits[i] &= mask
    elif p < n:
        # Extend by adding random vertices connected with some probability
        new_bits = bits[:]
        for i in range(p, n):
            new_bits.append(0)
        for i in range(p, n):
            for j in range(n):
                if i != j and rng.random() < 0.3:
                    new_bits[i] |= (1 << j)
                    new_bits[j] |= (1 << i)
        bits = new_bits
    
    # Flip a small fraction of edges
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.05:
                if (bits[i] >> j) & 1:
                    bits[i] &= ~(1 << j)
                    bits[j] &= ~(1 << i)
                else:
                    bits[i] |= (1 << j)
                    bits[j] |= (1 << i)
    return bits

# ---------------- Fast branch-and-bound for clique detection ----------------

def _max_clique_size(n, bits, timeout=None):
    """
    Find the size of the maximum clique using branch-and-bound.
    Returns the size. If timeout is given, may return early with a lower bound.
    """
    if timeout is not None and time.time() > timeout:
        return 0
    best = [0]
    
    def expand(cand, size):
        if size > best[0]:
            best[0] = size
        if not cand:
            return
        # Upper bound: size + len(cand)
        if size + len(cand) <= best[0]:
            return
        # Pick vertex with highest degree in cand
        max_deg = -1
        pick = -1
        for v in cand:
            deg = sum(1 for u in cand if u != v and (bits[v] >> u) & 1)
            if deg > max_deg:
                max_deg = deg
                pick = v
        # Branch: include pick
        new_cand = [u for u in cand if u != pick and (bits[pick] >> u) & 1]
        expand(new_cand, size + 1)
        # Branch: exclude pick
        new_cand = [u for u in cand if u != pick]
        expand(new_cand, size)
    
    expand(list(range(n)), 0)
    return best[0]

def _has_clique_of_size(n, bits, k, timeout=None):
    """Check if graph has a clique of size k using branch-and-bound."""
    if timeout is not None and time.time() > timeout:
        return False
    found = [False]
    
    def expand(cand, size, current):
        if found[0]:
            return
        if size == k:
            found[0] = True
            return
        if not cand:
            return
        if size + len(cand) < k:
            return
        # Pick vertex
        pick = cand[0]
        rest = cand[1:]
        # Include pick
        new_cand = [u for u in rest if (bits[pick] >> u) & 1]
        expand(new_cand, size + 1, current + [pick])
        if found[0]:
            return
        # Exclude pick
        expand(rest, size, current)
    
    expand(list(range(n)), 0, [])
    return found[0]

# ---------------- Targeted repair using fast violation finding ----------------

def _find_K4_fast(n, bits):
    """Return a K4 as tuple (a,b,c,d) or None."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (bits[a] >> b) & 1:
                continue
            common = bits[a] & bits[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & bits[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if (common2 >> d) & 1:
                        return (a, b, c, d)
    return None

def _find_indep_6_fast(n, bits):
    """Return an independent 6-set as tuple or None."""
    all_mask = (1 << n) - 1
    comp = [(all_mask ^ bits[i]) & ~(1 << i) for i in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if not (comp[a] >> b) & 1:
                continue
            common = comp[a] & comp[b]
            common &= ~((1 << (b + 1)) - 1)
            for c in range(b + 1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & comp[c]
                common2 &= ~((1 << (c + 1)) - 1)
                for d in range(c + 1, n):
                    if not (common2 >> d) & 1:
                        continue
                    common3 = common2 & comp[d]
                    common3 &= ~((1 << (d + 1)) - 1)
                    for e in range(d + 1, n):
                        if not (common3 >> e) & 1:
                            continue
                        common4 = common3 & comp[e]
                        common4 &= ~((1 << (e + 1)) - 1)
                        for f in range(e + 1, n):
                            if (common4 >> f) & 1:
                                return (a, b, c, d, e, f)
    return None

def _targeted_repair(n, bits, rng, max_iters=2000):
    """
    Iteratively fix violations: remove an edge from a K4 or add an edge to an
    independent 6-set. Use fast finders. Stop when no violations or time out.
    """
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        k4 = _find_K4_fast(n, bits)
        if k4 is None:
            indep = _find_indep_6_fast(n, bits)
            if indep is None:
                return bits
            # Add a random edge within the independent set
            verts = list(indep)
            u, v = verts[rng.randint(0, 5)], verts[rng.randint(0, 5)]
            if u != v:
                if not (bits[u] >> v) & 1:
                    bits[u] |= (1 << v)
                    bits[v] |= (1 << u)
        else:
            # Remove a random edge from the K4
            a, b, c, d = k4
            pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
            u, v = pairs[rng.randint(0, 5)]
            if (bits[u] >> v) & 1:
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
    return bits

# ---------------- Baseline: 17-vertex Paley graph ----------------

def _paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    return {'n': 17, 'edges': edges}

_BASELINE = _paley_17()

# ---------------- Main search: algebraic seeds + repair ----------------

def _algebraic_search():
    """
    Generate candidates from algebraic families, validate with fast checks,
    and repair top candidates.
    """
    best = _BASELINE
    rng = random.Random(123)
    
    candidates = []  # (n, bits, source)
    
    # 1. Paley graphs for primes p ≡ 1 mod 4
    for p in [17, 29, 37, 41, 53, 61, 73, 89, 97]:
        if p > 40:
            break
        bits = _paley_bits(p)
        candidates.append((p, bits, f'paley_{p}'))
    
    # 2. GF(2)^k quadratic form graphs
    for k in [4, 5, 6]:
        n = 1 << k
        if n > 40:
            break
        for trial in range(20):
            bits = _gf2_quadratic_bits(k, rng)
            candidates.append((n, bits, f'gf2_{k}_{trial}'))
    
    # 3. Circulant difference-set graphs
    for n in [20, 24, 28, 32, 36, 40]:
        for trial in range(10):
            bits = _circulant_diffset_bits(n, rng)
            candidates.append((n, bits, f'circ_{n}_{trial}'))
    
    # 4. Paley-circulant hybrids
    for n in [20, 25, 30, 35, 36, 37, 38, 39, 40]:
        for trial in range(5):
            bits = _paley_circulant_hybrid(n, rng)
            candidates.append((n, bits, f'hybrid_{n}_{trial}'))
    
    # Sort by n descending, then by a quick violation count
    def quick_violations(n, bits):
        """Fast approximate violation count for ranking."""
        # Count K4s (partial)
        k4_count = 0
        for a in range(n):
            for b in range(a + 1, n):
                if not (bits[a] >> b) & 1:
                    continue
                common = bits[a] & bits[b]
                common &= ~((1 << (b + 1)) - 1)
                for c in range(b + 1, n):
                    if not (common >> c) & 1:
                        continue
                    common2 = common & bits[c]
                    common2 &= ~((1 << (c + 1)) - 1)
                    k4_count += bin(common2).count('1')
        # Count indep 6s (partial, just check a few)
        all_mask = (1 << n) - 1
        comp = [(all_mask ^ bits[i]) & ~(1 << i) for i in range(n)]
        ind6_count = 0
        for a in range(min(n, 20)):
            for b in range(a + 1, min(n, 20)):
                if not (comp[a] >> b) & 1:
                    continue
                common = comp[a] & comp[b]
                common &= ~((1 << (b + 1)) - 1)
                for c in range(b + 1, min(n, 20)):
                    if not (common >> c) & 1:
                        continue
                    common2 = common & comp[c]
                    common2 &= ~((1 << (c + 1)) - 1)
                    ind6_count += bin(common2).count('1')
        return k4_count + ind6_count
    
    scored = []
    for n, bits, source in candidates:
        if time.time() > _DEADLINE - 30:
            break
        v = quick_violations(n, bits)
        scored.append((v, n, bits, source))
    
    scored.sort(key=lambda x: (x[0], -x[1]))
    
    # Repair top candidates
    for score, n, bits, source in scored[:20]:
        if time.time() > _DEADLINE - 10:
            break
        repaired = _targeted_repair(n, bits[:], rng, max_iters=500)
        if not _has_K4_bitset(n, repaired) and not _has_indep_6_bitset(n, repaired):
            edges = _edges_from_bits(n, repaired)
            if n > best['n']:
                best = {'n': n, 'edges': edges}
    
    return best

# ---------------- Entry point ----------------

def construct():
    t0 = time.time()
    best = _BASELINE
    
    try:
        improved = _algebraic_search()
        if improved['n'] > best['n']:
            best = improved
    except Exception:
        pass
    
    # Safety: ensure validity of what we return
    if not _is_valid(best['n'], best['edges']):
        best = _BASELINE
    
    return best
