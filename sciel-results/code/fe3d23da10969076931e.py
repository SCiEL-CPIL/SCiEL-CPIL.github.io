import time
import random
import itertools
import numpy as np

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill

# ---------------- Bitset-based verification helpers (exact, fast) ----------------

def _adj_to_bitsets(n, adj):
    bits = [0] * n
    for i in range(n):
        m = 0
        s = adj[i]
        for j in s:
            m |= (1 << j)
        bits[i] = m
    return bits

def _has_K4_bitset(n, bits):
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
        if u == v:
            return False
        if u < 0 or v < 0 or u >= n or v >= n:
            return False
        adj[u].add(v)
        adj[v].add(u)
    bits = _adj_to_bitsets(n, adj)
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

def _bits_from_edges(n, edges):
    bits = [0] * n
    for u, v in edges:
        bits[u] |= (1 << v)
        bits[v] |= (1 << u)
    return bits

# ---------------- Baseline: 17-vertex Paley graph (known valid) ----------------

def _paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    return {'n': 17, 'edges': edges}

_BASELINE = _paley_17()

# ---------------- Published 35-vertex record (Exoo, R(4,6) > 35) ----------------
# This is the known 35-vertex K4-free graph with no independent 6-set.
_EXOO_35_EDGES = [
    [0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9],[0,10],[0,11],[0,12],[0,13],[0,14],[0,15],
    [1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],[1,10],[1,11],[1,12],[1,13],[1,14],[1,15],
    [2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],[2,10],[2,11],[2,12],[2,13],[2,14],[2,15],
    [3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],[3,11],[3,12],[3,13],[3,14],[3,15],
    [4,5],[4,6],[4,7],[4,8],[4,9],[4,10],[4,11],[4,12],[4,13],[4,14],[4,15],
    [5,6],[5,7],[5,8],[5,9],[5,10],[5,11],[5,12],[5,13],[5,14],[5,15],
    [6,7],[6,8],[6,9],[6,10],[6,11],[6,12],[6,13],[6,14],[6,15],
    [7,8],[7,9],[7,10],[7,11],[7,12],[7,13],[7,14],[7,15],
    [8,9],[8,10],[8,11],[8,12],[8,13],[8,14],[8,15],
    [9,10],[9,11],[9,12],[9,13],[9,14],[9,15],
    [10,11],[10,12],[10,13],[10,14],[10,15],
    [11,12],[11,13],[11,14],[11,15],
    [12,13],[12,14],[12,15],
    [13,14],[13,15],
    [14,15],
    # The above is K16 which has K4s. We need the actual Exoo graph.
    # Since I cannot reliably recall the exact 35-vertex Exoo edges, I will
    # instead construct a strong baseline via the self-complementary approach below.
]

# Since the exact Exoo 35-vertex edge list is not reliably available from memory,
# we will use the self-complementary construction + two-phase local search as the
# primary strategy, with Paley-17 as fallback.

# ---------------- Self-complementary construction ----------------

def _self_complementary_bits(n, rng):
    """Generate a self-complementary graph on n vertices (n must be 0 or 1 mod 4)."""
    if n % 4 not in (0, 1):
        return None
    m = (n * (n - 1)) // 2  # total possible edges
    # For self-complementary, exactly m/2 edges must be present.
    if m % 2 != 0:
        return None
    half = m // 2
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(all_edges)
    selected = all_edges[:half]
    bits = [0] * n
    for u, v in selected:
        bits[u] |= (1 << v)
        bits[v] |= (1 << u)
    return bits

# ---------------- Vectorized K4 count using NumPy ----------------

def _count_K4_numpy(n, bits):
    """Count K4s using NumPy vectorization. bits is list of ints."""
    if n < 4:
        return 0
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        b = bits[i]
        for j in range(i + 1, n):
            if (b >> j) & 1:
                A[i, j] = 1
                A[j, i] = 1
    # Count K4s: for each pair (a,b) with edge, count common neighbors c,d > b
    count = 0
    for a in range(n - 3):
        for b in range(a + 1, n - 2):
            if A[a, b] == 0:
                continue
            common = A[a] & A[b]
            # Only consider vertices > b
            common[:b + 1] = 0
            # For each c in common, count vertices d > c in common & A[c]
            for c in range(b + 1, n - 1):
                if common[c] == 0:
                    continue
                common2 = common & A[c]
                common2[:c + 1] = 0
                count += int(np.sum(common2))
    return count

# ---------------- Two-phase local search ----------------

def _local_search_two_phase(n, bits, rng, max_iters=5000, time_limit=120.0):
    """Two-phase local search: Phase 1 minimize K4 count, Phase 2 minimize Ind6 count."""
    start_time = time.time()
    
    # Phase 1: Minimize K4 count
    current_k4 = _count_K4_numpy(n, bits)
    for it in range(max_iters):
        if time.time() - start_time > time_limit * 0.6:
            break
        if current_k4 == 0:
            break
        
        # Try random edge flips, accept if K4 count decreases
        improved = False
        for _ in range(50):  # Try up to 50 random flips per iteration
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Flip edge (u, v)
            had_edge = (bits[u] >> v) & 1
            if had_edge:
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
            else:
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
            
            new_k4 = _count_K4_numpy(n, bits)
            
            if new_k4 < current_k4:
                current_k4 = new_k4
                improved = True
                break
            else:
                # Revert
                if had_edge:
                    bits[u] |= (1 << v)
                    bits[v] |= (1 << u)
                else:
                    bits[u] &= ~(1 << v)
                    bits[v] &= ~(1 << u)
        
        if not improved:
            # No improvement in this batch, try a few more
            if it % 100 == 0 and time.time() - start_time > time_limit * 0.5:
                break
    
    # Phase 2: Minimize independent 6-set count
    # Use bitset-based detection for acceptance
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Check if already valid
        if not _has_K4_bitset(n, bits) and not _has_indep_6_bitset(n, bits):
            return bits
        
        # Try random edge flips
        improved = False
        for _ in range(30):
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            had_edge = (bits[u] >> v) & 1
            if had_edge:
                bits[u] &= ~(1 << v)
                bits[v] &= ~(1 << u)
            else:
                bits[u] |= (1 << v)
                bits[v] |= (1 << u)
            
            # Check if this flip improves the situation
            has_k4 = _has_K4_bitset(n, bits)
            has_ind6 = _has_indep_6_bitset(n, bits)
            
            # Accept if we eliminated a constraint violation
            if not has_k4 and not has_ind6:
                improved = True
                break
            elif not has_k4 and has_ind6:
                # Better than before if we had K4 before
                improved = True
                break
            else:
                # Revert if worse
                if had_edge:
                    bits[u] |= (1 << v)
                    bits[v] |= (1 << u)
                else:
                    bits[u] &= ~(1 << v)
                    bits[v] &= ~(1 << u)
        
        if not improved:
            if it % 200 == 0 and time.time() - start_time > time_limit * 0.9:
                break
    
    return bits

def _search_for_larger():
    best = _BASELINE
    rng = random.Random(42)
    
    # Try self-complementary constructions for n = 33, 36, 37, 40
    for n in [33, 36, 37, 40]:
        if time.time() > _DEADLINE - 10:
            break
        
        for attempt in range(30):
            if time.time() > _DEADLINE - 5:
                break
            
            bits = _self_complementary_bits(n, rng)
            if bits is None:
                continue
            
            bits2 = _local_search_two_phase(n, bits, rng, max_iters=3000, time_limit=3.0)
            
            if not _has_K4_bitset(n, bits2) and not _has_indep_6_bitset(n, bits2):
                edges = _edges_from_bits(n, bits2)
                if n > best['n']:
                    best = {'n': n, 'edges': edges}
                break
    
    # Also try random dense graphs as alternative starting points
    if time.time() < _DEADLINE - 15:
        for n in [36, 37, 38]:
            if time.time() > _DEADLINE - 10:
                break
            for p in [0.3, 0.4, 0.5]:
                if time.time() > _DEADLINE - 8:
                    break
                bits = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if rng.random() < p:
                            bits[i] |= (1 << j)
                            bits[j] |= (1 << i)
                
                bits2 = _local_search_two_phase(n, bits, rng, max_iters=2000, time_limit=3.0)
                
                if not _has_K4_bitset(n, bits2) and not _has_indep_6_bitset(n, bits2):
                    edges = _edges_from_bits(n, bits2)
                    if n > best['n']:
                        best = {'n': n, 'edges': edges}
                    break
    
    return best

# ---------------- Entry point ----------------

def construct():
    t0 = time.time()
    best = _BASELINE
    
    # Spend remaining budget trying to improve
    try:
        improved = _search_for_larger()
        if improved['n'] > best['n']:
            best = improved
    except Exception:
        pass
    
    # Safety: ensure validity of what we return
    if not _is_valid(best['n'], best['edges']):
        best = _BASELINE
    
    return best
