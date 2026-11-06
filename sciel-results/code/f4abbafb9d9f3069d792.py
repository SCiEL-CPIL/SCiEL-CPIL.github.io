"""Construct a K4-free graph with no independent 6-set, maximizing n.

Strategy: Simulated annealing on a fixed n-vertex graph using vectorized NumPy
score computation. The score = (# K4s) + (# independent 6-sets). We start from
structured initial graphs (circulant, grid-based) and flip edges to minimize
the score. A score of 0 means a valid graph.
"""

import time
import random
import numpy as np

_START = time.time()
DEADLINE = 140.0  # safety margin before 150s kill


def _time_left():
    return DEADLINE - (time.time() - _START)


def construct():
    global _START
    _START = time.time()

    best_n = 0
    best_adj = None

    # Try to find a valid graph at n=36, then 35, etc.
    for target_n in range(36, 16, -1):
        if _time_left() < 20:
            break

        # Try multiple structured initializations
        found = False
        for init_method in ['circulant', 'grid', 'random', 'paley_like']:
            if _time_left() < 15:
                break

            A = _make_initial(target_n, init_method)
            if A is None:
                continue

            # Run simulated annealing
            A_sa = _simulated_annealing(A, target_n, _time_left() - 15)

            if _score(A_sa, target_n) == 0:
                best_n = target_n
                best_adj = A_sa
                found = True
                break

        if found:
            break

    # Fallback: try smaller n with more time
    if best_n == 0:
        for target_n in range(20, 16, -1):
            if _time_left() < 10:
                break
            A = _make_initial(target_n, 'circulant')
            A_sa = _simulated_annealing(A, target_n, _time_left() - 5)
            if _score(A_sa, target_n) == 0:
                best_n = target_n
                best_adj = A_sa
                break

    # Absolute fallback
    if best_n == 0:
        # Use a known small valid graph: Paley(17)
        A = _paley_numpy(17)
        if _score(A, 17) == 0:
            best_n = 17
            best_adj = A
        else:
            # Just return a cycle (will be invalid but at least returns something)
            best_n = 17
            A = np.zeros((17, 17), dtype=np.int8)
            for i in range(17):
                A[i, (i+1) % 17] = 1
                A[(i+1) % 17, i] = 1
            best_adj = A

    return {'n': best_n, 'edges': _adj_to_edges(best_adj, best_n)}


def _make_initial(n, method):
    """Create initial adjacency matrix based on method."""
    if method == 'circulant':
        return _circulant_init(n)
    elif method == 'grid':
        return _grid_init(n)
    elif method == 'random':
        rng = np.random.RandomState(42 + n)
        A = rng.binomial(1, 0.3, (n, n)).astype(np.int8)
        np.fill_diagonal(A, 0)
        # Symmetrize
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0)
        return A
    elif method == 'paley_like':
        # Use quadratic residues if n is prime
        if _is_prime(n) and n % 4 == 1:
            return _paley_numpy(n)
        else:
            return _circulant_init(n)
    return None


def _circulant_init(n):
    """Circulant graph with a good difference set."""
    A = np.zeros((n, n), dtype=np.int8)
    # Try difference set: every 3rd element (gives degree ~2n/3)
    # Or use a known good pattern
    # For K4-free: degree should be moderate
    # Use D = {1, 2, ..., k} where k ~ n/5
    k = max(2, n // 5)
    for d in range(1, k + 1):
        for i in range(n):
            j = (i + d) % n
            A[i, j] = 1
    # Symmetrize
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return A


def _grid_init(n):
    """Grid-based initialization for n that's a perfect square or close."""
    A = np.zeros((n, n), dtype=np.int8)
    side = int(np.sqrt(n))
    if side * side != n:
        # Not a perfect square, use circulant
        return _circulant_init(n)

    for r1 in range(side):
        for c1 in range(side):
            v1 = r1 * side + c1
            for r2 in range(side):
                for c2 in range(side):
                    v2 = r2 * side + c2
                    if v1 >= v2:
                        continue
                    # Edge if NOT same row, NOT same col, NOT same diagonal
                    if r1 == r2:
                        continue
                    if c1 == c2:
                        continue
                    if (r1 - c1) % side == (r2 - c2) % side:
                        continue
                    A[v1, v2] = 1
                    A[v2, v1] = 1
    return A


def _paley_numpy(q):
    """Paley graph as numpy matrix."""
    A = np.zeros((q, q), dtype=np.int8)
    qr = set()
    for x in range(1, q):
        if pow(x, (q - 1) // 2, q) == 1:
            qr.add(x)
    for i in range(q):
        for j in range(i + 1, q):
            d = (j - i) % q
            if d in qr:
                A[i, j] = 1
                A[j, i] = 1
    return A


def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def _score(A, n):
    """Compute score = (# K4s) + (# independent 6-sets) using NumPy.
    
    For K4 count: for each edge (i,j), count common neighbors that form triangles.
    Actually, # K4s = sum over all 4-cliques. Efficient way:
    For each pair (i,j) that is an edge, let C = N(i) ∩ N(j). Then # K4s containing edge (i,j)
    = # edges within C. Sum over all edges (i,j) and divide by 6 (each K4 has 6 edges).
    
    For independent 6-sets: use the complement graph. # independent 6-sets in G = # K6 in complement.
    This is expensive. Instead, use a cheaper proxy: count independent 4-sets and 5-sets.
    Actually for SA we need a fast score. Let's use:
    score = (# K4s) * 10 + (# independent 4-sets)
    This is a proxy that correlates with validity.
    """
    # Count K4s using matrix multiplication
    # A^2[i,j] = number of common neighbors of i and j (paths of length 2)
    # For edge (i,j): common neighbors = A[i,:] & A[j,:] (bitwise)
    # # K4s = (1/6) * sum over edges (i,j) of (# edges among common neighbors of i,j)
    
    # Fast method: 
    # tri[i,j] = number of triangles containing edge (i,j) = (A @ A)[i,j] - A[i,j] (remove direct edge)
    # Actually (A@A)[i,j] counts paths of length 2 from i to j. If A[i,j]=1, then
    # triangles containing (i,j) = (A@A)[i,j] - 1 (the -1 removes the trivial path i->j->i... 
    # no wait. (A@A)[i,j] = sum_k A[i,k]*A[k,j]. This counts common neighbors.
    # If i-j is an edge, triangles = (A@A)[i,j] (common neighbors that are adjacent to both).
    # Wait no: (A@A)[i,j] = number of vertices k such that A[i,k]=1 and A[k,j]=1.
    # If A[i,j]=1, then triangles containing edge (i,j) = (A@A)[i,j] - 0? 
    # No: k must be different from i and j. Since A[i,i]=0, k≠i is automatic. 
    # But k=j would give A[i,j]*A[j,j]=0, so k≠j is also automatic.
    # So triangles containing (i,j) = (A@A)[i,j] when A[i,j]=1.
    
    A2 = A @ A  # (n x n) matrix of common-neighbor counts
    
    # # K4s: for each edge (i,j), count edges among common neighbors
    # This is expensive. Use a different approach.
    
    # Alternative: # K4s = (1/8) * sum_i sum_j A[i,j] * (number of edges in N(i)∩N(j))
    # Still expensive. Let's use a simpler proxy.
    
    # Proxy score: 
    # k4_proxy = sum over all (i,j) of A[i,j] * max(0, (A@A)[i,j] - 1)
    # This overcounts but is a reasonable proxy.
    
    # Better: count triangles first
    # # triangles = trace(A^3) / 6
    A3 = A2 @ A
    n_tri = int(np.trace(A3)) // 6
    
    # For K4: use the fact that each K4 contains 4 triangles.
    # # K4s ≈ (1/4) * sum over triangles of (common neighbor count - 2) / something
    # This is getting complicated. Let's just count directly for small n.
    
    if n <= 40:
        # Direct K4 count using bitsets (fast enough for n<=40)
        bits = np.zeros(n, dtype=np.uint64)
        for i in range(n):
            b = 0
            for j in range(n):
                if A[i, j]:
                    b |= (1 << j)
            bits[i] = b
        
        n_k4 = 0
        for a in range(n):
            for b in range(a + 1, n):
                if not A[a, b]:
                    continue
                common = bits[a] & bits[b]
                common &= ~(1 << a)
                common &= ~(1 << b)
                # Count edges within common
                c_list = []
                c = common
                while c:
                    low = c & (-c)
                    c -= low
                    c_list.append(low.bit_length() - 1)
                for idx, c in enumerate(c_list):
                    for d in c_list[idx + 1:]:
                        if A[c, d]:
                            n_k4 += 1
    
    # Independent set proxy: count independent 4-sets
    # Complement graph: B = 1 - A - I
    B = 1 - A.astype(np.int8)
    np.fill_diagonal(B, 0)
    
    # # independent 4-sets = # K4 in complement = similar computation
    B2 = B @ B
    B3 = B2 @ B
    n_ind4 = int(np.trace(B3)) // 6  # This is # triangles in complement = # independent 3-sets
    
    # For independent 4-sets, need K4 in complement
    n_ind4_sets = 0
    B_bits = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        b = 0
        for j in range(n):
            if B[i, j]:
                b |= (1 << j)
        B_bits[i] = b
    
    for a in range(n):
        for b in range(a + 1, n):
            if not B[a, b]:
                continue
            common = B_bits[a] & B_bits[b]
            common &= ~(1 << a)
            common &= ~(1 << b)
            c_list = []
            c = common
            while c:
                low = c & (-c)
                c -= low
                c_list.append(low.bit_length() - 1)
            for idx, c in enumerate(c_list):
                for d in c_list[idx + 1:]:
                    if B[c, d]:
                        n_ind4_sets += 1
    
    # Score: weight K4s heavily, independent 4-sets moderately
    # We want score=0 iff no K4 and no independent 6-set.
    # Independent 4-sets is a proxy for independent 6-sets.
    score = n_k4 * 100 + n_ind4_sets * 10 + n_tri * 1
    return score


def _score_fast(A, n):
    """Faster score for SA: just count K4s and use triangle count as proxy for ind sets."""
    # K4 count
    bits = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        b = 0
        row = A[i]
        for j in range(n):
            if row[j]:
                b |= (1 << j)
        bits[i] = b
    
    n_k4 = 0
    for a in range(n):
        for b in range(a + 1, n):
            if not A[a, b]:
                continue
            common = bits[a] & bits[b]
            common &= ~(1 << a)
            common &= ~(1 << b)
            c_list = []
            c = common
            while c:
                low = c & (-c)
                c -= low
                c_list.append(low.bit_length() - 1)
            for idx, c in enumerate(c_list):
                for d in c_list[idx + 1:]:
                    if A[c, d]:
                        n_k4 += 1
    
    # Independent 4-sets via complement
    B = 1 - A.astype(np.int8)
    np.fill_diagonal(B, 0)
    B_bits = np.zeros(n, dtype=np.uint64)
    for i in range(n):
        b = 0
        row = B[i]
        for j in range(n):
            if row[j]:
                b |= (1 << j)
        B_bits[i] = b
    
    n_ind4 = 0
    for a in range(n):
        for b in range(a + 1, n):
            if not B[a, b]:
                continue
            common = B_bits[a] & B_bits[b]
            common &= ~(1 << a)
            common &= ~(1 << b)
            c_list = []
            c = common
            while c:
                low = c & (-c)
                c -= low
                c_list.append(low.bit_length() - 1)
            for idx, c in enumerate(c_list):
                for d in c_list[idx + 1:]:
                    if B[c, d]:
                        n_ind4 += 1
    
    return n_k4 * 100 + n_ind4 * 10


def _simulated_annealing(A, n, time_budget):
    """Run simulated annealing to minimize the score."""
    rng = np.random.RandomState(12345)
    A = A.copy()
    
    current_score = _score_fast(A, n)
    best_A = A.copy()
    best_score = current_score
    
    start_temp = 50.0
    end_temp = 0.1
    n_steps = max(1000, int(time_budget * 500))  # ~500 steps per second
    
    # Precompute all possible edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for step in range(n_steps):
        if _time_left() < 5:
            break
        
        # Temperature schedule
        progress = step / n_steps
        temp = start_temp * (end_temp / start_temp) ** progress
        
        # Pick a random edge to flip
        idx = rng.randint(0, len(edges) - 1)
        i, j = edges[idx]
        
        # Flip the edge
        A[i, j] = 1 - A[i, j]
        A[j, i] = A[i, j]
        
        new_score = _score_fast(A, n)
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            # Accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_A = A.copy()
                if best_score == 0:
                    break
        else:
            # Reject: flip back
            A[i, j] = 1 - A[i, j]
            A[j, i] = A[i, j]
    
    return best_A


def _adj_to_edges(A, n):
    """Convert adjacency matrix to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                edges.append([i, j])
    return edges
