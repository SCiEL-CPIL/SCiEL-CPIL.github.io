"""Model-and-optimize: parameterize near-Turán graphs, optimize partition sizes,
then hill-climb with fast bitset-based triangle-degree-sum updates."""
import random
import time
import math
import copy

N = 16
TARGET_EDGES = 65
DEADLINE = 135.0  # safety margin before 150s kill


def _triangles_from_bitset(adj_bits):
    """
    Enumerate all triangles using bitset adjacency.
    adj_bits[i] is an integer where bit j is set if edge (i,j) exists.
    Returns list of (i,j,k) with i<j<k.
    """
    tris = []
    for i in range(N):
        for j in range(i+1, N):
            if not (adj_bits[i] >> j) & 1:
                continue
            # Common neighbors of i and j with index > j
            common = adj_bits[i] & adj_bits[j]
            # Mask to only vertices > j
            mask = ((1 << N) - 1) ^ ((1 << (j+1)) - 1)
            common &= mask
            # Iterate through set bits
            c = common
            while c:
                k = c & (-c).bit_length() - 1
                # Actually, let's use a simpler approach
                k = (c & (-c)).bit_length() - 1
                tris.append((i, j, k))
                c &= c - 1
    return tris


def _max_tri_deg_sum_bitset(adj_bits, deg):
    """Compute max degree sum over all triangles using bitset adjacency."""
    if not any(adj_bits[i] & ((1 << N) - 1) for i in range(N)):
        return 0
    
    best = 0
    for i in range(N):
        for j in range(i+1, N):
            if not (adj_bits[i] >> j) & 1:
                continue
            common = adj_bits[i] & adj_bits[j]
            mask = ((1 << N) - 1) ^ ((1 << (j+1)) - 1)
            common &= mask
            c = common
            while c:
                k = (c & (-c)).bit_length() - 1
                s = deg[i] + deg[j] + deg[k]
                if s > best:
                    best = s
                c &= c - 1
    return best


def _score_full_bitset(adj_bits):
    """Full score computation from bitset adjacency."""
    deg = [0] * N
    for i in range(N):
        deg[i] = bin(adj_bits[i] & ((1 << N) - 1)).count('1')
    score = _max_tri_deg_sum_bitset(adj_bits, deg)
    return score, deg


def _build_turan_plus_one():
    """Build K_{8,8} plus one internal edge in one part. Returns (adj_bits, deg, score)."""
    adj_bits = [0] * N
    # K_{8,8}: vertices 0..7 in A, 8..15 in B
    for i in range(8):
        for j in range(8, 16):
            adj_bits[i] |= (1 << j)
            adj_bits[j] |= (1 << i)
    # One internal edge in A: (0,1)
    adj_bits[0] |= (1 << 1)
    adj_bits[1] |= (1 << 0)
    
    deg = [0]*N
    for i in range(N):
        deg[i] = bin(adj_bits[i] & ((1 << N) - 1)).count('1')
    
    score = _max_tri_deg_sum_bitset(adj_bits, deg)
    return adj_bits, deg, score


def _apply_swap_bitset(adj_bits, deg, e_remove, e_add):
    """
    Apply an edge swap using bitset representation.
    Returns (new_adj_bits, new_deg, new_score).
    """
    u, v = e_remove
    x, y = e_add
    
    new_adj_bits = adj_bits[:]
    new_deg = deg[:]
    
    # Remove edge (u,v)
    new_adj_bits[u] &= ~(1 << v)
    new_adj_bits[v] &= ~(1 << u)
    new_deg[u] -= 1
    new_deg[v] -= 1
    
    # Add edge (x,y)
    new_adj_bits[x] |= (1 << y)
    new_adj_bits[y] |= (1 << x)
    new_deg[x] += 1
    new_deg[y] += 1
    
    new_score = _max_tri_deg_sum_bitset(new_adj_bits, new_deg)
    return new_adj_bits, new_deg, new_score


def construct():
    t0 = time.time()
    rng = random.Random(12345)
    
    # Start with the Turán + 1 construction
    adj_bits, deg, score = _build_turan_plus_one()
    best_adj_bits = adj_bits[:]
    best_deg = deg[:]
    best_score = score
    
    def elapsed():
        return time.time() - t0
    
    # Phase 1: Try different internal edge placements and multi-edge structures
    for a_size in range(4, 13):
        b_size = N - a_size
        cross = a_size * b_size
        internal_needed = TARGET_EDGES - cross
        if internal_needed < 0 or internal_needed > a_size*(a_size-1)//2 + b_size*(b_size-1)//2:
            continue
        
        if internal_needed == 0:
            continue
        
        # Build cross edges
        adj_try_bits = [0] * N
        for i in range(a_size):
            for j in range(a_size, N):
                adj_try_bits[i] |= (1 << j)
                adj_try_bits[j] |= (1 << i)
        
        internal_candidates_a = [(i,j) for i in range(a_size) for j in range(i+1, a_size)]
        internal_candidates_b = [(i,j) for i in range(a_size, N) for j in range(i+1, N)]
        all_internal = internal_candidates_a + internal_candidates_b
        
        for trial in range(30):
            if elapsed() > DEADLINE:
                break
            if internal_needed <= len(all_internal):
                chosen = rng.sample(all_internal, internal_needed)
            else:
                continue
            
            adj_try2_bits = adj_try_bits[:]
            for (i, j) in chosen:
                adj_try2_bits[i] |= (1 << j)
                adj_try2_bits[j] |= (1 << i)
            
            deg_try = [0]*N
            for i in range(N):
                deg_try[i] = bin(adj_try2_bits[i] & ((1 << N) - 1)).count('1')
            
            score_try = _max_tri_deg_sum_bitset(adj_try2_bits, deg_try)
            
            if score_try < best_score:
                best_score = score_try
                best_adj_bits = adj_try2_bits[:]
                best_deg = deg_try[:]
    
    # Phase 2: Simulated annealing with bitset representation
    adj_bits = best_adj_bits[:]
    deg = best_deg[:]
    cur_score = best_score
    
    # Precompute edge and non-edge lists
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
    nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
    
    # Simulated annealing with better schedule
    start_temp = 4.0
    end_temp = 0.01
    max_iters = 200000
    cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
    temp = start_temp
    iters = 0
    
    while iters < max_iters:
        if elapsed() > DEADLINE:
            break
        
        # Pick random edge and non-edge
        if not edges or not nonedges:
            break
        
        e_remove = edges[rng.randrange(len(edges))]
        e_add = nonedges[rng.randrange(len(nonedges))]
        
        new_adj_bits, new_deg, new_score = _apply_swap_bitset(adj_bits, deg, e_remove, e_add)
        
        # Acceptance decision
        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.2
        else:
            delta = new_score - cur_score
            if temp < 1e-10:
                accept = False
            else:
                accept = math.exp(-delta / temp) > rng.random()
        
        if accept:
            adj_bits = new_adj_bits
            deg = new_deg
            cur_score = new_score
            
            if new_score < best_score:
                best_score = new_score
                best_adj_bits = adj_bits[:]
                best_deg = deg[:]
            
            # Update edge/non-edge lists
            edges.remove(e_remove)
            nonedges.append(e_remove)
            nonedges.remove(e_add)
            edges.append(e_add)
        
        iters += 1
        temp = max(end_temp, temp * cooling)
    
    # Phase 3: Deterministic hill-climbing polish
    if elapsed() < DEADLINE - 5.0:
        adj_bits = best_adj_bits[:]
        deg = best_deg[:]
        cur_score = best_score
        
        all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
        nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
        
        improved = True
        polish_iters = 0
        max_polish_iters = 50000
        
        while improved and polish_iters < max_polish_iters and elapsed() < DEADLINE - 2.0:
            improved = False
            polish_iters += 1
            
            for e_remove in edges:
                if elapsed() > DEADLINE - 2.0:
                    break
                for e_add in nonedges:
                    new_adj_bits, new_deg, new_score = _apply_swap_bitset(adj_bits, deg, e_remove, e_add)
                    if new_score < cur_score:
                        adj_bits = new_adj_bits
                        deg = new_deg
                        cur_score = new_score
                        
                        if cur_score < best_score:
                            best_score = cur_score
                            best_adj_bits = adj_bits[:]
                            best_deg = deg[:]
                        
                        edges.remove(e_remove)
                        nonedges.append(e_remove)
                        nonedges.remove(e_add)
                        edges.append(e_add)
                        
                        improved = True
                        break
                if improved:
                    break
    
    # Return the best found
    result_edges = [[i, j] for i in range(N) for j in range(i+1, N) if (best_adj_bits[i] >> j) & 1]
    return {'n': N, 'edges': result_edges}
