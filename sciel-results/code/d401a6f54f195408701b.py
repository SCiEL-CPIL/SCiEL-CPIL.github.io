"""Model-and-optimize: parameterize near-Turán graphs, optimize partition sizes,
then hill-climb with incremental triangle tracking and multi-restart SA."""
import random
import time
import math

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0  # safety margin before 150s kill
FULL_MASK = (1 << N) - 1


def _build_adj_from_edges(edge_set):
    """Build bitset adjacency from a set of edges."""
    adj_bits = [0] * N
    for (u, v) in edge_set:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    return adj_bits


def _compute_degrees(adj_bits):
    """Compute degrees from bitset adjacency."""
    deg = [0] * N
    for i in range(N):
        deg[i] = adj_bits[i].bit_count()
    return deg


def _enumerate_triangles(adj_bits):
    """Enumerate all triangles. Returns list of (i,j,k) with i<j<k."""
    tris = []
    for i in range(N):
        for j in range(i + 1, N):
            if not (adj_bits[i] >> j) & 1:
                continue
            common = adj_bits[i] & adj_bits[j]
            mask = FULL_MASK ^ ((1 << (j + 1)) - 1)
            common &= mask
            c = common
            while c:
                k = (c & (-c)).bit_length() - 1
                tris.append((i, j, k))
                c &= c - 1
    return tris


def _max_tri_deg_sum(adj_bits, deg, tris):
    """Compute max degree sum over all triangles."""
    if not tris:
        return 0
    best = 0
    for (i, j, k) in tris:
        s = deg[i] + deg[j] + deg[k]
        if s > best:
            best = s
    return best


def _build_tri_counts(adj_bits):
    """
    For each vertex v, tri_count[v] = number of triangles containing v.
    Also returns the list of triangles.
    """
    tris = _enumerate_triangles(adj_bits)
    tri_count = [0] * N
    for (i, j, k) in tris:
        tri_count[i] += 1
        tri_count[j] += 1
        tri_count[k] += 1
    return tris, tri_count


def _score_graph(adj_bits):
    """Full score computation."""
    deg = _compute_degrees(adj_bits)
    tris = _enumerate_triangles(adj_bits)
    score = _max_tri_deg_sum(adj_bits, deg, tris)
    return score, deg, tris


def _compute_max_tri_sum_fast(adj_bits, deg):
    """
    Compute max triangle degree sum using bitset operations.
    For each edge (i,j), check common neighbors k>j.
    """
    best = 0
    for i in range(N):
        for j in range(i + 1, N):
            if not (adj_bits[i] >> j) & 1:
                continue
            common = adj_bits[i] & adj_bits[j]
            mask = FULL_MASK ^ ((1 << (j + 1)) - 1)
            common &= mask
            c = common
            while c:
                k = (c & (-c)).bit_length() - 1
                s = deg[i] + deg[j] + deg[k]
                if s > best:
                    best = s
                c &= c - 1
    return best


def _build_turan_plus_one():
    """Build K_{8,8} plus one internal edge in one part."""
    adj_bits = [0] * N
    for i in range(8):
        for j in range(8, 16):
            adj_bits[i] |= (1 << j)
            adj_bits[j] |= (1 << i)
    # One internal edge in A: (0,1)
    adj_bits[0] |= (1 << 1)
    adj_bits[1] |= (1 << 0)
    
    deg = _compute_degrees(adj_bits)
    tris = _enumerate_triangles(adj_bits)
    score = _max_tri_deg_sum(adj_bits, deg, tris)
    return adj_bits, deg, tris, score


def _eval_swap_fast(adj_bits, deg, e_remove, e_add):
    """
    Evaluate the effect of removing edge e_remove=(u,v) and adding edge e_add=(x,y)
    on the max triangle degree sum, using incremental updates.
    Returns the new max score (or -1 if invalid).
    
    Key insight: only triangles involving u, v, x, y can change.
    We compute the new degrees for these 4 vertices and check all affected triangles.
    """
    u, v = e_remove
    x, y = e_add
    
    # Validate: can't remove non-edge, can't add existing edge
    if not (adj_bits[u] >> v) & 1:
        return -1
    if (adj_bits[x] >> y) & 1:
        return -1
    # Can't add self-loop
    if x == y:
        return -1
    
    # New degrees for affected vertices
    new_deg = deg[:]
    new_deg[u] -= 1
    new_deg[v] -= 1
    new_deg[x] += 1
    new_deg[y] += 1
    
    # Build new adj for the 4 affected vertices (for triangle checks)
    new_adj_u = (adj_bits[u] & ~(1 << v))
    new_adj_v = (adj_bits[v] & ~(1 << u))
    new_adj_x = (adj_bits[x] | (1 << y))
    new_adj_y = (adj_bits[y] | (1 << x))
    
    # Map vertex -> its new adjacency
    new_adj = adj_bits[:]
    new_adj[u] = new_adj_u
    new_adj[v] = new_adj_v
    new_adj[x] = new_adj_x
    new_adj[y] = new_adj_y
    
    # Find all triangles that include at least one of {u, v, x, y}
    # A triangle (a,b,c) is affected if any of a,b,c is in {u,v,x,y}
    # AND the triangle exists in new_adj but not old, or vice versa.
    # Simpler: enumerate all triangles in new_adj that include u, v, x, or y,
    # and also check that no triangle in old_adj (that survives) has higher sum.
    
    # Actually, the safest fast approach: enumerate all triangles in new_adj
    # that contain at least one of the 4 affected vertices.
    # Plus, we need to make sure we don't miss triangles that existed before
    # but are now "unaffected" - but those haven't changed, so their degree sums
    # are the same. The max over unchanged triangles is <= old max.
    # So new_max = max(max over affected triangles in new graph, 
    #                  max over unaffected triangles)
    # The unaffected triangles have the same degree sums as before.
    # But we don't track per-triangle sums. 
    # 
    # Alternative: just compute max over ALL triangles in new_adj.
    # For N=16, this is at most C(16,3)=560 triangles, and with bitsets it's fast.
    
    # Let's just do the full computation on new_adj but only for triangles 
    # involving the 4 affected vertices, plus we need the global max.
    # Actually for N=16 the full enumeration is fast enough. Let's benchmark.
    
    # Full fast computation on new_adj:
    best = 0
    for i in range(N):
        for j in range(i + 1, N):
            if not (new_adj[i] >> j) & 1:
                continue
            common = new_adj[i] & new_adj[j]
            mask = FULL_MASK ^ ((1 << (j + 1)) - 1)
            common &= mask
            c = common
            while c:
                k = (c & (-c)).bit_length() - 1
                s = new_deg[i] + new_deg[j] + new_deg[k]
                if s > best:
                    best = s
                c &= c - 1
    
    return best


def _apply_swap(adj_bits, deg, e_remove, e_add):
    """
    Apply an edge swap. Returns (new_adj_bits, new_deg, new_score, new_tris).
    Uses incremental triangle update.
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
    
    # Recompute triangles (fast for N=16)
    new_tris = _enumerate_triangles(new_adj_bits)
    new_score = _max_tri_deg_sum(new_adj_bits, new_deg, new_tris)
    
    return new_adj_bits, new_deg, new_score, new_tris


def _apply_swap_fast(adj_bits, deg, e_remove, e_add):
    """
    Fast swap that only computes the new score without building full triangle list.
    Returns (new_adj_bits, new_deg, new_score).
    """
    u, v = e_remove
    x, y = e_add
    
    new_adj_bits = adj_bits[:]
    new_deg = deg[:]
    
    new_adj_bits[u] &= ~(1 << v)
    new_adj_bits[v] &= ~(1 << u)
    new_deg[u] -= 1
    new_deg[v] -= 1
    
    new_adj_bits[x] |= (1 << y)
    new_adj_bits[y] |= (1 << x)
    new_deg[x] += 1
    new_deg[y] += 1
    
    new_score = _compute_max_tri_sum_fast(new_adj_bits, new_deg)
    
    return new_adj_bits, new_deg, new_score


def _try_internal_edges(a_size, rng, n_trials=20):
    """Try random internal edge placements for a given bipartition size."""
    b_size = N - a_size
    cross = a_size * b_size
    internal_needed = TARGET_EDGES - cross
    if internal_needed < 0:
        return None
    max_internal = a_size * (a_size - 1) // 2 + b_size * (b_size - 1) // 2
    if internal_needed > max_internal:
        return None
    
    # Build cross edges
    adj_base = [0] * N
    for i in range(a_size):
        for j in range(a_size, N):
            adj_base[i] |= (1 << j)
            adj_base[j] |= (1 << i)
    
    internal_candidates_a = [(i, j) for i in range(a_size) for j in range(i + 1, a_size)]
    internal_candidates_b = [(i, j) for i in range(a_size, N) for j in range(i + 1, N)]
    all_internal = internal_candidates_a + internal_candidates_b
    
    best_result = None
    best_score = float('inf')
    
    for _ in range(n_trials):
        if internal_needed > len(all_internal):
            break
        chosen = rng.sample(all_internal, internal_needed)
        
        adj_try = adj_base[:]
        for (i, j) in chosen:
            adj_try[i] |= (1 << j)
            adj_try[j] |= (1 << i)
        
        deg_try = _compute_degrees(adj_try)
        score_try = _compute_max_tri_sum_fast(adj_try, deg_try)
        
        if score_try < best_score:
            best_score = score_try
            tris_try = _enumerate_triangles(adj_try)
            best_result = (adj_try, deg_try, tris_try, score_try)
    
    return best_result


def construct():
    t0 = time.time()
    rng = random.Random(42)
    
    def elapsed():
        return time.time() - t0
    
    # Phase 1: Try different initial structures
    best_adj_bits, best_deg, best_tris, best_score = _build_turan_plus_one()
    
    # Try various bipartition sizes
    for a_size in range(5, 12):
        if elapsed() > DEADLINE - 60:
            break
        result = _try_internal_edges(a_size, rng, n_trials=30)
        if result and result[3] < best_score:
            best_adj_bits, best_deg, best_tris, best_score = result
    
    # Also try some random non-bipartite starts
    for trial in range(20):
        if elapsed() > DEADLINE - 60:
            break
        # Start from random graph with 65 edges
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random_edges = set(rng.sample(all_pairs, TARGET_EDGES))
        adj_try = _build_adj_from_edges(random_edges)
        deg_try = _compute_degrees(adj_try)
        score_try = _compute_max_tri_sum_fast(adj_try, deg_try)
        if score_try < best_score:
            tris_try = _enumerate_triangles(adj_try)
            best_adj_bits, best_deg, best_tris, best_score = (
                adj_try, deg_try, tris_try, score_try)
    
    # Phase 2: Simulated Annealing with multiple restarts
    time_budget_sa = DEADLINE - elapsed() - 15.0  # reserve 15s for final polish
    if time_budget_sa > 5.0:
        # Determine number of SA restarts based on available time
        n_restarts = max(1, int(time_budget_sa / 20.0))
        time_per_restart = time_budget_sa / n_restarts
        
        for restart in range(n_restarts):
            if elapsed() > DEADLINE - 15.0:
                break
            
            # For restarts > 0, perturb the best solution
            if restart == 0:
                adj_bits = best_adj_bits[:]
                deg = best_deg[:]
                tris = best_tris[:]
                cur_score = best_score
            else:
                # Perturb: do k random swaps
                adj_bits = best_adj_bits[:]
                deg = _compute_degrees(adj_bits)
                cur_score = _compute_max_tri_sum_fast(adj_bits, deg)
                
                all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
                edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
                nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
                
                for _ in range(5):
                    if not edges or not nonedges:
                        break
                    e_remove = edges[rng.randrange(len(edges))]
                    e_add = nonedges[rng.randrange(len(nonedges))]
                    new_adj, new_deg, new_score = _apply_swap_fast(
                        adj_bits, deg, e_remove, e_add)
                    adj_bits, deg = new_adj, new_deg
                    cur_score = new_score
                    edges.remove(e_remove)
                    nonedges.append(e_remove)
                    nonedges.remove(e_add)
                    edges.append(e_add)
            
            # SA parameters
            start_temp = 3.0
            end_temp = 0.01
            max_iters = 100000
            cooling = (end_temp / start_temp) ** (1.0 / max(1, max_iters - 1))
            temp = start_temp
            
            all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
            nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
            
            iters = 0
            while iters < max_iters:
                if elapsed() > DEADLINE - 15.0:
                    break
                
                if not edges or not nonedges:
                    break
                
                e_remove = edges[rng.randrange(len(edges))]
                e_add = nonedges[rng.randrange(len(nonedges))]
                
                new_adj_bits, new_deg, new_score = _apply_swap_fast(
                    adj_bits, deg, e_remove, e_add)
                
                # Acceptance
                if new_score < cur_score:
                    accept = True
                elif new_score == cur_score:
                    accept = rng.random() < 0.3
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
                        best_tris = _enumerate_triangles(adj_bits)
                    
                    edges.remove(e_remove)
                    nonedges.append(e_remove)
                    nonedges.remove(e_add)
                    edges.append(e_add)
                
                iters += 1
                temp = max(end_temp, temp * cooling)
    
    # Phase 3: Deterministic hill-climbing polish (multiple passes)
    if elapsed() < DEADLINE - 3.0:
        adj_bits = best_adj_bits[:]
        deg = best_deg[:]
        cur_score = best_score
        
        all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
        edges = [(i, j) for (i, j) in all_pairs if (adj_bits[i] >> j) & 1]
        nonedges = [(i, j) for (i, j) in all_pairs if not ((adj_bits[i] >> j) & 1)]
        
        # Multiple passes of best-improvement hill climbing
        for polish_pass in range(10):
            if elapsed() > DEADLINE - 3.0:
                break
            
            improved = False
            # Try all edge swaps, keep best improvement
            best_swap = None
            best_new_score = cur_score
            best_new_adj = None
            best_new_deg = None
            
            for e_remove in edges:
                if elapsed() > DEADLINE - 3.0:
                    break
                for e_add in nonedges:
                    new_adj_bits, new_deg, new_score = _apply_swap_fast(
                        adj_bits, deg, e_remove, e_add)
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_swap = (e_remove, e_add)
                        best_new_adj = new_adj_bits
                        best_new_deg = new_deg
            
            if best_swap is not None and best_new_score < cur_score:
                e_remove, e_add = best_swap
                adj_bits = best_new_adj
                deg = best_new_deg
                cur_score = best_new_score
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_adj_bits = adj_bits[:]
                    best_deg = deg[:]
                    best_tris = _enumerate_triangles(adj_bits)
                
                edges.remove(e_remove)
                nonedges.append(e_remove)
                nonedges.remove(e_add)
                edges.append(e_add)
                improved = True
            
            if not improved:
                break
    
    # Final verification
    final_deg = _compute_degrees(best_adj_bits)
    final_tris = _enumerate_triangles(best_adj_bits)
    final_score = _max_tri_deg_sum(best_adj_bits, final_deg, final_tris)
    
    result_edges = [[i, j] for i in range(N) for j in range(i + 1, N)
                    if (best_adj_bits[i] >> j) & 1]
    
    # Verify edge count
    assert len(result_edges) == TARGET_EDGES, f"Expected {TARGET_EDGES} edges, got {len(result_edges)}"
    
    return {'n': N, 'edges': result_edges}
