<<<<<<< SEARCH
import time
import random
from scipy.optimize import milp, LinearConstraint, Bounds
import numpy as np

N = 16

def _groetzsch_edges():
    edges = []
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    orig_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    for u, v in orig_edges:
        edges.append([v, 5 + u])
        edges.append([u, 5 + v])
    for i in range(5):
        edges.append([10, 5 + i])
    return edges

def _adj_from_edges(edge_list):
    adj = [0] * N
    for u, v in edge_list:
        if u < N and v < N:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def _edge_count(adj):
    return sum(bin(a).count("1") for a in adj) // 2

def _is_triangle_free(adj):
    for u in range(N):
        mask = adj[u]
        while mask:
            lb = mask & (-mask)
            v = lb.bit_length() - 1
            mask ^= lb
            if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                return False
    return True

def _is_3_colorable(adj):
    colors = [-1] * N
    order = sorted(range(N), key=lambda x: bin(adj[x]).count('1'), reverse=True)
    order_pos = [0] * N
    for i, v in enumerate(order):
        order_pos[v] = i

    def backtrack(idx):
        if idx == N:
            return True
        v = order[idx]
        forbidden = 0
        m = adj[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if order_pos[u] < idx:
                forbidden |= (1 << colors[u])
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False

    return backtrack(0)

def _is_valid(adj):
    return _is_triangle_free(adj) and not _is_3_colorable(adj)

def _try_add_edge(adj, u, v):
    if (adj[u] >> v) & 1:
        return False
    if adj[u] & adj[v]:
        return False
    return True

def _add_edge(adj, u, v):
    adj[u] |= 1 << v
    adj[v] |= 1 << u

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _greedy_add_edges(adj, deadline, rng):
    cnt = _edge_count(adj)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        missing = []
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                break
            if _try_add_edge(adj, u, v):
                _add_edge(adj, u, v)
                cnt += 1
                improved = True
    return adj, cnt

def _local_search_fast(adj, deadline, rng, core_edges_set):
    """
    Local search that never removes core edges.
    Does not check 3-colorability (guaranteed by core).
    """
    best_adj = [a for a in adj]
    best_cnt = _edge_count(adj)
    cur_adj = [a for a in adj]
    cur_cnt = _edge_count(cur_adj)

    # Greedy add
    cur_adj, cur_cnt = _greedy_add_edges(cur_adj, deadline, rng)
    if cur_cnt > best_cnt:
        best_cnt = cur_cnt
        best_adj = [a for a in cur_adj]

    # Swaps
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        # Get non-core edges
        existing = []
        for u in range(N):
            for v in range(u + 1, N):
                if (cur_adj[u] >> v) & 1:
                    if (u, v) not in core_edges_set and (v, u) not in core_edges_set:
                        existing.append((u, v))
        
        if not existing:
            break

        # Try removing one non-core edge and adding more
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            rem_u, rem_v = rng.choice(existing)
            adj_copy = [a for a in cur_adj]
            _remove_edge(adj_copy, rem_u, rem_v)
            adj_copy, cnt_copy = _greedy_add_edges(adj_copy, deadline, rng)
            if cnt_copy > cur_cnt:
                cur_adj = adj_copy
                cur_cnt = cnt_copy
                improved = True
                if cur_cnt > best_cnt:
                    best_cnt = cnt_copy
                    best_adj = [a for a in cur_adj]
                    
        # Try pure add
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not ((cur_adj[u] >> v) & 1)]
        rng.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 2.0:
                break
            if _try_add_edge(cur_adj, u, v):
                _add_edge(cur_adj, u, v)
                cur_cnt += 1
                improved = True
                if cur_cnt > best_cnt:
                    best_cnt = cur_cnt
                    best_adj = [a for a in cur_adj]
                break

    return best_adj, best_cnt

def _solve_milp(core_edges):
    """
    Solve the maximum triangle-free graph problem containing core_edges using MILP.
    """
    # Total edges in K16
    total_edges = N * (N - 1) // 2
    # Map edge to index
    edge_to_idx = {}
    idx = 0
    for u in range(N):
        for v in range(u + 1, N):
            edge_to_idx[(u, v)] = idx
            idx += 1
    
    # Objective: maximize sum x_e
    c = np.ones(total_edges)
    
    # Constraints:
    # 1. Core edges must be 1
    # 2. Triangle constraints: x_uv + x_vw + x_wu <= 2
    
    # We can use LinearConstraint
    # But scipy.optimize.milp expects A_ub x <= b_ub
    # Let's build the matrix A for triangle constraints.
    
    # Number of triangles in K16 is C(16,3) = 560
    num_triangles = 560
    A = np.zeros((num_triangles, total_edges), dtype=np.int8)
    b = np.full(num_triangles, 2, dtype=np.int8)
    
    tri_idx = 0
    for u in range(N):
        for v in range(u + 1, N):
            for w in range(v + 1, N):
                # Triangle u-v-w
                i_uv = edge_to_idx[(u, v)]
                i_vw = edge_to_idx[(v, w)]
                i_uw = edge_to_idx[(u, w)]
                A[tri_idx, i_uv] = 1
                A[tri_idx, i_vw] = 1
                A[tri_idx, i_uw] = 1
                tri_idx += 1
                
    # Core constraints: x_e = 1 for e in core_edges
    # This can be handled by fixing the variables or adding constraints.
    # scipy.optimize.milp supports bounds.
    # If we set lower bound = 1 and upper bound = 1 for core edges, they are fixed.
    
    lb = np.zeros(total_edges)
    ub = np.ones(total_edges)
    
    for u, v in core_edges:
        if u > v: u, v = v, u
        i = edge_to_idx[(u, v)]
        lb[i] = 1
        ub[i] = 1
        
    bounds = Bounds(lb, ub)
    
    # We don't need explicit equality constraints for core if bounds are set.
    # We only have inequality constraints for triangles.
    # A_ub = A, b_ub = b
    
    try:
        res = milp(c=c, constraints=LinearConstraint(A, ub=b), bounds=bounds, integrality=np.ones(total_edges))
        if res.status == 0:
            x = res.x
            # Reconstruct graph
            adj = [0] * N
            count = 0
            for u in range(N):
                for v in range(u + 1, N):
                    i = edge_to_idx[(u, v)]
                    if x[i] > 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        count += 1
            return adj, count
    except Exception:
        pass
    return None, 0

def construct():
    start = time.time()
    deadline = start + 140.0

    base_edges = _groetzsch_edges()
    base_adj = _adj_from_edges(base_edges)
    core_edges_set = set()
    for u, v in base_edges:
        if u > v: u, v = v, u
        core_edges_set.add((u, v))
        
    # Try MILP first
    milp_adj, milp_cnt = _solve_milp(base_edges)
    if milp_adj is not None and _is_valid(milp_adj):
        # Verify
        if _is_triangle_free(milp_adj) and not _is_3_colorable(milp_adj):
            # Convert to list
            out = []
            for u in range(N):
                m = milp_adj[u]
                while m:
                    lb = m & (-m)
                    v = lb.bit_length() - 1
                    m ^= lb
                    if v > u:
                        out.append([u, v])
            return out

    # Fallback to fast local search
    best_adj = [a for a in base_adj]
    best_cnt = _edge_count(base_adj)
    rng = random.Random(42)
    
    # Run local search multiple times
    for _ in range(10):
        if time.time() > deadline - 10.0:
            break
        adj = [a for a in base_adj]
        result_adj, result_cnt = _local_search_fast(adj, deadline, rng, core_edges_set)
        if result_cnt > best_cnt and _is_triangle_free(result_adj):
            # Check 3-colorability just once for the best found
            if not _is_3_colorable(result_adj):
                best_cnt = result_cnt
                best_adj = [a for a in result_adj]

    # Final check
    if not _is_valid(best_adj):
        best_adj = [a for a in base_adj]
        best_cnt = _edge_count(base_adj)

    out = []
    for u in range(N):
        m = best_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                out.append([u, v])
    return out
=======
import time
import random

N = 16
TOTAL_EDGES = N * (N - 1) // 2  # 120

def _edge_index(u, v):
    if u > v:
        u, v = v, u
    # u < v, index = u*(2N-1-u)//2 + v - u - 1  ... let's use a simpler formula
    # edges are (0,1),(0,2),...,(0,15),(1,2),...,(1,15),...,(14,15)
    # For vertex u, the edges starting at u go to u+1, u+2, ..., N-1
    # Number of edges before vertex u: sum_{i=0}^{u-1} (N-1-i) = u*(N-1) - u*(u-1)//2
    base = u * (N - 1) - u * (u - 1) // 2
    offset = v - u - 1
    return base + offset

def _edge_from_index(idx):
    u = 0
    while True:
        cnt = N - 1 - u
        if idx < cnt:
            v = u + 1 + idx
            return u, v
        idx -= cnt
        u += 1

def _groetzsch_edges():
    """Grötzsch graph: Mycielski of C5. Vertices 0-4 (C5), 5-9 (copies), 10 (apex)."""
    edges = []
    # C5: 0-1-2-3-4-0
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # Copies 5-9: edge (i,j) in C5 -> (j, 5+i) and (i, 5+j)
    for i in range(5):
        j = (i + 1) % 5
        edges.append([j, 5 + i])
        edges.append([i, 5 + j])
    # Apex 10 connected to all copies 5-9
    for i in range(5):
        edges.append([10, 5 + i])
    return edges

def _adj_from_edges(edge_list):
    adj = [0] * N
    for u, v in edge_list:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _edge_count(adj):
    return sum(bin(a).count("1") for a in adj) // 2

def _is_triangle_free(adj):
    for u in range(N):
        m = adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if adj[v] & adj[u] & ~(1 << u) & ~(1 << v):
                return False
    return True

def _is_3_colorable(adj):
    """Exact 3-colorability check via backtracking with degree ordering."""
    colors = [-1] * N
    order = sorted(range(N), key=lambda x: bin(adj[x]).count('1'), reverse=True)
    order_pos = [0] * N
    for i, v in enumerate(order):
        order_pos[v] = i

    def backtrack(idx):
        if idx == N:
            return True
        v = order[idx]
        forbidden = 0
        m = adj[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            if order_pos[u] < idx:
                forbidden |= (1 << colors[u])
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False

    return backtrack(0)

def _branch_and_bound(core_edge_mask, deadline):
    """
    Branch-and-bound on the line graph of K16.
    Each 'node' is a potential edge. Two edges are incompatible if they share a vertex
    AND adding both would create a triangle with some third edge... actually no.
    
    In the line graph of K16, two edges (u,v) and (u,w) are adjacent (incompatible in IS)
    if they share vertex u. An independent set in the line graph is a set of pairwise
    non-adjacent edges, i.e., no two edges share a vertex. That's a matching, not what we want.
    
    Wait - I need to reconsider. Triangle-free means: for every triple (u,v,w), not all three
    edges (u,v),(v,w),(w,u) are present. This is NOT the same as a matching.
    
    Let me use a different formulation: we're selecting a subset of the 120 edges of K16
    such that no triangle is fully selected. This is the maximum 2-set packing / triangle-free
    subgraph problem.
    
    Branch-and-bound: decide for each edge whether to include or exclude.
    Pruning: if including an edge would complete a triangle with two already-included edges, skip.
    Bound: current_count + remaining_edges <= best_count -> prune.
    Additional pruning: if the current partial graph (with all remaining edges added optimally)
    is 3-colorable, we can't guarantee 4-chromatic... actually we need the FINAL graph to be
    non-3-colorable. We can check: if even with all remaining edges the graph is 3-colorable,
    prune.
    """
    # Precompute: for each edge, which other edges would form a triangle with it
    # Edge e1=(a,b) and e2=(b,c) form a triangle if (a,c) is also present
    # So for edge (u,v), the "forbidden pairs" are: for each w != u,v, if both (u,w) and (v,w) are in,
    # then adding (u,v) creates a triangle.
    
    # Better: for each edge e, precompute the set of edges that are "triangle partners"
    # Edge (u,v) + edge (u,w) + edge (v,w) = triangle
    # So (u,v) conflicts with pair {(u,w), (v,w)} for each w
    
    # For B&B: when we add edge e=(u,v), we must ensure that for no w are BOTH (u,w) and (v,w) already selected.
    
    # Let's use a bitset representation for the 120 edges
    # edge_mask: 120-bit integer, bit i = 1 means edge i is selected
    
    # Precompute for each edge i, the list of (j, k) pairs such that {i, j, k} forms a triangle
    # i.e., edge i = (u,v), and j = (u,w), k = (v,w) for some w
    
    tri_partners = [[] for _ in range(TOTAL_EDGES)]
    for i in range(TOTAL_EDGES):
        u, v = _edge_from_index(i)
        for w in range(N):
            if w == u or w == v:
                continue
            j = _edge_index(u, w)
            k = _edge_index(v, w)
            tri_partners[i].append((j, k))
    
    # Also precompute: for each edge i, the mask of edges that share a vertex with i
    # (not directly needed for triangle-free, but useful for bounding)
    
    best_count = 0
    best_mask = core_edge_mask  # Start with core
    
    # Verify core is valid
    core_adj = [0] * N
    m = core_edge_mask
    while m:
        lb = m & (-m)
        i = lb.bit_length() - 1
        m ^= lb
        u, v = _edge_from_index(i)
        core_adj[u] |= (1 << v)
        core_adj[v] |= (1 << u)
    
    if not _is_triangle_free(core_adj):
        return None, 0
    if _is_3_colorable(core_adj):
        return None, 0
    
    best_count = _edge_count(core_adj)
    
    # Order edges: core edges first (already decided), then others by some heuristic
    # Actually, let's order non-core edges by how "constraining" they are
    # An edge that participates in many triangles is more constraining
    
    edge_order = []
    for i in range(TOTAL_EDGES):
        if (core_edge_mask >> i) & 1:
            continue
        edge_order.append(i)
    
    # Sort by number of triangle partners (more constraining first)
    edge_order.sort(key=lambda i: len(tri_partners[i]), reverse=True)
    
    # B&B state: current mask, current count
    # We process edges in order, deciding include/exclude
    
    import sys
    sys.setrecursionlimit(10000)
    
    def bnb(pos, mask, count):
        nonlocal best_count, best_mask
        
        if time.time() > deadline:
            return
        
        if pos == len(edge_order):
            if count > best_count:
                # Verify 3-colorability
                adj = [0] * N
                m = mask
                while m:
                    lb = m & (-m)
                    i = lb.bit_length() - 1
                    m ^= lb
                    u, v = _edge_from_index(i)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                if not _is_3_colorable(adj):
                    best_count = count
                    best_mask = mask
            return
        
        # Upper bound: count + (len(edge_order) - pos)
        if count + (len(edge_order) - pos) <= best_count:
            return
        
        e = edge_order[pos]
        u, v = _edge_from_index(e)
        
        # Try including edge e
        # Check: does adding e create a triangle?
        # For each (j, k) in tri_partners[e], if both j and k are in mask, triangle!
        can_include = True
        for j, k in tri_partners[e]:
            if ((mask >> j) & 1) and ((mask >> k) & 1):
                can_include = False
                break
        
        if can_include:
            new_mask = mask | (1 << e)
            # Quick 3-colorability pruning: if the graph with ALL remaining edges is 3-colorable, prune
            # This is expensive, so only do it periodically
            bnb(pos + 1, new_mask, count + 1)
        
        # Try excluding edge e
        bnb(pos + 1, mask, count)
    
    bnb(0, core_edge_mask, best_count)
    
    if best_mask == core_edge_mask and best_count <= _edge_count(core_adj):
        return None, 0
    
    # Reconstruct adjacency
    adj = [0] * N
    m = best_mask
    while m:
        lb = m & (-m)
        i = lb.bit_length() - 1
        m ^= lb
        u, v = _edge_from_index(i)
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    return adj, best_count

def _greedy_fill(adj, deadline):
    """Greedily add edges that don't create triangles."""
    cnt = _edge_count(adj)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for u in range(N):
            for v in range(u + 1, N):
                if (adj[u] >> v) & 1:
                    continue
                # Check triangle-free: for all w, not (adj[u] has w AND adj[v] has w)
                if adj[u] & adj[v]:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                cnt += 1
                improved = True
    return adj, cnt

def _local_search(adj, deadline, rng, core_mask):
    """Local search: try removing non-core edges and re-filling."""
    best_adj = [a for a in adj]
    best_cnt = _edge_count(adj)
    cur_adj = [a for a in adj]
    cur_cnt = _edge_count(cur_adj)
    
    # Greedy fill
    cur_adj, cur_cnt = _greedy_fill(cur_adj, deadline)
    if cur_cnt > best_cnt:
        best_cnt = cur_cnt
        best_adj = [a for a in cur_adj]
    
    iterations = 0
    while time.time() < deadline - 3.0:
        iterations += 1
        # Pick a random non-core edge to remove
        non_core_edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if (cur_adj[u] >> v) & 1:
                    ei = _edge_index(u, v)
                    if not ((core_mask >> ei) & 1):
                        non_core_edges.append((u, v))
        
        if not non_core_edges:
            break
        
        # Try removing up to 3 random non-core edges
        for _ in range(min(5, len(non_core_edges))):
            if time.time() > deadline - 3.0:
                break
            rem_count = rng.randint(1, 3)
            to_remove = rng.sample(non_core_edges, min(rem_count, len(non_core_edges)))
            adj_copy = [a for a in cur_adj]
            for u, v in to_remove:
                adj_copy[u] &= ~(1 << v)
                adj_copy[v] &= ~(1 << u)
            
            adj_copy, cnt_copy = _greedy_fill(adj_copy, deadline)
            if cnt_copy > cur_cnt:
                cur_adj = adj_copy
                cur_cnt = cnt_copy
                if cur_cnt > best_cnt:
                    best_cnt = cnt_cnt
                    best_adj = [a for a in cur_adj]
    
    return best_adj, best_cnt

def construct():
    start = time.time()
    deadline = start + 135.0  # Safety margin before 150s kill
    
    # Build Grötzsch core
    core_edges = _groetzsch_edges()
    core_mask = 0
    for u, v in core_edges:
        ei = _edge_index(u, v)
        core_mask |= (1 << ei)
    
    core_adj = _adj_from_edges(core_edges)
    
    # Verify core
    assert _is_triangle_free(core_adj), "Core is not triangle-free!"
    assert not _is_3_colorable(core_adj), "Core is 3-colorable!"
    
    best_adj = [a for a in core_adj]
    best_cnt = _edge_count(core_adj)
    
    # Phase 1: Branch-and-bound
    bnb_adj, bnb_cnt = _branch_and_bound(core_mask, deadline - 30.0)
    if bnb_adj is not None and bnb_cnt > best_cnt:
        if _is_triangle_free(bnb_adj) and not _is_3_colorable(bnb_adj):
            best_cnt = bnb_cnt
            best_adj = [a for a in bnb_adj]
    
    # Phase 2: Local search from best found
    if time.time() < deadline - 5.0:
        rng = random.Random(42)
        for seed in range(20):
            if time.time() > deadline - 5.0:
                break
            rng = random.Random(seed)
            adj_copy = [a for a in best_adj]
            ls_adj, ls_cnt = _local_search(adj_copy, deadline, rng, core_mask)
            if ls_cnt > best_cnt:
                if _is_triangle_free(ls_adj) and not _is_3_colorable(ls_adj):
                    best_cnt = ls_cnt
                    best_adj = [a for a in ls_adj]
    
    # Final verification
    if not _is_triangle_free(best_adj) or _is_3_colorable(best_adj):
        best_adj = [a for a in core_adj]
        best_cnt = _edge_count(core_adj)
    
    # Convert to edge list
    out = []
    for u in range(N):
        m = best_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                out.append([u, v])
    return out
>>>>>>> REPLACE

I need to fix a typo in the `_local_search` function (`best_cnt_cnt` should be `best_cnt`):

