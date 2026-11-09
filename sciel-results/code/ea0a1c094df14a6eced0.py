"""C4-free graph with high minimum degree (R(C4,K1,39) lower bound)."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def _can_add_edge(adj, n, u, v):
    """Check if adding edge (u,v) creates a C4 using bitset intersection."""
    nu = adj[u]
    nv = adj[v]
    # A C4 exists if there's an edge between N(u)\{v} and N(v)\{u} connecting distinct vertices
    # i.e., exists y in N(u), x in N(v) with x != y and edge(x,y)
    temp = nu & ~nv  # vertices in N(u) but not in N(v) -- but we need edges between N(u) and N(v)
    # Better: for each y in N(u), check if adj[y] intersects N(v) \ {y}
    temp = nu
    while temp:
        y_bit = temp & (-temp)
        y = y_bit.bit_length() - 1
        temp ^= y_bit
        # Check if any neighbor of y (in current graph) is in N(v) and != y
        common = adj[y] & nv
        common &= ~(1 << y)
        if common:
            return False
    return True

def _min_degree(adj, n):
    """Compute minimum degree from adjacency bitmasks."""
    md = n
    for i in range(n):
        d = bin(adj[i]).count('1')
        if d < md:
            md = d
    return md

def _is_c4_free(adj, n):
    """Verify graph has no C4."""
    for u in range(n):
        nu = adj[u]
        temp = nu
        while temp:
            v_bit = temp & (-temp)
            v = v_bit.bit_length() - 1
            temp ^= v_bit
            if v <= u:
                continue
            # Check if adding (u,v) would create C4 (it's already there, so check the rest)
            # Actually, just verify: no two common neighbors... wait, C4 is u-x-v-y-u
            # For each pair of distinct neighbors x,y of u, check if x,y both connect to some common vertex
            # Simpler: for the existing edge (u,v), check if there exist x in N(u)\{v}, y in N(v)\{u} with edge(x,y) and x!=y
            nv = adj[v]
            temp2 = nu & ~(1 << v)
            while temp2:
                x_bit = temp2 & (-temp2)
                x = x_bit.bit_length() - 1
                temp2 ^= x_bit
                common = adj[x] & nv & ~(1 << u) & ~(1 << x)
                if common:
                    return False
    return True

def _edges_from_adj(adj, n):
    """Convert adjacency bitmasks to edge list."""
    edge_list = []
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                edge_list.append([i, j])
    return edge_list

def _find_all_c4s(adj, n):
    """Find all C4s in the graph. Returns list of (u, x, v, y) tuples."""
    c4s = []
    for u in range(n):
        nu = list(bin(adj[u]).replace('0b','').zfill(n)[::-1])
        # Get neighbors of u
        nbrs_u = []
        temp = adj[u]
        while temp:
            b = temp & (-temp)
            nbrs_u.append(b.bit_length() - 1)
            temp ^= b
        for i in range(len(nbrs_u)):
            for j in range(i + 1, len(nbrs_u)):
                x = nbrs_u[i]
                y = nbrs_u[j]
                # x and y are both neighbors of u. Check if they have a common neighbor v != u
                common = adj[x] & adj[y] & ~(1 << u)
                temp2 = common
                while temp2:
                    b = temp2 & (-temp2)
                    v = b.bit_length() - 1
                    temp2 ^= b
                    c4s.append((u, x, v, y))
    return c4s

def _break_c4s_matching(adj, n, c4s, max_iter=5000):
    """
    Use a greedy matching approach to break C4s.
    Each C4 (u,x,v,y) can be broken by removing any of its 4 edges:
    (u,x), (x,v), (v,y), (y,u).
    We want to remove a minimum set of edges to break all C4s.
    This is a hitting set problem. We use a greedy set cover approach.
    """
    if not c4s:
        return adj, 0
    
    # Build edge -> list of C4s it hits
    edge_to_c4s = {}
    c4_edges = []
    for idx, (u, x, v, y) in enumerate(c4s):
        edges_in_c4 = [(u, x), (x, v), (v, y), (y, u)]
        c4_edges.append(edges_in_c4)
        for e in edges_in_c4:
            e_key = (min(e[0], e[1]), max(e[0], e[1]))
            if e_key not in edge_to_c4s:
                edge_to_c4s[e_key] = set()
            edge_to_c4s[e_key].add(idx)
    
    # Greedy: repeatedly pick the edge that hits the most remaining C4s
    remaining = set(range(len(c4s)))
    removed_edges = set()
    
    while remaining and len(removed_edges) < n * n:
        # Find edge hitting most remaining C4s
        best_edge = None
        best_count = 0
        for e_key, c4_set in edge_to_c4s.items():
            if e_key in removed_edges:
                continue
            count = len(c4_set & remaining)
            if count > best_count:
                best_count = count
                best_edge = e_key
            elif count == best_count and best_edge is not None:
                # Tie-break: prefer edges with higher degree endpoints (to minimize degree loss)
                pass
        
        if best_edge is None or best_count == 0:
            break
        
        removed_edges.add(best_edge)
        remaining -= edge_to_c4s[best_edge]
    
    # Remove the selected edges from adj
    for (a, b) in removed_edges:
        adj[a] &= ~(1 << b)
        adj[b] &= ~(1 << a)
    
    return adj, len(removed_edges)

def _greedy_build_structured(n, rng, min_degree_target):
    """
    Build a C4-free graph on n vertices with minimum degree at least min_degree_target.
    Strategy: Start with a dense random graph, then iteratively break C4s
    using a matching/set-cover approach, prioritizing removal of edges
    that preserve minimum degree.
    """
    # Start with a random dense graph
    adj = [0] * n
    # Add edges with probability tuned to get close to target density
    target_density = min_degree_target / n
    p = min(1.0, target_density * 1.2)  # slightly above target
    
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Iteratively find and break C4s
    for iteration in range(200):
        c4s = _find_all_c4s(adj, n)
        if not c4s:
            break
        
        # Use matching to break C4s
        adj, removed = _break_c4s_matching(adj, n, c4s)
    
    md = _min_degree(adj, n)
    if md < min_degree_target:
        return None
    
    return {'n': n, 'edges': _edges_from_adj(adj, n)}

def _local_search(adj, n, min_degree_target, rng, max_iters=5000):
    """
    Local search: try to add back edges that were removed, if they don't create C4s,
    to boost minimum degree.
    """
    for _ in range(max_iters):
        # Find vertex with minimum degree
        min_deg = n
        min_v = 0
        for i in range(n):
            d = bin(adj[i]).count('1')
            if d < min_deg:
                min_deg = d
                min_v = i
        
        if min_deg >= min_degree_target:
            break
        
        # Try to add an edge from min_v to a non-neighbor
        non_nbrs = []
        for j in range(n):
            if j != min_v and not (adj[min_v] >> j) & 1:
                non_nbrs.append(j)
        
        rng.shuffle(non_nbrs)
        added = False
        for j in non_nbrs:
            if _can_add_edge(adj, n, min_v, j):
                adj[min_v] |= (1 << j)
                adj[j] |= (1 << min_v)
                added = True
                break
        
        if not added:
            # Try removing an edge from a high-degree vertex to free up C4 constraints
            # Find a high-degree vertex
            max_deg = -1
            max_v = 0
            for i in range(n):
                d = bin(adj[i]).count('1')
                if d > max_deg:
                    max_deg = d
                    max_v = i
            
            if max_deg > min_degree_target:
                # Remove a random edge from max_v
                nbrs = []
                temp = adj[max_v]
                while temp:
                    b = temp & (-temp)
                    nbrs.append(b.bit_length() - 1)
                    temp ^= b
                
                if nbrs:
                    rng.shuffle(nbrs)
                    for nb in nbrs:
                        if bin(adj[max_v]).count('1') > min_degree_target:
                            adj[max_v] &= ~(1 << nb)
                            adj[nb] &= ~(1 << max_v)
                            break
    
    md = _min_degree(adj, n)
    if md < min_degree_target:
        return None
    return {'n': n, 'edges': _edges_from_adj(adj, n)}

def _try_build(n, rng, seed_offset=0):
    """Try to build a C4-free graph on n vertices with min degree >= n-39."""
    min_degree_target = n - 39
    if min_degree_target < 1:
        min_degree_target = 1
    
    result = _greedy_build_structured(n, rng, min_degree_target)
    if result is not None:
        return result
    
    # Fallback: try local search from a different initialization
    adj = [0] * n
    p = min(1.0, min_degree_target / n * 1.3)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Break C4s
    for iteration in range(300):
        c4s = _find_all_c4s(adj, n)
        if not c4s:
            break
        adj, _ = _break_c4s_matching(adj, n, c4s)
    
    return _local_search(adj, n, min_degree_target, rng)

def construct():
    deadline = time.time() + 135.0  # safety margin under 150s
    best = None  # (n, mindeg, result)
    
    seeds = [12345, 999, 42, 20240501, 7, 123, 555, 31337, 88, 271828,
             314159, 271, 4242, 13579, 24680, 11, 22, 33, 44, 55,
             66, 77, 888, 9999, 123456, 654321, 777, 8888, 99999, 100000,
             1234567, 7654321, 31415926, 27182818, 16180339]
    
    for n in range(47, 38, -1):
        found = False
        for s in seeds:
            if time.time() > deadline:
                break
            rng = random.Random(s * 1000 + n * 7)
            res = _try_build(n, rng)
            if res is not None:
                deg = [0] * n
                for e in res['edges']:
                    deg[e[0]] += 1
                    deg[e[1]] += 1
                md = min(deg)
                
                if best is None or n > best[0] or (n == best[0] and md > best[1]):
                    best = (n, md, res)
                found = True
                if md >= n - 39 + 3:
                    break
        if found:
            break
    
    if best is not None:
        return best[2]
    
    return {'n': 39, 'edges': []}
# EVOLVE-BLOCK-END
