"""C6-free graph construction on 40 vertices via randomized local search."""
import time
import random
import itertools
import numpy as np

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 115.0  # safety margin before 120s limit


def _has_c6(adj, n):
    """Check if adjacency matrix (n x n boolean) contains a C6.
    Returns True if a 6-cycle exists."""
    # A C6 is a cycle of length 6: v0-v1-v2-v3-v4-v5-v0
    # We check for cycles of length exactly 6.
    # Use the fact that C6 exists iff there's a path of length 5 from v0 to v0
    # with all distinct vertices.
    
    # Efficient check: for each pair of vertices at distance 3 in the cycle,
    # we need two common neighbors at distance 1 from each endpoint.
    # Actually, let's use a more direct approach.
    
    # A 6-cycle v0,v1,v2,v3,v4,v5 exists iff:
    # adj[v0][v1] & adj[v1][v2] & adj[v2][v3] & adj[v3][v4] & adj[v4][v5] & adj[v5][v0]
    # and all vi are distinct.
    
    # For efficiency, we can iterate over pairs (v0, v3) which are opposite in the cycle,
    # and check if there exist v1, v2, v4, v5 such that the cycle closes.
    
    # Alternative: enumerate all 6-tuples. For n=40, that's too many.
    
    # Better approach: For each ordered pair (a, b) with a != b,
    # count the number of length-3 paths from a to b. If any pair has
    # at least 2 such paths that are vertex-disjoint (except endpoints),
    # we have a C6. Actually that's not quite right either.
    
    # Let me think more carefully. A C6 can be decomposed into two paths of length 3
    # between the same pair of vertices (v0 and v3), where the two paths share only
    # the endpoints.
    
    # So: for each pair (v0, v3), find all length-3 paths from v0 to v3.
    # If there exist two such paths that are internally vertex-disjoint, we have a C6.
    
    # For n=40, let's precompute neighbors.
    nbrs = [set(i for i in range(n) if adj[i][j]) for j in range(n)]
    
    for v0 in range(n):
        for v3 in range(v0 + 1, n):
            # Find all length-3 paths from v0 to v3: v0-a-b-v3
            # where a is neighbor of v0, b is neighbor of v3, and a-b is an edge.
            paths = []
            for a in nbrs[v0]:
                if a == v3:
                    continue
                for b in nbrs[v3]:
                    if b == v0 or b == a:
                        continue
                    if adj[a][b]:
                        paths.append((a, b))
            # Check if any two paths are internally vertex-disjoint
            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    a1, b1 = paths[i]
                    a2, b2 = paths[j]
                    # Internal vertices are a and b for each path
                    # They must be disjoint: {a1, b1} ∩ {a2, b2} = empty
                    if a1 != a2 and a1 != b2 and b1 != a2 and b1 != b2:
                        return True
    return False


def _has_c6_fast(adj, n):
    """Faster C6 check using bitsets."""
    # Represent neighbor sets as integers for fast intersection
    nbr_bits = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        nbr_bits[i] = mask
    
    for v0 in range(n):
        for v3 in range(v0 + 1, n):
            # Length-3 paths from v0 to v3: v0-a-b-v3
            # a in N(v0), b in N(v3), edge(a,b)
            # Collect all internal vertex pairs (a, b)
            internal_pairs = []
            na = nbr_bits[v0]
            nb = nbr_bits[v3]
            # For each a in N(v0) \ {v3}, for each b in N(v3) \ {v0, a}, check edge(a,b)
            a_mask = na & ~(1 << v3)
            while a_mask:
                lb = a_mask & (-a_mask)
                a = lb.bit_length() - 1
                a_mask ^= lb
                b_mask = nb & ~(1 << v0) & ~(1 << a)
                while b_mask:
                    lb2 = b_mask & (-b_mask)
                    b = lb2.bit_length() - 1
                    b_mask ^= lb2
                    if adj[a][b]:
                        internal_pairs.append((a, b))
            # Check for two disjoint internal pairs
            m = len(internal_pairs)
            for i in range(m):
                a1, b1 = internal_pairs[i]
                for j in range(i + 1, m):
                    a2, b2 = internal_pairs[j]
                    if a1 != a2 and a1 != b2 and b1 != a2 and b1 != b2:
                        return True
    return False


def _count_edges(adj, n):
    return int(np.sum(adj)) // 2


def _random_regular_graph(n, d, seed):
    """Generate a random d-regular graph on n vertices using configuration model + local fixup."""
    rng = random.Random(seed)
    # Configuration model
    stubs = []
    for v in range(n):
        stubs.extend([v] * d)
    rng.shuffle(stubs)
    
    adj = np.zeros((n, n), dtype=bool)
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u != v:
            adj[u][v] = True
            adj[v][u] = True
    
    # Fix degree issues by swapping edges
    deg = adj.sum(axis=1).astype(int)
    for _ in range(1000):
        over = [i for i in range(n) if deg[i] > d]
        under = [i for i in range(n) if deg[i] < d]
        if not over or not under:
            break
        u = rng.choice(over)
        v = rng.choice(under)
        # Find a neighbor w of u such that v is not adjacent to w
        neighbors_u = [x for x in range(n) if adj[u][x] and x != v]
        if not neighbors_u:
            continue
        w = rng.choice(neighbors_u)
        if not adj[v][w]:
            adj[u][w] = False
            adj[w][u] = False
            adj[v][w] = True
            adj[w][v] = True
            deg[u] -= 1
            deg[w] -= 1
            deg[v] += 1
            deg[w] += 1  # wait, w loses one from u but gains one from v
            # Actually: u loses edge to w, v gains edge to w
            # deg[u] -= 1, deg[v] += 1, deg[w] unchanged (loses from u, gains from v)
            deg[u] -= 1
            deg[v] += 1
            # deg[w] stays same
            deg[w]  # no change
    return adj


def _local_search(adj, n, max_iter, rng):
    """Local search: try adding edges that don't create C6, and if stuck, remove edges."""
    best_adj = adj.copy()
    best_m = _count_edges(best_adj, n)
    
    for iteration in range(max_iter):
        if time.time() - _START_TIME > _DEADLINE:
            break
        
        current_m = _count_edges(adj, n)
        
        # Try to add random non-edges
        added = False
        # Collect non-edges
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not adj[i][j]:
                    non_edges.append((i, j))
        
        rng.shuffle(non_edges)
        
        for (i, j) in non_edges:
            if time.time() - _START_TIME > _DEADLINE:
                break
            # Try adding edge (i, j)
            adj[i][j] = True
            adj[j][i] = True
            if not _has_c6_fast(adj, n):
                current_m += 1
                added = True
            else:
                adj[i][j] = False
                adj[j][i] = False
        
        if not added:
            # Stuck: try removing a random edge and re-adding
            edges_list = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i][j]]
            if edges_list:
                (ri, rj) = rng.choice(edges_list)
                adj[ri][rj] = False
                adj[rj][ri] = False
                current_m -= 1
                # Now try to add edges again
                non_edges2 = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not adj[i][j]:
                            non_edges2.append((i, j))
                rng.shuffle(non_edges2)
                for (i, j) in non_edges2:
                    if time.time() - _START_TIME > _DEADLINE:
                        break
                    adj[i][j] = True
                    adj[j][i] = True
                    if not _has_c6_fast(adj, n):
                        current_m += 1
                    else:
                        adj[i][j] = False
                        adj[j][i] = False
        
        if current_m > best_m:
            best_m = current_m
            best_adj = adj.copy()
    
    return best_adj, best_m


def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = 115.0
    
    n = 40
    best_edges = None
    best_m = 0
    
    # Start with the known 105-edge baseline: 8 copies of K5
    base_adj = np.zeros((n, n), dtype=bool)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                base_adj[u][v] = True
                base_adj[v][u] = True
    base_m = _count_edges(base_adj, n)  # should be 80
    if base_m > best_m:
        best_m = base_m
        best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if base_adj[u][v]]
    
    # Try multiple random regular graph starts and local search
    seeds = list(range(50))
    rng = random.Random(42)
    rng.shuffle(seeds)
    
    for seed in seeds:
        if time.time() - _START_TIME > _DEADLINE:
            break
        
        # Try different target degrees
        for d in [5, 6]:
            if time.time() - _START_TIME > _DEADLINE:
                break
            
            adj = _random_regular_graph(n, d, seed)
            # Check if it's C6-free
            if _has_c6_fast(adj, n):
                continue
            
            m = _count_edges(adj, n)
            if m > best_m:
                best_m = m
                best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if adj[u][v]]
            
            # Local search to improve
            local_rng = random.Random(seed * 1000 + d)
            adj_opt, m_opt = _local_search(adj, n, max_iter=200, rng=local_rng)
            
            if m_opt > best_m:
                best_m = m_opt
                best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if adj_opt[u][v]]
    
    # Also try starting from the K5-disjoint graph and doing local search
    if time.time() - _START_TIME < _DEADLINE - 10:
        local_rng = random.Random(999)
        adj_opt, m_opt = _local_search(base_adj.copy(), n, max_iter=800, rng=local_rng)
        if m_opt > best_m:
            best_m = m_opt
            best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if adj_opt[u][v]]
    
    # Final polish: try to add any remaining edges to the best solution
    if best_edges is not None and time.time() - _START_TIME < _DEADLINE - 5:
        final_adj = np.zeros((n, n), dtype=bool)
        for u, v in best_edges:
            final_adj[u][v] = True
            final_adj[v][u] = True
        final_m = _count_edges(final_adj, n)
        
        # Try all non-edges one more time
        for i in range(n):
            for j in range(i + 1, n):
                if not final_adj[i][j]:
                    final_adj[i][j] = True
                    final_adj[j][i] = True
                    if not _has_c6_fast(final_adj, n):
                        final_m += 1
                    else:
                        final_adj[i][j] = False
                        final_adj[j][i] = False
        
        if final_m > best_m:
            best_m = final_m
            best_edges = [[u, v] for u in range(n) for v in range(u + 1, n) if final_adj[u][v]]
    
    if best_edges is None:
        # Fallback: return the K5-disjoint graph
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges
    
    return best_edges


if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
