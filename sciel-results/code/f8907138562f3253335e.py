"""Optimized graph construction using bitset representation and simulated annealing."""
import random
import time
import sys

def construct():
    """Find a 16-vertex, 65-edge graph minimizing max triangle degree sum."""
    deadline = time.time() + 130  # Safety margin before 150s kill
    N = 16
    M = 65
    rng = random.Random(42)

    # Bitset representation: adj[i] is a bitmask of neighbors of vertex i
    def init_graph():
        """K_{8,8} + edge (0,1)"""
        adj = [0] * N
        for u in range(8):
            for v in range(8, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        adj[0] |= (1 << 1)
        adj[1] |= (1 << 0)
        return adj

    def get_edges(adj):
        edges = []
        for u in range(N):
            mask = adj[u]
            while mask:
                v = (mask & (-mask)).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    edges.append([u, v])
        return edges

    def compute_score(adj):
        """Max degree sum over all triangles."""
        deg = [adj[i].bit_count() for i in range(N)]
        best = 0
        for u in range(N):
            mu = adj[u]
            for v in range(u + 1, N):
                if not (mu >> v) & 1:
                    continue
                common = mu & adj[v]
                while common:
                    w = (common & (-common)).bit_length() - 1
                    common &= common - 1
                    if w > v:
                        s = deg[u] + deg[v] + deg[w]
                        if s > best:
                            best = s
        return best

    # Precompute triangle list for faster delta computation
    def compute_triangles(adj):
        """Return list of (u, v, w) triangles."""
        tris = []
        for u in range(N):
            mu = adj[u]
            for v in range(u + 1, N):
                if not (mu >> v) & 1:
                    continue
                common = mu & adj[v]
                while common:
                    w = (common & (-common)).bit_length() - 1
                    common &= common - 1
                    if w > v:
                        tris.append((u, v, w))
        return tris

    def triangle_degree_sum(tri, deg):
        u, v, w = tri
        return deg[u] + deg[v] + deg[w]

    # Edge swap: remove (a,b), add (c,d) where a,c in same side, b,d in other
    def try_swap(adj, tris, deg, a, b, c, d):
        """Swap edge (a,b) with (c,d). Returns (new_adj, new_deg, new_tris, new_score) or None."""
        # Check (a,b) exists and (c,d) doesn't
        if not ((adj[a] >> b) & 1):
            return None
        if (adj[c] >> d) & 1:
            return None
        # a,c must be on same side, b,d on other (for valid bipartite swap)
        # Actually, we just need the swap to be valid: remove (a,b), add (c,d)
        # New edges: remove (a,b), add (c,d)
        # Degree changes: a-1, b-1, c+1, d+1 (if a!=c, b!=d, etc.)
        
        new_adj = adj[:]
        # Remove (a,b)
        new_adj[a] &= ~(1 << b)
        new_adj[b] &= ~(1 << a)
        # Add (c,d)
        new_adj[c] |= (1 << d)
        new_adj[d] |= (1 << c)
        
        new_deg = deg[:]
        new_deg[a] -= 1
        new_deg[b] -= 1
        new_deg[c] += 1
        new_deg[d] += 1
        
        # Compute new score from triangles
        # Triangles affected: those containing a or b (lost) or c or d (gained)
        # Simplest: recompute all triangles
        new_tris = compute_triangles(new_adj)
        new_score = 0
        for tri in new_tris:
            s = triangle_degree_sum(tri, new_deg)
            if s > new_score:
                new_score = s
        
        return new_adj, new_deg, new_tris, new_score

    def local_search(adj, deg, tris, score, time_limit):
        """Greedy local search: try all swaps, take best improvement."""
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            best_delta = 0
            best_swap = None
            # Try all possible swaps: remove edge (a,b), add edge (c,d)
            # where {a,b} != {c,d} and (c,d) is not an edge
            edges = []
            for u in range(N):
                mask = adj[u]
                while mask:
                    v = (mask & (-mask)).bit_length() - 1
                    mask &= mask - 1
                    if u < v:
                        edges.append((u, v))
            
            for (a, b) in edges:
                for c in range(N):
                    if c == a or c == b:
                        continue
                    for d in range(c + 1, N):
                        if d == a or d == b:
                            continue
                        if (c, d) == (a, b) or (c, d) == (b, a):
                            continue
                        # Skip if (c,d) already an edge
                        if (adj[c] >> d) & 1:
                            continue
                        # Also skip if (a,b) and (c,d) share a vertex in a way that
                        # creates a multigraph - but since we check (c,d) not edge, ok
                        result = try_swap(adj, tris, deg, a, b, c, d)
                        if result is None:
                            continue
                        new_adj, new_deg, new_tris, new_score = result
                        delta = score - new_score
                        if delta > best_delta:
                            best_delta = delta
                            best_swap = (a, b, c, d, new_adj, new_deg, new_tris, new_score)
            if best_swap is not None:
                a, b, c, d, adj, deg, tris, score = best_swap
                improved = True
        return adj, deg, tris, score

    def simulated_annealing(adj, deg, tris, score, time_limit):
        """SA with strict edge swaps."""
        T = 5.0
        T_min = 0.01
        cooling = 0.999
        best_adj = adj[:]
        best_score = score
        best_deg = deg[:]
        best_tris = tris[:]
        
        # Precompute all possible swap pairs for speed
        # Edge (a,b) to remove, edge (c,d) to add
        edges = []
        for u in range(N):
            mask = adj[u]
            while mask:
                v = (mask & (-mask)).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    edges.append((u, v))
        
        all_swaps = []
        for (a, b) in edges:
            for c in range(N):
                if c == a or c == b:
                    continue
                for d in range(c + 1, N):
                    if d == a or d == b:
                        continue
                    if (c, d) == (a, b) or (c, d) == (b, a):
                        continue
                    if (adj[c] >> d) & 1:
                        continue
                    all_swaps.append((a, b, c, d))
        
        iter_count = 0
        while T > T_min:
            if time.time() > time_limit:
                break
            # Do several iterations per temperature
            iters = min(100, max(10, int(1000 * T)))
            for _ in range(iters):
                if time.time() > time_limit:
                    break
                iter_count += 1
                # Pick random swap
                idx = rng.randrange(len(all_swaps))
                a, b, c, d = all_swaps[idx]
                # Check if swap is still valid
                if not ((adj[a] >> b) & 1):
                    continue
                if (adj[c] >> d) & 1:
                    continue
                
                result = try_swap(adj, tris, deg, a, b, c, d)
                if result is None:
                    continue
                new_adj, new_deg, new_tris, new_score = result
                delta = score - new_score  # positive = improvement
                
                if delta > 0 or rng.random() < pow(2.718281828, delta / T):
                    adj, deg, tris, score = new_adj, new_deg, new_tris, new_score
                    if score < best_score:
                        best_score = score
                        best_adj = adj[:]
                        best_deg = deg[:]
                        best_tris = tris[:]
            T *= cooling
        
        return best_adj, best_score

    # Main
    adj = init_graph()
    deg = [adj[i].bit_count() for i in range(N)]
    tris = compute_triangles(adj)
    score = compute_score(adj)
    
    # Phase 1: Local search
    time_limit = time.time() + 30
    adj, deg, tris, score = local_search(adj, deg, tris, score, time_limit)
    
    # Phase 2: Simulated annealing with restarts
    best_adj = adj[:]
    best_score = score
    best_deg = deg[:]
    best_tris = tris[:]
    
    while time.time() < deadline - 10:
        # SA from current best
        adj_sa, score_sa = simulated_annealing(adj, deg, tris, score, time.time() + 20)
        if score_sa < best_score:
            best_score = score_sa
            best_adj = adj_sa
            # Recompute deg and tris
            best_deg = [best_adj[i].bit_count() for i in range(N)]
            best_tris = compute_triangles(best_adj)
        
        # Occasionally restart from a perturbed version
        if rng.random() < 0.3:
            # Perturb: do a few random swaps
            adj_p = best_adj[:]
            deg_p = [adj_p[i].bit_count() for i in range(N)]
            tris_p = compute_triangles(adj_p)
            score_p = compute_score(adj_p)
            for _ in range(5):
                edges = []
                for u in range(N):
                    mask = adj_p[u]
                    while mask:
                        v = (mask & (-mask)).bit_length() - 1
                        mask &= mask - 1
                        if u < v:
                            edges.append((u, v))
                if not edges:
                    break
                a, b = rng.choice(edges)
                c = rng.randrange(N)
                d = rng.randrange(N)
                if c == a or c == b or d == a or d == b:
                    continue
                if c == d:
                    continue
                if (adj_p[c] >> d) & 1:
                    continue
                result = try_swap(adj_p, tris_p, deg_p, a, b, c, d)
                if result is not None:
                    adj_p, deg_p, tris_p, score_p = result
            # SA from perturbed
            adj_sa2, score_sa2 = simulated_annealing(adj_p, deg_p, tris_p, score_p, time.time() + 15)
            if score_sa2 < best_score:
                best_score = score_sa2
                best_adj = adj_sa2
                best_deg = [best_adj[i].bit_count() for i in range(N)]
                best_tris = compute_triangles(best_adj)
    
    # Final local search on best
    if time.time() < deadline - 5:
        adj_final, deg_final, tris_final, score_final = local_search(
            best_adj, best_deg, best_tris, best_score, deadline - 1)
        if score_final < best_score:
            best_adj = adj_final
            best_score = score_final
    
    edges = get_edges(best_adj)
    assert len(edges) == M, f"Expected {M} edges, got {len(edges)}"
    
    return {'n': N, 'edges': edges}

if __name__ == '__main__':
    result = construct()
    print(f"Edges: {len(result['edges'])}")
