"""Minimize the largest book (max triangles per edge) on 40 vertices, 401 edges."""
import random
import time

N = 40
M = 401


def _book_scores(adj):
    """Return dict: (u,v) -> number of triangles containing edge (u,v)."""
    scores = {}
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u][v]:
                cnt = 0
            for w in range(N):
                if w != u and w != v and adj[u][v] and adj[u][w] and adj[v][w]:
                    pass  # placeholder
    # Simpler: iterate all triangles
    scores = {}
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u][v]:
                c = 0
                for w in range(N):
                    if w > v and adj[u][w] and adj[v][w]:
                        c += 1
                scores[(u, v)] = c
    return scores


def _max_book(adj):
    """Return the maximum book size."""
    best = 0
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u][v]:
                c = 0
                for w in range(N):
                    if w > v and adj[u][w] and adj[v][w]:
                        c += 1
                if c > best:
                    best = c
    return best


def _edge_count(adj):
    cnt = 0
    for u in range(N):
        for v in range(u + 1, N):
            if adj[u][v]:
                cnt += 1
    return cnt


def _make_initial():
    """T_{20,20} plus one intra-part edge, then try to reduce book size."""
    adj = [[False] * N for _ in range(N)]
    for a in range(20):
        for b in range(20, 40):
            adj[a][b] = adj[b][a] = True
    # Add one edge within A: (0,1)
    adj[0][1] = adj[1][0] = True
    return adj


def construct():
    deadline = time.time() + 140  # safety margin before 150s
    best_adj = _make_initial()
    best_score = _max_book(best_adj)
    
    # Local search: try to reduce max book
    # Strategy: start from T_{20,20}+(0,1), then do random edge swaps
    # to break up the large book.
    
    rng = random.Random(42)
    adj = [row[:] for row in best_adj]
    score = best_score
    
    # Try many random restarts with different initial edges
    # and local search from each
    
    iterations = 0
    while time.time() < deadline:
        # Occasionally restart with a fresh random perturbation
        if iterations % 500 == 0 and iterations > 0:
            # Perturb: remove 5 random edges, add 5 random non-edges
            adj2 = [row[:] for row in adj]
            edges = [(u, v) for u in range(N) for v in range(u + 1, N) if adj2[u][v]]
            non_edges = [(u, v) for u in range(N) for v in range(u + 1, N) if not adj2[u][v]]
            
            for _ in range(5):
                if edges and non_edges:
                    u, v = rng.choice(edges)
                    adj2[u][v] = adj2[v][u] = False
                    edges.remove((u, v))
                    u2, v2 = rng.choice(non_edges)
                    adj2[u2][v2] = adj2[v2][u2] = True
                    non_edges.remove((u2, v2))
            
            s2 = _max_book(adj2)
            if s2 <= score:
                adj = [row[:] for row in adj2]
                score = s2
                if score < best_score:
                    best_adj = [row[:] for row in adj]
                    best_score = score
        
        # Single edge swap
        edges = [(u, v) for u in range(N) for v in range(u + 1, N) if adj[u][v]]
        non_edges = [(u, v) for u in range(N) for v in range(u + 1, N) if not adj[u][v]]
        
        if not edges or not non_edges:
            iterations += 1
            continue
            
        u, v = rng.choice(edges)
        u2, v2 = rng.choice(non_edges)
        
        # Apply swap
        adj[u][v] = adj[v][u] = False
        adj[u2][v2] = adj[v2][u2] = True
        
        new_score = _max_book(adj)
        
        if new_score <= score:
            score = new_score
            if score < best_score:
                best_adj = [row[:] for row in adj]
                best_score = score
        else:
            # Revert
            adj[u][v] = adj[v][u] = True
            adj[u2][v2] = adj[v2][u2] = False
        
        iterations += 1
    
    # Build edge list from best_adj
    result = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_adj[u][v]:
                result.append([u, v])
    return result
