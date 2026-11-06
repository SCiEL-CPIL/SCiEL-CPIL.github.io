"""Construct a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph."""
import random
import itertools
from copy import deepcopy

def _is_k4_free(n, adj):
    """Check if graph with adjacency matrix adj on n vertices is K4-free."""
    for a in range(n):
        for b in range(a+1, n):
            if not adj[a][b]:
                continue
            for c in range(b+1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _max_triangle_free_induced(n, adj):
    """Exact branch and bound for largest triangle-free induced subgraph using bitmasks."""
    # Precompute neighbor bitmasks
    nmask = (1 << n) - 1
    nbr = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and adj[i][j]:
                nbr[i] |= (1 << j)
    
    # A set S is triangle-free iff for each v in S, nbr[v] & S has no edges among themselves
    # i.e., for each v in S, the subgraph induced by (nbr[v] & S) is an independent set
    # Equivalently: for each v in S, (nbr[v] & S) must be an independent set in the original graph
    # which means: for each v in S, no two vertices in (nbr[v] & S) are adjacent
    
    best = [0]
    
    def dfs(start, current_mask, current_size):
        if current_size + (n - start) <= best[0]:
            return
        if current_size > best[0]:
            best[0] = current_size
        
        for v in range(start, n):
            vbit = 1 << v
            if current_mask & vbit:
                continue
            # Check if adding v keeps triangle-free
            # v forms a triangle with u1,u2 in current if adj[u1][v] and adj[u2][v] and adj[u1][u2]
            # So: let N_v = nbr[v] & current_mask
            # For triangle-free, N_v must be an independent set in the induced subgraph
            # i.e., no two vertices in N_v are adjacent
            N_v = nbr[v] & current_mask
            # Check independence of N_v: for each u in N_v, (nbr[u] & N_v) should be 0
            # More efficiently: for each u in N_v, check if nbr[u] & N_v has any other bit
            ok = True
            temp = N_v
            while temp:
                lsb = temp & (-temp)
                u = lsb.bit_length() - 1
                if nbr[u] & N_v & ~lsb:
                    ok = False
                    break
                temp &= temp - 1
            if not ok:
                continue
            
            dfs(v + 1, current_mask | vbit, current_size + 1)
    
    dfs(0, 0, 0)
    return best[0]

def _score_graph(n, adj):
    """Return the largest triangle-free induced subgraph size."""
    return _max_triangle_free_induced(n, adj)

def _random_k4_free_graph(n, trials=2000, seed=42):
    """Generate random K4-free graphs and return the best one found."""
    rng = random.Random(seed)
    best_score = n + 1
    best_adj = None
    
    for trial in range(trials):
        adj = [[False] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i][j] = adj[j][i] = True
        
        # Remove edges to make K4-free
        changed = True
        while changed:
            changed = False
            for a in range(n):
                for b in range(a+1, n):
                    if not adj[a][b]:
                        continue
                    for c in range(b+1, n):
                        if not adj[a][c] or not adj[b][c]:
                            continue
                        for d in range(c+1, n):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                edges_in_k4 = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                                e = rng.choice(edges_in_k4)
                                adj[e[0]][e[1]] = adj[e[1]][e[0]] = False
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
                if changed:
                    break
        
        if not _is_k4_free(n, adj):
            continue
        
        score = _score_graph(n, adj)
        if score < best_score:
            best_score = score
            best_adj = [row[:] for row in adj]
            if best_score <= 7:
                break
    
    if best_adj is None:
        adj = [[False] * n for _ in range(n)]
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a][b] = adj[b][a] = True
        return adj
    
    return best_adj

def _local_search(n, adj, max_iters=5000, seed=42):
    """Local search: flip edges to minimize score while maintaining K4-free."""
    rng = random.Random(seed)
    best_adj = [row[:] for row in adj]
    best_score = _score_graph(n, best_adj)
    
    for it in range(max_iters):
        # Pick a random edge to flip
        i = rng.randint(0, n-1)
        j = rng.randint(0, n-1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Try flipping
        new_adj = [row[:] for row in best_adj]
        new_adj[i][j] = new_adj[j][i] = not new_adj[i][j]
        
        # Check K4-free
        if not _is_k4_free(n, new_adj):
            continue
        
        new_score = _score_graph(n, new_adj)
        
        # Accept if better or equal (with some randomness to escape plateaus)
        if new_score < best_score or (new_score == best_score and rng.random() < 0.1):
            best_adj = new_adj
            best_score = new_score
            if best_score <= 7:
                break
    
    return best_adj, best_score

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph."""
    n = 18
    
    # Start with several initial candidates
    candidates = []
    
    # 1. Complete tripartite K6,6,6
    adj1 = [[False] * n for _ in range(n)]
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj1[a][b] = adj1[b][a] = True
    candidates.append(adj1)
    
    # 2. Complete 4-partite 5,4,4,5
    adj2 = [[False] * n for _ in range(n)]
    sizes = [5, 4, 4, 5]
    start = 0
    parts2 = []
    for s in sizes:
        parts2.append(list(range(start, start+s)))
        start += s
    for i in range(4):
        for j in range(i+1, 4):
            for a in parts2[i]:
                for b in parts2[j]:
                    adj2[a][b] = adj2[b][a] = True
    candidates.append(adj2)
    
    # 3. Random K4-free graphs
    random.seed(42)
    adj3 = _random_k4_free_graph(n, trials=1500, seed=42)
    candidates.append(adj3)
    
    # Evaluate all candidates
    best_adj = None
    best_score = n + 1
    for adj in candidates:
        if not _is_k4_free(n, adj):
            continue
        score = _score_graph(n, adj)
        if score < best_score:
            best_score = score
            best_adj = [row[:] for row in adj]
        if best_score <= 7:
            break
    
    if best_adj is None:
        best_adj = adj1
        best_score = _score_graph(n, best_adj)
    
    # Local search from best candidate
    improved_adj, improved_score = _local_search(n, best_adj, max_iters=8000, seed=42)
    if improved_score < best_score:
        best_adj = improved_adj
        best_score = improved_score
    
    # Additional random restarts with local search
    for seed in [123, 456, 789]:
        adj_r = _random_k4_free_graph(n, trials=300, seed=seed)
        if not _is_k4_free(n, adj_r):
            continue
        score_r = _score_graph(n, adj_r)
        if score_r <= best_score:
            improved_adj, improved_score = _local_search(n, adj_r, max_iters=3000, seed=seed)
            if improved_score < best_score:
                best_adj = improved_adj
                best_score = improved_score
        if best_score <= 7:
            break
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
