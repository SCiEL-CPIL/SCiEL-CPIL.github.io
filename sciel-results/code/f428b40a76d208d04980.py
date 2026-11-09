"""K4-free graph on 18 vertices minimizing largest triangle-free induced subgraph."""
import time
import random
import itertools

N = 18
DEADLINE = 130.0  # safety margin before 150s kill

def _adj_matrix_from_edges(edges):
    adj = [[0]*N for _ in range(N)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    return adj

def _is_k4_free(adj):
    # Check all 4-cliques
    for a in range(N):
        for b in range(a+1, N):
            if not adj[a][b]:
                continue
            for c in range(b+1, N):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c+1, N):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _is_triangle_free_induced(vertices, adj):
    """Check if the induced subgraph on 'vertices' is triangle-free."""
    vset = set(vertices)
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            u, v = vertices[i], vertices[j]
            if not adj[u][v]:
                continue
            for w in vertices:
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    return False
    return True

def _max_triangle_free_size(adj):
    """Find the maximum size of a triangle-free induced subgraph."""
    best = 0
    # Use branch and bound / backtracking
    # We try to find a large triangle-free induced subgraph.
    # A triangle-free induced subgraph means no 3 vertices form a triangle.
    # Equivalently, the complement of the induced subgraph has no K3, i.e., it's triangle-free in original.
    
    # Better approach: try all subsets of size s from high to low, check if triangle-free.
    # But that's expensive. Let's use a smarter approach.
    
    # Actually, let's just iterate over all subsets by size from 9 down to 1
    # (since we expect the answer to be around 9-10 based on the knowledge)
    # For each size s, check if any s-subset is triangle-free.
    
    # Optimization: if s <= 2, always triangle-free
    if N <= 2:
        return N
    
    # Try sizes from min(N, 12) down to 1
    for s in range(min(N, 14), 1, -1):
        found = False
        for subset in itertools.combinations(range(N), s):
            if _is_triangle_free_induced(subset, adj):
                return s
        # If we didn't find any of size s, continue to s-1
    return 2  # minimum possible for N >= 2

def _max_triangle_free_size_fast(adj, max_check=10):
    """Faster version: only check sizes from max_check down to 1."""
    for s in range(min(N, max_check), 1, -1):
        for subset in itertools.combinations(range(N), s):
            if _is_triangle_free_induced(subset, adj):
                return s
    return 2

def _score(adj):
    """Return the largest triangle-free induced subgraph size."""
    return _max_triangle_free_size(adj)

def construct():
    start_time = time.time()
    random.seed(42)
    
    best_score = None
    best_edges = None
    
    # Start with a reasonable initial graph
    # Complete tripartite K6,6,6 gives score 12
    # Let's try to improve via local search
    
    # Initial: complete tripartite
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = [[a, b] for i in range(3) for j in range(i+1, 3) for a in parts[i] for b in parts[j]]
    adj = _adj_matrix_from_edges(edges)
    
    # Verify K4-free
    assert _is_k4_free(adj), "Initial graph is not K4-free"
    
    cur_score = _score(adj)
    best_score = cur_score
    best_edges = list(edges)
    
    # Local search: try adding/removing edges while maintaining K4-free
    # and minimizing the largest triangle-free induced subgraph
    
    def try_edge_addition(adj, edges):
        """Try to add an edge; return (new_adj, new_edges, new_score) if improves."""
        nonlocal best_score
        # Find non-edges
        candidates = []
        for u in range(N):
            for v in range(u+1, N):
                if not adj[u][v]:
                    candidates.append((u, v))
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() - start_time > DEADLINE - 10:
                break
            # Check if adding edge creates a K4
            # A K4 containing edge (u,v) needs two other vertices x,y such that
            # all of u,v,x,y are mutually connected
            creates_k4 = False
            for x in range(N):
                if x == u or x == v:
                    continue
                if not adj[u][x] or not adj[v][x]:
                    continue
                for y in range(x+1, N):
                    if y == u or y == v:
                        continue
                    if adj[u][y] and adj[v][y] and adj[x][y]:
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            if creates_k4:
                continue
            
            # Try adding the edge
            new_adj = [row[:] for row in adj]
            new_adj[u][v] = 1
            new_adj[v][u] = 1
            new_edges = edges + [[u, v]]
            
            # Quick score check - only check if it could improve
            # The score can only stay same or decrease when adding edges
            # (adding edges makes it harder to find triangle-free subgraphs)
            new_score = _max_triangle_free_size_fast(new_adj, max_check=best_score)
            
            if new_score < best_score:
                return new_adj, new_edges, new_score
            elif new_score < cur_score:
                return new_adj, new_edges, new_score
        return None
    
    def try_edge_removal(adj, edges):
        """Try to remove an edge; may increase score, so only do if we're stuck."""
        nonlocal best_score
        candidates = list(edges)
        random.shuffle(candidates)
        
        for u, v in candidates:
            if time.time() - start_time > DEADLINE - 10:
                break
            new_adj = [row[:] for row in adj]
            new_adj[u][v] = 0
            new_adj[v][u] = 0
            new_edges = [e for e in edges if not (e[0] == u and e[1] == v) and not (e[0] == v and e[1] == u)]
            
            new_score = _max_triangle_free_size_fast(new_adj, max_check=best_score + 2)
            
            if new_score < best_score:
                return new_adj, new_edges, new_score
        return None
    
    # Main search loop
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < DEADLINE - 15:
        # Try adding edges (preferable, as it can only help)
        result = try_edge_addition(adj, best_edges if best_score == cur_score else list(best_edges))
        if result is None:
            # Use current adj and edges
            if best_score == cur_score:
                result = try_edge_addition(adj, list(best_edges))
        
        if result is not None:
            new_adj, new_edges, new_score = result
            if new_score < best_score:
                best_score = new_score
                best_edges = new_edges
                adj = new_adj
                cur_score = new_score
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Try removals or restart
            no_improve_count += 1
        
        if no_improve_count >= max_no_improve:
            # Restart with a different random initial graph
            no_improve_count = 0
            # Generate a random K4-free graph
            # Use a random 3-coloring and make it complete tripartite with some modifications
            coloring = [random.randint(0, 2) for _ in range(N)]
            # Ensure all 3 colors are used
            while len(set(coloring)) < 3:
                coloring[random.randint(0, N-1)] = random.randint(0, 2)
            
            edges = []
            for u in range(N):
                for v in range(u+1, N):
                    if coloring[u] != coloring[v]:
                        edges.append([u, v])
            
            adj = _adj_matrix_from_edges(edges)
            if _is_k4_free(adj):
                cur_score = _score(adj)
                if cur_score < best_score:
                    best_score = cur_score
                    best_edges = list(edges)
    
    return best_edges

# Quick self-test
if __name__ == "__main__":
    edges = construct()
    adj = _adj_matrix_from_edges(edges)
    print(f"K4-free: {_is_k4_free(adj)}")
    print(f"Score: {_score(adj)}")
    print(f"Number of edges: {len(edges)}")
