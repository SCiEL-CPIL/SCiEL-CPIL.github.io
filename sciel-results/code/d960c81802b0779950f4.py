import random
import time
import sys

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    n = 18
    # Initialize adjacency matrix
    # adj[i][j] = 1 if edge exists
    
    def is_k4_free(adj):
        # Check for K4 in O(n^4) or faster
        # For n=18, n^4 is ~100k, very fast.
        for a in range(n):
            for b in range(a+1, n):
                if adj[a][b] == 0: continue
                for c in range(b+1, n):
                    if adj[a][c] == 0 or adj[b][c] == 0: continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True

    def can_add_edge(adj, u, v):
        # Check if adding (u,v) creates a K4
        # A K4 is formed if there are w, x in N(u) cap N(v) such that w-x is an edge.
        # Or more simply, if there is a triangle u-w-x and v is connected to w and x.
        # Let's check all pairs w, x in N(u) cap N(v).
        common = []
        for k in range(n):
            if k != u and k != v and adj[u][k] and adj[v][k]:
                common.append(k)
        for i in range(len(common)):
            for j in range(i+1, len(common)):
                w = common[i]
                x = common[j]
                if adj[w][x]:
                    return False
        return True

    def max_triangle_free_size(adj):
        # Heuristic to find largest triangle-free induced subgraph
        # Start with all vertices
        S = list(range(n))
        # Calculate triangles in G[S]
        # To speed up, we can just iterate.
        # Greedy removal
        # Precompute triangles? 
        # For n=18, we can just check all triples in S.
        # But S shrinks.
        
        # Let's use a simple greedy:
        # Find a triangle, remove the vertex with highest degree in the triangle hypergraph?
        # Or just remove a random one?
        # Better: Remove the vertex that is part of the most triangles in S.
        
        # To be faster, let's just do a greedy independent set on the "triangle conflict" graph?
        # No, let's stick to simple backtracking or greedy.
        # Given n=18, a simple recursive search with memoization or pruning might be okay if we don't do it too often.
        # But local search might call this 1000s of times.
        # Let's optimize the heuristic.
        
        # Heuristic:
        # 1. Start with S = all vertices.
        # 2. While there is a triangle in S:
        #    Find all triangles in S.
        #    Pick a vertex v in S that is in the most triangles.
        #    Remove v from S.
        # 3. Return len(S).
        
        # Implementation:
        # We can maintain a list of active vertices.
        # To find triangles efficiently, we can iterate over edges.
        
        active = [True] * n
        
        while True:
            # Find a triangle
            tri_found = False
            # Count triangles per vertex
            tri_count = [0] * n
            
            # Iterate over all triples? 18 choose 3 = 816. Very fast.
            for i in range(n):
                if not active[i]: continue
                for j in range(i+1, n):
                    if not active[j]: continue
                    if adj[i][j] == 0: continue
                    for k in range(j+1, n):
                        if not active[k]: continue
                        if adj[i][k] and adj[j][k]:
                            # Triangle i-j-k
                            tri_found = True
                            tri_count[i] += 1
                            tri_count[j] += 1
                            tri_count[k] += 1
            
            if not tri_found:
                break
            
            # Find vertex with max tri_count among active
            max_count = -1
            v_to_remove = -1
            for i in range(n):
                if active[i] and tri_count[i] > max_count:
                    max_count = tri_count[i]
                    v_to_remove = i
            
            if v_to_remove == -1:
                # Should not happen if tri_found
                break
                
            active[v_to_remove] = False
            
        return sum(active)

    def random_k4_free_graph():
        adj = [[0]*n for _ in range(n)]
        # Random edges
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1
                    adj[j][i] = 1
        # Remove edges to break K4s
        # Loop until no K4
        # Since we just need a valid start, we can do this.
        changed = True
        while changed:
            changed = False
            for a in range(n):
                if changed: break
                for b in range(a+1, n):
                    if changed: break
                    if adj[a][b] == 0: continue
                    for c in range(b+1, n):
                        if changed: break
                        if adj[a][c] == 0 or adj[b][c] == 0: continue
                        for d in range(c+1, n):
                            if adj[a][d] and adj[b][d] and adj[c][d]:
                                # Found K4 a,b,c,d. Remove a random edge.
                                # Let's remove (c,d)
                                adj[c][d] = 0
                                adj[d][c] = 0
                                changed = True
                                break
        return adj

    best_adj = None
    best_size = n + 1
    
    # Initial random graph
    adj = random_k4_free_graph()
    current_size = max_triangle_free_size(adj)
    if current_size < best_size:
        best_size = current_size
        best_adj = [row[:] for row in adj]
        
    # Local Search
    # We want to add edges to create more triangles, reducing the max triangle-free set size.
    # So we try to add non-edges.
    
    # Precompute list of non-edges?
    # Or just iterate all pairs.
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < timeout:
        improved = False
        
        # Try to add a random edge
        # Shuffle pairs to avoid bias
        pairs = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j] == 0]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if time.time() - start_time > timeout:
                break
                
            if can_add_edge(adj, u, v):
                # Try adding
                adj[u][v] = 1
                adj[v][u] = 1
                
                new_size = max_triangle_free_size(adj)
                
                if new_size < current_size:
                    current_size = new_size
                    improved = True
                    if current_size < best_size:
                        best_size = current_size
                        best_adj = [row[:] for row in adj]
                    # If improved, keep the edge and restart the loop (or continue)
                    # Restarting is safer to re-evaluate other options
                    break
                else:
                    # Revert
                    adj[u][v] = 0
                    adj[v][u] = 0
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck, try removing an edge?
            # Or just restart with a new random graph?
            # Removing edges might allow larger triangle-free sets, so it's risky.
            # But maybe it allows adding *different* edges later?
            # For now, if stuck, do a random restart or just stop.
            # Let's try a random restart after some failures.
            if no_improve_count > max_no_improve:
                adj = random_k4_free_graph()
                current_size = max_triangle_free_size(adj)
                if current_size < best_size:
                    best_size = current_size
                    best_adj = [row[:] for row in adj]
                no_improve_count = 0

    # Convert best_adj to list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(f"Found {len(edges)} edges")
