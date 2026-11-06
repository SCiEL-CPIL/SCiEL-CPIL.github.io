import time
import random
import sys

def construct():
    # Deadline: 110 seconds (safety margin for 120s limit)
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 40
    # Adjacency matrix
    adj = [[False] * n for _ in range(n)]
    
    def is_c6_free_add(u, v):
        """
        Check if adding edge (u, v) to the current graph creates a C6.
        A C6 is formed if there is a path of length 5 between u and v in the current graph.
        Path: u - a - b - c - d - v
        This is equivalent to:
        There exists a node b such that:
        b is at distance 2 from u (u-a-b)
        AND b is at distance 3 from v (v-d-c-b)
        So we check if N^2(u) intersects N^3(v).
        """
        # Calculate N^2(u): nodes at distance exactly 2 from u
        n2_u = set()
        for a in range(n):
            if adj[u][a]:
                for b in range(n):
                    if adj[a][b] and b != u:
                        n2_u.add(b)
        
        # Calculate N^3(v): nodes at distance exactly 3 from v
        n3_v = set()
        # First get N(v)
        n1_v = [x for x in range(n) if adj[v][x]]
        # Then N^2(v)
        n2_v = set()
        for d in n1_v:
            for c in range(n):
                if adj[d][c] and c != v:
                    n2_v.add(c)
        # Then N^3(v)
        for c in n2_v:
            for b in range(n):
                if adj[c][b] and b != v:
                    n3_v.add(b)
                    
        # Check intersection
        if n2_u & n3_v:
            return False
        return True

    def get_edges():
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append((i, j))
        return edges

    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True

    def remove_edge(u, v):
        adj[u][v] = False
        adj[v][u] = False

    # 1. Greedy Construction with Randomization
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # We will run multiple random greedy restarts to find a good starting point
    best_edge_count = 0
    best_adj = None
    
    # Time budget for greedy phase: ~20s
    greedy_deadline = start_time + 20.0
    
    while time.time() < greedy_deadline:
        random.shuffle(all_edges)
        # Reset adj
        for i in range(n):
            for j in range(n):
                adj[i][j] = False
        
        current_edges = 0
        for u, v in all_edges:
            if is_c6_free_add(u, v):
                add_edge(u, v)
                current_edges += 1
        
        if current_edges > best_edge_count:
            best_edge_count = current_edges
            best_adj = [row[:] for row in adj]
            
    # Restore best adj
    if best_adj:
        for i in range(n):
            for j in range(n):
                adj[i][j] = best_adj[i][j]
    else:
        # Fallback to seed if greedy failed (unlikely)
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k + a
                    v = 5*k + b
                    add_edge(u, v)
        best_edge_count = 80

    # 2. Local Search (1-Swap: Remove one edge, then greedily add as many as possible)
    # Time budget for local search: remaining time
    # We will perform swaps until no improvement or time limit.
    
    # To speed up, we can maintain a list of missing edges
    # But recalculating is simpler and V=40 is small.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
        
        # Get list of current edges
        current_edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
        random.shuffle(current_edge_list)
        
        for u, v in current_edge_list:
            if time.time() > deadline:
                break
                
            # Try removing (u, v)
            remove_edge(u, v)
            
            # Now try to add any missing edge that doesn't create C6
            added_any = False
            # We can iterate through all possible edges
            # Optimization: Only try edges that are "close" or just all of them?
            # With 780 edges, checking all is fast enough.
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i][j]:
                        if is_c6_free_add(i, j):
                            add_edge(i, j)
                            added_any = True
                            
            new_edges = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
            
            if new_edges > current_edges:
                improved = True
                current_edges = new_edges
                # Accept the change
            else:
                # Revert
                # We need to remove the edges we just added
                # This is tricky. We should have tracked them.
                # Let's restart the local search loop if we revert?
                # Or better: track added edges.
                
                # Actually, the logic above is flawed because if we revert, we don't know which edges to remove.
                # Let's re-structure the swap logic.
                
                # We removed (u,v). We added a set of edges E_added.
                # If new_edges <= current_edges, we must revert:
                # 1. Remove all edges in E_added.
                # 2. Re-add (u,v).
                
                # To do this, we need to track E_added.
                
                pass # The loop above is implemented in a way that modifies adj directly.
                # We need to fix the implementation to track added edges.

    # Re-implementing the local search loop properly
    
    # Reset to best state
    if best_adj:
        for i in range(n):
            for j in range(n):
                adj[i][j] = best_adj[i][j]
                
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_edges = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
        
        # Get list of current edges
        current_edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i][j]]
        random.shuffle(current_edge_list)
        
        for u, v in current_edge_list:
            if time.time() > deadline:
                break
                
            # 1. Remove (u, v)
            remove_edge(u, v)
            
            # 2. Try to add missing edges
            added_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if not adj[i][j]:
                        if is_c6_free_add(i, j):
                            add_edge(i, j)
                            added_edges.append((i, j))
                            
            new_edges = current_edges - 1 + len(added_edges)
            
            if new_edges > current_edges:
                improved = True
                current_edges = new_edges
                # Accept: (u,v) is already removed, added_edges are already added.
            else:
                # Revert
                for i, j in added_edges:
                    remove_edge(i, j)
                add_edge(u, v)
                
    # Return the final list of edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
                
    return edges

# Run construct to verify it works and measure time
if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Edges: {len(res)}")
    print(f"Time: {end - start:.2f}s")
