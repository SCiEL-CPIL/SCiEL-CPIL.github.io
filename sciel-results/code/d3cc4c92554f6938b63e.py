import time
import random
import sys

def construct():
    n = 16
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Initial Graph: Groetzsch Graph (11 vertices) + 5 isolated
    # Vertices 0..4: Cycle
    # Vertices 5..9: Copies
    # Vertex 10: Apex
    # Vertices 11..15: Isolated
    
    def create_groetzsch():
        edges = set()
        # Cycle 0-1-2-3-4-0
        for i in range(5):
            edges.add((i, (i + 1) % 5))
        # Copies 5..9
        for i in range(5):
            u = i
            v = (i + 1) % 5
            edges.add((u, 5 + v))
            edges.add((v, 5 + u))
        # Apex 10
        for i in range(5):
            edges.add((5 + i, 10))
        return edges

    # Helper: Check if adding edge (u, v) creates a triangle
    def can_add_edge(adj, u, v):
        if u == v:
            return False
        # Check common neighbors
        # adj is a list of sets or list of lists
        # Using sets for O(1) lookup
        for w in adj[u]:
            if w in adj[v]:
                return False
        return True

    # Helper: Greedily add all possible non-triangle edges
    def greedy_fill(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        if can_add_edge(adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            changed = True
        return adj

    # Helper: Count edges
    def count_edges(adj):
        return sum(len(neigh) for neigh in adj) // 2

    # Initialize
    init_edges = create_groetzsch()
    adj = [set() for _ in range(n)]
    for u, v in init_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Initial Fill
    adj = greedy_fill(adj)
    best_adj = [set(s) for s in adj]
    best_count = count_edges(adj)
    
    # Local Search
    # Strategy: Randomly remove an edge, then greedy fill.
    # If the result is better, keep it.
    # Also try removing multiple edges?
    
    # To speed up can_add_edge, we can precompute common neighbors?
    # Or just use the set intersection.
    # Intersection of adj[u] and adj[v] is empty?
    
    def can_add_edge_fast(adj, u, v):
        if u == v:
            return False
        # If they share any neighbor, triangle formed
        if adj[u].intersection(adj[v]):
            return False
        return True

    # Update greedy_fill to use fast check
    def greedy_fill_fast(adj):
        changed = True
        while changed:
            changed = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v not in adj[u]:
                        if can_add_edge_fast(adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            changed = True
        return adj

    # Re-run initial fill with fast check
    adj = [set() for _ in range(n)]
    for u, v in init_edges:
        adj[u].add(v)
        adj[v].add(u)
    adj = greedy_fill_fast(adj)
    best_adj = [set(s) for s in adj]
    best_count = count_edges(adj)
    
    # Search Loop
    while time.time() < deadline:
        # Pick a random edge to remove
        # We need a list of edges
        current_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    current_edges.append((u, v))
        
        if not current_edges:
            break
            
        # Randomly select an edge to remove
        # Sometimes remove 2 or 3 to escape local optima?
        # Let's start with 1.
        e_u, e_v = random.choice(current_edges)
        
        # Create a copy
        trial_adj = [set(s) for s in adj]
        trial_adj[e_u].remove(e_v)
        trial_adj[e_v].remove(e_u)
        
        # Greedy Fill
        trial_adj = greedy_fill_fast(trial_adj)
        trial_count = count_edges(trial_adj)
        
        if trial_count > best_count:
            best_count = trial_count
            best_adj = [set(s) for s in trial_adj]
            # Update current adj to this new best to continue from there
            adj = [set(s) for s in trial_adj]
            
        # Also try a random swap?
        # Remove edge (u,v), try to add (w,x)?
        # This is more complex. The remove-and-fill is a good heuristic.
        
        # Occasionally, try removing 2 random edges
        if random.random() < 0.1:
             if len(current_edges) >= 2:
                 e1 = random.choice(current_edges)
                 e2 = random.choice(current_edges)
                 if e1 != e2:
                     trial_adj2 = [set(s) for s in adj]
                     trial_adj2[e1[0]].remove(e1[1])
                     trial_adj2[e1[1]].remove(e1[0])
                     trial_adj2[e2[0]].remove(e2[1])
                     trial_adj2[e2[1]].remove(e2[0])
                     trial_adj2 = greedy_fill_fast(trial_adj2)
                     trial_count2 = count_edges(trial_adj2)
                     if trial_count2 > best_count:
                         best_count = trial_count2
                         best_adj = [set(s) for s in trial_adj2]
                         adj = [set(s) for s in trial_adj2]

    # Convert best_adj to list of lists
    result_edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result_edges.append([u, v])
                
    return result_edges
