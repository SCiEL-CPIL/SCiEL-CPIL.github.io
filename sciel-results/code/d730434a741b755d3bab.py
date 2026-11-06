import random
import time
import copy
from itertools import combinations

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit, 5s safety margin
    
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    
    # Build conflict graph: two edge-vertices conflict if they can't coexist in K2,3-free graph
    # Two edges (u,v) and (a,b) conflict if:
    # 1. They share a vertex and create a K2,3 with some other edges
    # More precisely, we need to check if both edges being present forces a K2,3
    # 
    # Actually, the constraint is: no two vertices have 3 common neighbors.
    # So edges (u,v) and (a,b) are compatible as long as their addition doesn't create
    # a pair of vertices with 3 common neighbors.
    #
    # For branch and bound, we need a conflict graph where edge i and edge j are adjacent
    # if they CANNOT both be in the graph.
    #
    # Two edges e1=(u,v) and e2=(a,b) conflict if:
    # - They share a vertex, say u=a, and there exist vertices w1,w2,w3 such that
    #   w1,w2,w3 are all neighbors of both v and b (but this depends on other edges too)
    #
    # This is complex. Let me use a different approach:
    # Use the original local search but with better heuristics and a conflict-based pruning.
    
    def is_valid_addition(adj, c, u, v):
        if c[u][v] > 2:
            return False
        for w in adj[u]:
            if c[w][v] > 1:
                return False
        for w in adj[v]:
            if c[u][w] > 1:
                return False
        return True

    def add_edge(state, u, v):
        adj, c, edges = state
        edges.append([u, v])
        adj[u].add(v)
        adj[v].add(u)
        c[u][v] += 1
        c[v][u] += 1
        for w in adj[u]:
            c[w][v] += 1
            c[v][w] += 1
        for w in adj[v]:
            c[u][w] += 1
            c[w][u] += 1

    def remove_edge(state, u, v):
        adj, c, edges = state
        for i, e in enumerate(edges):
            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                edges.pop(i)
                break
        adj[u].remove(v)
        adj[v].remove(u)
        c[u][v] -= 1
        c[v][u] -= 1
        for w in adj[u]:
            c[w][v] -= 1
            c[v][w] -= 1
        for w in adj[v]:
            c[u][w] -= 1
            c[w][u] -= 1

    def build_state(edges):
        adj = [set() for _ in range(n)]
        c = [[0] * n for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            c[u][v] += 1
            c[v][u] += 1
            for w in adj[u]:
                c[w][v] += 1
                c[v][w] += 1
            for w in adj[v]:
                c[u][w] += 1
                c[w][u] += 1
        return adj, c, list(edges)

    def greedy_construct():
        current_edges = []
        current_adj = [set() for _ in range(n)]
        current_c = [[0] * n for _ in range(n)]
        state = (current_adj, current_c, current_edges)
        
        missing_edges = list(all_edges)
        random.shuffle(missing_edges)
        
        for u, v in missing_edges:
            if time.time() > deadline:
                break
            if is_valid_addition(current_adj, current_c, u, v):
                add_edge(state, u, v)
        
        return current_edges

    def local_search(edges):
        best_local_edges = list(edges)
        best_local_count = len(edges)
        
        current_adj, current_c, current_edges = build_state(edges)
        state = (current_adj, current_c, current_edges)
        
        # Phase 1: Try to add edges greedily
        for _ in range(50):
            if time.time() > deadline:
                break
            missing = [e for e in all_edges if e not in set(map(tuple, current_edges))]
            random.shuffle(missing)
            added_any = False
            for u, v in missing:
                if is_valid_addition(current_adj, current_c, u, v):
                    add_edge(state, u, v)
                    added_any = True
            if not added_any:
                break
        
        if len(current_edges) > best_local_count:
            best_local_count = len(current_edges)
            best_local_edges = list(current_edges)
        
        # Phase 2: Edge swaps (remove one, try to add multiple)
        max_swaps = 200
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            if not current_edges:
                break
            
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            
            saved_edges = list(current_edges)
            saved_adj = [set(s) for s in current_adj]
            saved_c = [row[:] for row in current_c]
            
            remove_edge(state, u, v)
            
            missing = [e for e in all_edges if e not in set(map(tuple, current_edges))]
            random.shuffle(missing)
            for a, b in missing:
                if is_valid_addition(current_adj, current_c, a, b):
                    add_edge(state, a, b)
            
            if len(current_edges) > best_local_count:
                best_local_count = len(current_edges)
                best_local_edges = list(current_edges)
            else:
                current_edges = saved_edges
                current_adj = saved_adj
                current_c = saved_c
                state = (current_adj, current_c, current_edges)
        
        return best_local_edges

    def branch_and_bound(edges, depth=0):
        """Branch and bound to find a larger independent set (K2,3-free subgraph)."""
        if time.time() > deadline:
            return list(edges)
        
        current_adj, current_c, current_edges = build_state(edges)
        state = (current_adj, current_c, current_edges)
        
        best = list(current_edges)
        best_count = len(current_edges)
        
        # Find all edges that can be added
        addable = []
        for u, v in all_edges:
            if (u, v) not in set(map(tuple, current_edges)) and is_valid_addition(current_adj, current_c, u, v):
                addable.append((u, v))
        
        if not addable:
            return best
        
        # Sort addable edges by some heuristic (e.g., degree in conflict graph)
        # For now, just try all in random order with pruning
        random.shuffle(addable)
        
        # Try adding each addable edge and recurse
        for u, v in addable:
            if time.time() > deadline:
                break
            
            # Save state
            saved_edges = list(current_edges)
            saved_adj = [set(s) for s in current_adj]
            saved_c = [row[:] for row in current_c]
            
            # Add edge
            add_edge(state, u, v)
            
            # Recurse
            result = branch_and_bound(current_edges, depth + 1)
            
            if len(result) > best_count:
                best_count = len(result)
                best = result
            
            # Restore state
            current_edges = saved_edges
            current_adj = saved_adj
            current_c = saved_c
            state = (current_adj, current_c, current_edges)
        
        return best

    def improved_local_search(edges):
        """Enhanced local search with multiple swap strategies."""
        best_local_edges = list(edges)
        best_local_count = len(edges)
        
        current_adj, current_c, current_edges = build_state(edges)
        state = (current_adj, current_c, current_edges)
        
        # Phase 1: Greedy addition
        for _ in range(100):
            if time.time() > deadline:
                break
            missing = [e for e in all_edges if e not in set(map(tuple, current_edges))]
            if not missing:
                break
            random.shuffle(missing)
            added_any = False
            for u, v in missing:
                if is_valid_addition(current_adj, current_c, u, v):
                    add_edge(state, u, v)
                    added_any = True
            if not added_any:
                break
        
        if len(current_edges) > best_local_count:
            best_local_count = len(current_edges)
            best_local_edges = list(current_edges)
        
        # Phase 2: 1-for-1 swaps
        for _ in range(300):
            if time.time() > deadline:
                break
            if not current_edges:
                break
            
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            
            saved_edges = list(current_edges)
            saved_adj = [set(s) for s in current_adj]
            saved_c = [row[:] for row in current_c]
            
            remove_edge(state, u, v)
            
            missing = [e for e in all_edges if e not in set(map(tuple, current_edges))]
            random.shuffle(missing)
            for a, b in missing:
                if is_valid_addition(current_adj, current_c, a, b):
                    add_edge(state, a, b)
                    break
            
            if len(current_edges) > best_local_count:
                best_local_count = len(current_edges)
                best_local_edges = list(current_edges)
            else:
                current_edges = saved_edges
                current_adj = saved_adj
                current_c = saved_c
                state = (current_adj, current_c, current_edges)
        
        # Phase 3: 1-for-2 swaps (remove one, add two)
        for _ in range(100):
            if time.time() > deadline:
                break
            if not current_edges:
                break
            
            idx = random.randint(0, len(current_edges) - 1)
            u, v = current_edges[idx]
            
            saved_edges = list(current_edges)
            saved_adj = [set(s) for s in current_adj]
            saved_c = [row[:] for row in current_c]
            
            remove_edge(state, u, v)
            
            missing = [e for e in all_edges if e not in set(map(tuple, current_edges))]
            random.shuffle(missing)
            added = 0
            for a, b in missing:
                if is_valid_addition(current_adj, current_c, a, b):
                    add_edge(state, a, b)
                    added += 1
                    if added >= 2:
                        break
            
            if len(current_edges) > best_local_count:
                best_local_count = len(current_edges)
                best_local_edges = list(current_edges)
            else:
                current_edges = saved_edges
                current_adj = saved_adj
                current_c = saved_c
                state = (current_adj, current_c, current_edges)
        
        return best_local_edges

    # Main loop
    iterations = 0
    last_improved_ls = 0
    
    while time.time() < deadline:
        edges = greedy_construct()
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
        
        iterations += 1
        
        if iterations % 100 == 0 and time.time() - last_improved_ls > 2.0:
            last_improved_ls = time.time()
            improved = improved_local_search(best_edges)
            if len(improved) > best_count:
                best_count = len(improved)
                best_edges = improved

    return best_edges
