import random
import sys

def is_triangle_free(edges_set, n=16):
    # For efficiency, we can check triangles during the search moves
    # But for the initial check or validation, we can do this
    # However, in the search, we will maintain the invariant.
    pass

def is_3_colorable(edges, n=16):
    # Backtracking 3-coloring
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    colors = [-1] * n
    
    def backtrack(v):
        if v == n:
            return True
        # Try colors 0, 1, 2
        for c in range(3):
            valid = True
            for u in range(n):
                if adj[v][u] and colors[u] == c:
                    valid = False
                    break
            if valid:
                colors[v] = c
                if backtrack(v + 1):
                    return True
                colors[v] = -1
        return False

    return backtrack(0)

def construct():
    # Start with the known valid graph from the current program
    # This serves as a good seed for local search
    edges_list = []
    for i in range(5):
        edges_list.append([i, (i + 1) % 5])
    for i in range(5):
        u = i
        v = (i + 1) % 5
        edges_list.append([5 + u, v])
        edges_list.append([5 + v, u])
    for i in range(5):
        edges_list.append([10, 5 + i])
    independent_set = [0, 2, 5, 7]
    for new_v in range(11, 16):
        for u in independent_set:
            edges_list.append([new_v, u])
            
    # Convert to set for fast lookup
    edge_set = set()
    for u, v in edges_list:
        if u > v: u, v = v, u
        edge_set.add((u, v))
        
    current_edges = list(edge_set)
    
    # Verify initial state
    if is_3_colorable(current_edges):
        # Fallback if seed is invalid (shouldn't happen)
        return edges_list

    best_edges = list(edge_set)
    best_count = len(edge_set)
    
    # Local Search / Simulated Annealing
    # We want to maximize edges.
    # Temperature schedule
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    import time
    start_time = time.time()
    time_limit = 110 # Leave 10s for overhead
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Generate a neighbor
        move_type = random.random()
        
        new_edge_set = set(edge_set)
        
        if move_type < 0.7:
            # Try to add an edge
            # Pick a random pair not in edge_set
            while True:
                u = random.randint(0, 15)
                v = random.randint(0, 15)
                if u == v: continue
                if u > v: u, v = v, u
                if (u, v) not in new_edge_set:
                    break
            
            # Check if adding (u,v) creates a triangle
            # Triangle exists if there is w such that (u,w) and (v,w) are in new_edge_set
            creates_triangle = False
            for w in range(16):
                if w != u and w != v:
                    uw = (min(u,w), max(u,w))
                    vw = (min(v,w), max(v,w))
                    if uw in new_edge_set and vw in new_edge_set:
                        creates_triangle = True
                        break
                        
            if creates_triangle:
                continue # Reject move, try again
            
            new_edge_set.add((u, v))
            
        elif move_type < 0.9:
            # Try to remove an edge
            if not edge_set: continue
            e = random.choice(list(edge_set))
            new_edge_set.remove(e)
            
        else:
            # Swap: remove one, add one
            if not edge_set: continue
            e_remove = random.choice(list(edge_set))
            new_edge_set.remove(e_remove)
            
            # Try to add a random edge
            while True:
                u = random.randint(0, 15)
                v = random.randint(0, 15)
                if u == v: continue
                if u > v: u, v = v, u
                if (u, v) not in new_edge_set:
                    break
            
            # Check triangle for new edge (u,v)
            creates_triangle = False
            for w in range(16):
                if w != u and w != v:
                    uw = (min(u,w), max(u,w))
                    vw = (min(v,w), max(v,w))
                    if uw in new_edge_set and vw in new_edge_set:
                        creates_triangle = True
                        break
                        
            if creates_triangle:
                continue
            
            new_edge_set.add((u, v))

        # Evaluate new state
        # 1. Must be triangle-free (guaranteed by construction of moves)
        # 2. Must be non-3-colorable
        if is_3_colorable(list(new_edge_set)):
            # Penalty: this state is invalid for our goal
            # We can still accept it with low probability to escape, but it reduces score
            # For simplicity, reject if it becomes 3-colorable, unless we are stuck
            # Let's reject to keep search in valid space
            continue
            
        new_count = len(new_edge_set)
        delta = new_count - len(edge_set)
        
        # Accept if better, or with probability exp(delta/T)
        if delta >= 0:
            edge_set = new_edge_set
            if new_count > best_count:
                best_count = new_count
                best_edges = list(new_edge_set)
        else:
            if random.random() < pow(2.71828, delta / T):
                edge_set = new_edge_set
                
        T *= cooling
        
    # Return best found
    return [[u, v] for u, v in best_edges]
