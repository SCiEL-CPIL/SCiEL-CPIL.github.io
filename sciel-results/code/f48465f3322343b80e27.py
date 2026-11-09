import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n = 40
    target_edges = 401
    
    # Initialize with a nearly complete bipartite graph
    # Split into two parts of 20
    part_a = list(range(20))
    part_b = list(range(20, 40))
    
    # Start with complete bipartite K_{20,20} = 400 edges
    edges_set = set()
    for a in part_a:
        for b in part_b:
            edges_set.add((a, b))
    
    # Add one more edge within part_a
    edges_set.add((0, 1))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    
    def compute_book_sizes():
        """Compute book size for each edge."""
        book_sizes = {}
        for u, v in edges_set:
            common = adj[u] & adj[v]
            book_sizes[(u, v)] = len(common)
            book_sizes[(v, u)] = len(common)
        return book_sizes
    
    def max_book():
        if not edges_set:
            return 0
        return max(len(adj[u] & adj[v]) for u, v in edges_set)
    
    def try_swap(u, v, x, y):
        """Try to remove edge (u,v) and add edge (x,y).
        Returns True if swap was applied."""
        if (u, v) not in edges_set and (v, u) not in edges_set:
            return False
        if (x, y) in edges_set or (y, x) in edges_set:
            return False
        if u == x and v == y:
            return False
        if u == y and v == x:
            return False
        
        # Remove (u,v)
        adj[u].remove(v)
        adj[v].remove(u)
        if (u, v) in edges_set:
            edges_set.remove((u, v))
        else:
            edges_set.remove((v, u))
        
        # Add (x,y)
        adj[x].add(y)
        adj[y].add(x)
        edges_set.add((x, y))
        
        return True
    
    def undo_swap(u, v, x, y):
        """Undo the swap: remove (x,y) and add back (u,v)."""
        adj[x].remove(y)
        adj[y].remove(x)
        if (x, y) in edges_set:
            edges_set.remove((x, y))
        else:
            edges_set.remove((y, x))
        
        adj[u].add(v)
        adj[v].add(u)
        edges_set.add((u, v))
    
    current_max = max_book()
    best_max = current_max
    best_edges = set(edges_set)
    
    # Simulated annealing
    temperature = 2.0
    cooling = 0.995
    min_temp = 0.01
    
    iteration = 0
    while temperature > min_temp and time.time() < deadline:
        # Pick a random existing edge to remove
        edge_list = list(edges_set)
        u, v = random.choice(edge_list)
        
        # Pick a random non-edge to add
        # Sample random pairs
        attempts = 0
        x, y = None, None
        while attempts < 50:
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y:
                continue
            if (x, y) in edges_set or (y, x) in edges_set:
                continue
            if x == u and y == v:
                continue
            if x == v and y == u:
                continue
            break
            attempts += 1
        else:
            continue
        
        # Perform swap
        try_swap(u, v, x, y)
        new_max = max_book()
        
        # Accept if better or with probability
        if new_max < current_max:
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_edges = set(edges_set)
        elif new_max == current_max:
            # Keep it (neutral move)
            pass
        else:
            # Worse - accept with probability exp(-(new_max - current_max)/temperature)
            import math
            prob = math.exp(-(new_max - current_max) / temperature)
            if random.random() > prob:
                undo_swap(u, v, x, y)
        
        temperature *= cooling
        iteration += 1
    
    # Use best found
    edges_set = best_edges
    result = []
    for u, v in edges_set:
        result.append([u, v])
    
    return result
