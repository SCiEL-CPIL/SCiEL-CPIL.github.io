import time
import random
import math
import sys

def construct():
    """
    Constructs a graph with 40 vertices and 401 edges minimizing the largest book size.
    Uses Simulated Annealing to optimize the graph structure.
    """
    n = 40
    m = 401
    deadline = time.time() + 100  # Safety margin
    
    # State representation
    # adj: list of sets of neighbors
    adj = [set() for _ in range(n)]
    # edges: set of tuples (u, v) with u < v
    edges = set()
    # book: dict mapping edge tuple to book size
    book = {}
    
    def add_edge(u, v):
        if u == v: return False
        if u > v: u, v = v, u
        if (u, v) in edges: return False
        edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
        # Calculate book size for (u, v)
        common = adj[u] & adj[v]
        b = len(common)
        book[(u, v)] = b
        
        # Update book sizes for neighbors
        # For each neighbor w of u, if (w, v) is an edge, v is a common neighbor of (u, w)
        for w in adj[u]:
            if w == v: continue
            if (w, v) in edges:
                # Ensure normalized key
                key = (w, v) if w < v else (v, w)
                # Wait, (u, w) is the edge whose book size changes?
                # Book(u, w) counts common neighbors z.
                # If (w, v) is an edge, then v is a common neighbor of u and w?
                # Yes, if (u,v) and (w,v) exist.
                # So Book(u, w) increases by 1.
                key_uw = (u, w) if u < w else (w, u)
                book[key_uw] += 1
                
        for w in adj[v]:
            if w == u: continue
            if (w, u) in edges:
                key_vw = (v, w) if v < w else (w, v)
                book[key_vw] += 1
                
        return True

    def remove_edge(u, v):
        if u > v: u, v = v, u
        if (u, v) not in edges: return False
        
        # Remove from book
        b = book.pop((u, v))
        
        # Update book sizes for neighbors
        for w in adj[u]:
            if w == v: continue
            if (w, v) in edges:
                key_uw = (u, w) if u < w else (w, u)
                book[key_uw] -= 1
                
        for w in adj[v]:
            if w == u: continue
            if (w, u) in edges:
                key_vw = (v, w) if v < w else (w, v)
                book[key_vw] -= 1
        
        edges.remove((u, v))
        adj[u].remove(v)
        adj[v].remove(u)
        return True

    # Initialize random graph
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    for u, v in all_pairs[:m]:
        add_edge(u, v)
        
    # Initial energy
    current_energy = sum(b*b for b in book.values())
    current_max = max(book.values()) if book else 0
    
    best_energy = current_energy
    best_max = current_max
    best_edges = set(edges)
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.9995
    steps = 0
    
    # Precompute neighbors for faster random non-edge selection?
    # Just pick random u, v and check if edge exists.
    
    while time.time() < deadline:
        # Pick a random edge to remove
        if not edges:
            break
        u, v = random.choice(list(edges))
        
        # Pick a random non-edge to add
        # Retry a few times if it's an edge
        added = False
        for _ in range(10):
            x, y = random.sample(range(n), 2)
            if x > y: x, y = y, x
            if (x, y) not in edges:
                # Perform swap
                # 1. Remove (u, v)
                old_energy = current_energy
                remove_edge(u, v)
                
                # 2. Add (x, y)
                add_edge(x, y)
                
                new_energy = sum(b*b for b in book.values())
                delta = new_energy - old_energy
                
                # Check if better
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_max = max(book.values()) if book else 0
                    best_edges = set(edges)
                    
                # Acceptance probability
                if delta < 0:
                    current_energy = new_energy
                else:
                    if T > 0.01:
                        p = math.exp(-delta / T)
                        if random.random() < p:
                            current_energy = new_energy
                        else:
                            # Revert
                            remove_edge(x, y)
                            add_edge(u, v)
                            # Energy is back to old_energy
                            current_energy = old_energy
                    # else revert
                    else:
                        remove_edge(x, y)
                        add_edge(u, v)
                        current_energy = old_energy
                added = True
                break
        
        if not added:
            continue
            
        T *= cooling
        steps += 1
        
        # Optional: If stuck, try a small perturbation or restart?
        # For now, just continue SA.
        
    # Restore best graph
    # Clear current state
    edges.clear()
    for i in range(n):
        adj[i].clear()
    book.clear()
    
    for u, v in best_edges:
        add_edge(u, v)
        
    # Return in required format: list of lists [u, v]
    result = [[u, v] for u, v in best_edges]
    return result
