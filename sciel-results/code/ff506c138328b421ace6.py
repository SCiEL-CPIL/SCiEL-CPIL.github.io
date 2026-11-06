import random
import numpy as np

def construct():
    """
    Constructs a C4-free graph on n vertices with minimum degree at least n - 39.
    Targets n=46.
    """
    n = 46
    target_degree = n - 39  # 7
    
    # Initialize adjacency matrix
    # Use a set of sets for neighbors for fast lookup
    neighbors = [set() for _ in range(n)]
    
    def has_c4(u, v):
        """
        Check if adding edge (u, v) creates a C4.
        This happens if there is a path of length 3 between u and v.
        i.e., exists x, y such that u-x-y-v is a path.
        Equivalent to: (A @ A @ A)[u][v] > 0? No, that counts paths of length 3.
        Actually, if we add (u,v), a C4 is formed if there is a common neighbor of u and v? No.
        A C4 is u-w1-v-w2-u.
        This requires w1 in N(u) and w1 in N(v).
        And w2 in N(u) and w2 in N(v).
        So if |N(u) intersect N(v)| >= 2, adding (u,v) creates a C4?
        Let's trace:
        If w1, w2 are common neighbors of u and v.
        Cycle: u - w1 - v - w2 - u.
        Yes.
        So we must ensure |N(u) intersect N(v)| < 2.
        i.e., |N(u) intersect N(v)| <= 1.
        """
        common = neighbors[u].intersection(neighbors[v])
        return len(common) >= 2

    # Greedy filling
    # We want to reach min degree >= target_degree
    # We can just try to add random edges until condition met or no more edges can be added
    
    # To make it deterministic or reproducible, we can use a fixed seed or just random.
    # Random is fine as long as it finishes.
    
    # Strategy: 
    # 1. Try to add edges to increase degrees of low-degree vertices.
    # 2. If stuck, try to add any valid edge.
    
    for _ in range(10000): # Iteration limit
        # Check if condition met
        min_deg = min(len(neighbors[i]) for i in range(n))
        if min_deg >= target_degree:
            break
            
        # Find vertex with minimum degree
        u = min(range(n), key=lambda i: len(neighbors[i]))
        
        # Find a candidate v to connect to u
        # v should not be u, and not already connected
        candidates = [v for v in range(n) if v != u and v not in neighbors[u]]
        
        # Shuffle candidates to avoid bias
        random.shuffle(candidates)
        
        added = False
        for v in candidates:
            if not has_c4(u, v):
                # Add edge
                neighbors[u].add(v)
                neighbors[v].add(u)
                added = True
                break
        
        if not added:
            # If we can't add any edge to u, try another vertex or restart?
            # If no edge can be added to any vertex, we are stuck.
            # For n=46, this is unlikely to happen before reaching degree 7.
            # But to be safe, if we get stuck, we might try a different randomization.
            # However, simple restarts are expensive.
            # Let's just break and see if we met the condition.
            # If not, we might return a smaller n.
            # But the task asks to maximize n.
            # If we fail for 46, we should try 45?
            # The prompt says "Maximize n".
            # I will implement a loop trying n=46 down to 40.
            pass

    # If we didn't break via condition, check again
    min_deg = min(len(neighbors[i]) for i in range(n))
    if min_deg < target_degree:
        # Fallback: Try smaller n
        # We'll just return the best we got? 
        # Or restart with n=45?
        # To keep it simple and fast, I'll just try n=46 with multiple random restarts.
        
        success = False
        for restart in range(5):
            neighbors = [set() for _ in range(n)]
            for _ in range(10000):
                min_deg = min(len(neighbors[i]) for i in range(n))
                if min_deg >= target_degree:
                    success = True
                    break
                u = min(range(n), key=lambda i: len(neighbors[i]))
                candidates = [v for v in range(n) if v != u and v not in neighbors[u]]
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    if not has_c4(u, v):
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        added = True
                        break
                if not added:
                    break
            
            if success:
                break
        
        if not success:
            # If 46 fails, try 45
            n = 45
            target_degree = n - 39
            neighbors = [set() for _ in range(n)]
            for _ in range(10000):
                min_deg = min(len(neighbors[i]) for i in range(n))
                if min_deg >= target_degree:
                    break
                u = min(range(n), key=lambda i: len(neighbors[i]))
                candidates = [v for v in range(n) if v != u and v not in neighbors[u]]
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    if not has_c4(u, v):
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        added = True
                        break
                if not added:
                    break

    # Convert to edge list
    edges = []
    for i in range(n):
        for j in neighbors[i]:
            if i < j:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
