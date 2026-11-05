import random
import time

def construct():
    n = 20
    # Use a fixed seed for reproducibility? No, random is fine if valid.
    # But we want max edges. Random order might yield different results.
    # We can try a few random orders and keep the best.
    
    best_edges = []
    deadline = time.time() + 100
    
    for _ in range(10): # Try 10 random orderings
        if time.time() > deadline: break
        
        edges_set = set()
        adj = [set() for _ in range(n)]
        
        # Random permutation of all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a 4-cycle
            # A 4-cycle is created if there exists a path u-a-b-v of length 3
            # i.e., exists a in N(u), b in N(v) such that (a,b) is an edge
            creates_cycle = False
            for a in adj[u]:
                for b in adj[v]:
                    if (a, b) in edges_set or (b, a) in edges_set:
                        creates_cycle = True
                        break
                if creates_cycle:
                    break
            
            if not creates_cycle:
                edges_set.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        
        if len(edges_set) > len(best_edges):
            best_edges = list(edges_set)
            
    # Colour all edges with 0
    # The problem allows colours 0..19. Using 0 for all is valid if no 4-cycles.
    result = []
    for u, v in best_edges:
        result.append([u, v, 0])
        
    return result
