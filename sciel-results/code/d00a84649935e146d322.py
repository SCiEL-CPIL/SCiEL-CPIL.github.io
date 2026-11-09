import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 18
    # Adjacency matrix
    adj = [[0]*n for _ in range(n)]
    
    def is_k4_free(u, v):
        # Check if adding edge (u,v) creates a K4
        # A K4 containing (u,v) must have two other vertices x,y
        # such that x,y are connected to u and v, and x-y is an edge.
        # So we need to check if there is an edge (x,y) in the common neighborhood of u and v.
        common = []
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                common.append(w)
        # Check if any pair in common is connected
        for i in range(len(common)):
            for j in range(i+1, len(common)):
                if adj[common[i]][common[j]]:
                    return False
        return True

    def add_edge(u, v):
        adj[u][v] = 1
        adj[v][u] = 1

    def remove_edge(u, v):
        adj[u][v] = 0
        adj[v][u] = 0

    def get_max_triangle_free_size():
        # Check subsets of size s from high to low
        # Max possible is 18
        # We can optimize by checking only subsets of size s
        # But 18 is small.
        # Let's check if there exists a triangle-free subset of size s.
        # We can use a simple backtracking or just iterate combinations.
        # For s=10, C(18,10) = 43758.
        # For s=18, C(18,18)=1.
        # We want the LARGEST size.
        # So we check s=18, 17, ...
        # But we know it's at most 12 for seed.
        # Let's check from 12 down to 9?
        # Actually, we can just compute the max.
        
        # Optimization: 
        # A set is triangle-free if it contains no triangle.
        # We can precompute for each vertex pair if it's an edge.
        # For a subset S, it is triangle-free if for all u,v,w in S, not (adj[u][v] and adj[v][w] and adj[w][u]).
        
        # Let's try to find the max size.
        max_size = 0
        # We can use a heuristic: 
        # If we find a triangle-free set of size k, max_size >= k.
        # We want to minimize this.
        
        # Let's just check all subsets of size s from 18 down to 1?
        # That is too slow if we do it for every step.
        # But n=18 is small.
        
        # Better: 
        # We can maintain the max size.
        # When adding an edge, the max size can only decrease or stay same.
        # When removing an edge, the max size can only increase or stay same.
        
        # For the initial greedy phase, we start empty (max size 18).
        # We add edges. Max size decreases.
        
        # Let's implement a function that returns the current max triangle-free size.
        # To speed up, we can cache or use bitsets.
        
        # For simplicity, we check subsets of size s.
        # We only need to know if there is ANY triangle-free subset of size s.
        # If yes, max_size >= s.
        
        # Let's check s from 18 down to 1.
        # But we can start from the previous max size.
        
        # This function is called frequently.
        # Let's use a random sampling approach for speed? No, must be exact.
        
        # C(18, 9) = 48620.
        # C(18, 10) = 43758.
        # Total subsets is 2^18 = 262144.
        # Checking all subsets is feasible if optimized.
        
        # Let's represent the graph as a list of triangles.
        # If a subset contains a triangle, it's not triangle-free.
        
        # We'll just iterate all subsets of size s.
        
        # To make it faster, we can use the fact that if a set is triangle-free,
        # any subset is triangle-free.
        
        # Let's just return the max size found.
        
        # We'll check s=18, 17, ...
        # But we can stop early if we find one.
        
        # Actually, we want to MINIMIZE the max size.
        # So we want to know the current max size.
        
        # Let's assume the max size is at most 12 initially.
        # We check s=12, 11, 10, 9.
        
        # If no triangle-free set of size 12, check 11, etc.
        
        # We'll implement a check for "exists triangle-free set of size s".
        
        # For speed, we'll use combinations from itertools.
        
        # This might be slow in Python if called thousands of times.
        # But 150s is a lot.
        
        # Let's try to optimize the check.
        
        # We'll just do it.
        
        pass # Placeholder

    # Greedy construction
    # Start with random order of edges
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(edges)
    
    for u, v in edges:
        if time.time() > deadline: break
        if not adj[u][v]:
            if is_k4_free(u, v):
                # Check if adding improves (reduces) max triangle-free size
                # We need to evaluate.
                # For now, just add if it doesn't violate K4.
                # We can be smarter: only add if it breaks triangles.
                add_edge(u, v)
                
    # Local search
    # Try to add more edges
    # Try to remove edges?
    
    # Return edges
    res = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                res.append([i, j])
    return res
