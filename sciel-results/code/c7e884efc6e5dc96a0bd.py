import random
import time

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # Try to construct for n=47 first, then n=46
    for n in [47, 46]:
        target_min_deg = n - 39
        # If target_min_deg <= 0, any C4-free graph works, but we want high n.
        # For n=47, target is 8. For n=46, target is 7.
        
        max_attempts = 100000 # High number, but bounded by time
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
                
            # Randomized Greedy Algorithm
            adj = [0] * n
            edges = []
            
            # Generate random permutation of all possible edges
            # For n=47, there are 47*46/2 = 1081 edges
            # Creating a list of 1081 tuples and shuffling is fast
            possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(possible_edges)
            
            for u, v in possible_edges:
                # Check if adding (u, v) creates a C4
                # A C4 is formed if u and v have a common neighbor
                common = adj[u] & adj[v]
                if common:
                    continue
                
                # Safe to add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append((u, v))
            
            # Check minimum degree
            min_deg = min(bin(adj[i]).count('1') for i in range(n))
            
            if min_deg >= target_min_deg:
                # Success! Convert to required format
                # The task requires edges as [[u, v], ...]
                # Vertices are 0..n-1
                return {
                    'n': n,
                    'edges': [list(e) for e in edges]
                }
        
        # If we exhausted attempts for n=47, try n=46 in the next iteration
        # If we exhausted attempts for n=46, we might fail, but it's very unlikely.
        
    # Fallback: Return a valid small graph if large ones fail (should not happen)
    # For n=39, min_deg >= 0 is always true for any C4-free graph.
    # A simple cycle C_39 is C4-free and has min degree 2.
    n = 39
    edges = []
    for i in range(n):
        edges.append([i, (i+1)%n])
    return {
        'n': n,
        'edges': edges
    }

# Keep the original function name and signature
# The task says: "Keep the function names and signatures of the seed program."
# The seed program was:
# def construct():
#     return {'n': 39, 'edges': []}

# So we just define construct.

if __name__ == "__main__":
    res = construct()
    print(f"Found graph with n={res['n']}, edges={len(res['edges'])}")
