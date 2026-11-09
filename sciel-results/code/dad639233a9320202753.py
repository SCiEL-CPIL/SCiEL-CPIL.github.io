import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s for safety
    
    # Try to find a valid graph for n=19, 20, ... up to 35
    # Start with small n to ensure we return something valid if search fails
    
    best_result = None
    
    # Seed: Paley 17
    # We can just use the seed if nothing better is found
    
    # Strategy:
    # 1. Try Paley(19) with edge removal to kill K4s.
    # 2. Try Paley(23) with edge removal.
    # 3. Try random local search for n=20, 25, 30, 35.
    
    def is_valid(n, edges):
        # Check K4
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
            
        # Check K4
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]: continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]: continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False, "K4"
                            
        # Check Alpha 6
        # Use Bron-Kerbosch or simple enumeration for small n
        # For n=19, 23, 35, enumeration of 6-sets is C(35,6) ~ 1.6M, feasible.
        for comb in itertools.combinations(range(n), 6):
            is_independent = True
            for i in range(6):
                for j in range(i+1, 6):
                    if adj[comb[i]][comb[j]]:
                        is_independent = False
                        break
                if not is_independent:
                    break
            if is_independent:
                return False, "Alpha6"
                
        return True, "Valid"

    def get_paley(n):
        # n must be prime q = 1 mod 4
        # Check if n is prime
        def is_prime(k):
            if k < 2: return False
            for i in range(2, int(k**0.5)+1):
                if k % i == 0: return False
            return True
        
        if not is_prime(n):
            return None
            
        squares = {x*x % n for x in range(1, n)}
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (j-i) % n in squares:
                    edges.append([i, j])
        return edges

    # Try Paley(19)
    edges_19 = get_paley(19)
    if edges_19:
        # Remove edges to kill K4
        # Find all K4s
        n = 19
        adj = [[False]*n for _ in range(n)]
        for u, v in edges_19:
            adj[u][v] = True
            adj[v][u] = True
            
        k4s = []
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]: continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]: continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            k4s.append((a,b,c,d))
                            
        # Greedily remove edges
        # To minimize edge removal, we can just remove one edge per K4
        # But removing an edge might create new Alpha sets
        # Since Alpha was 4, removing edges might push it to 5 or 6
        # We will check validity after removal
        
        edges_mod = list(edges_19)
        # Sort K4s by some heuristic? Just iterate
        # We need to remove edges such that no K4 remains
        # This is a hitting set problem
        # For small n, we can just try removing edges
        
        # Simple approach: Iterate through K4s, if still present, remove first edge
        # This is O(K4 * E)
        
        # Convert to set for fast lookup
        edge_set = set(tuple(sorted(e)) for e in edges_mod)
        
        for k4 in k4s:
            # Check if this K4 is still present
            a,b,c,d = k4
            if all(tuple(sorted(p)) in edge_set for p in [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]):
                # Remove edge (a,b)
                edge_set.remove((a,b))
                
        edges_final = [list(e) for e in edge_set]
        
        valid, reason = is_valid(n, edges_final)
        if valid:
            return {'n': n, 'edges': edges_final}
            
    # Try Paley(23)
    edges_23 = get_paley(23)
    if edges_23:
        n = 23
        adj = [[False]*n for _ in range(n)]
        for u, v in edges_23:
            adj[u][v] = True
            adj[v][u] = True
            
        k4s = []
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]: continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]: continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            k4s.append((a,b,c,d))
                            
        edge_set = set(tuple(sorted(e)) for e in edges_23)
        
        for k4 in k4s:
            a,b,c,d = k4
            if all(tuple(sorted(p)) in edge_set for p in [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]):
                edge_set.remove((a,b))
                
        edges_final = [list(e) for e in edge_set]
        
        valid, reason = is_valid(n, edges_final)
        if valid:
            return {'n': n, 'edges': edges_final}
            
    # If all else fails, return seed
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
