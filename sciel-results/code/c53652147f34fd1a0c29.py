import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    # Target n
    # We will try to construct a graph for n=35.
    # If we fail, we might try 34, etc.
    # But 35 is the record. Let's focus on 35.
    
    n = 35
    # Initialize with random edges
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Local Search
    # We will iterate, finding violations and fixing them.
    # To avoid getting stuck in a loop, we can use a "tabu" list or just randomization.
    # Given the short time, simple random repair is best.
    
    for step in range(50000): # Max steps
        if time.time() > deadline - 5:
            break
            
        # Check for K4
        k4 = find_k4(adj, n)
        if k4:
            i, j, k, l = k4
            # Remove a random edge from the K4
            edges = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
            u, v = random.choice(edges)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
            
        # Check for Independent Set of 6 (K6 in complement)
        # Build complement
        full_mask = (1 << n) - 1
        cadj = [0] * n
        for i in range(n):
            cadj[i] = (full_mask ^ (1 << i)) & ~adj[i]
            
        k6 = find_k_clique(cadj, 6, n)
        if k6:
            # k6 is a list of 6 vertices
            # Add a random edge between two of them
            u, v = random.sample(k6, 2)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
            
        # If we get here, no K4 and no K6 in complement found by our finders.
        # This means the graph is valid?
        # We must ensure find_k4 and find_k_clique are EXACT.
        # If they are exact, then it is valid.
        
        # Verify
        if is_valid(adj, n):
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
        # If is_valid says False but finders didn't find anything?
        # This implies a bug in finders or is_valid.
        # For safety, do a random flip.
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v:
            if adj[u] & (1 << v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
    # Fallback
    # Return a small valid graph
    # Paley 17 is valid?
    # R(4,6) > 17 is true.
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def find_k4(adj, n):
    """Returns a tuple (i,j,k,l) if a K4 is found, else None."""
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                cn = adj[i] & adj[j]
                # cn contains common neighbors of i and j
                # We need to find an edge (k,l) in cn
                # Iterate k in cn
                temp = cn
                while temp:
                    lsb = temp & -temp
                    k = lsb.bit_length() - 1
                    temp -= lsb
                    # k is a common neighbor
                    # Check if k has any neighbor l in cn
                    # l must be in cn and adj[k]
                    l_mask = cn & adj[k]
                    if l_mask:
                        # Found l
                        l = (l_mask & -l_mask).bit_length() - 1
                        return (i, j, k, l)
    return None

def find_k_clique(adj, k, n):
    """
    Finds a clique of size k in graph adj.
    Returns a list of vertices if found, else None.
    Uses backtracking with bitsets.
    """
    # Initial candidates: all vertices
    candidates = (1 << n) - 1
    clique = []
    
    def search(candidates, clique, k_remaining):
        if k_remaining == 0:
            return clique[:]
        if candidates == 0:
            return None
        # Heuristic: pick vertex with highest degree in candidates?
        # For simplicity and speed in Python, picking the first one is O(1) but might be less effective for pruning.
        # Let's pick a random vertex from candidates to avoid bias?
        # Or just the lowest bit.
        
        # Pruning: if number of candidates < k_remaining, return None
        if candidates.bit_count() < k_remaining:
            return None
            
        # Pick a vertex v
        # Let's pick the one with highest degree in the induced subgraph of candidates
        # This is expensive to compute exactly.
        # Approximation: pick random.
        # Or pick lowest bit.
        v_bit = candidates & -candidates
        v = v_bit.bit_length() - 1
        
        # Remove v from candidates for the "exclude" branch
        # For "include" branch:
        # New candidates = candidates & adj[v] & ~v_bit
        # But we must ensure we only consider vertices > v? No, order doesn't matter for clique existence, 
        # but for avoiding duplicates, we usually restrict to vertices > v.
        # However, if we just want *a* clique, we can just search recursively.
        
        # To avoid exponential blowup, we should order vertices.
        # Let's use a simple recursive structure that passes a list of candidates.
        # Bitsets are faster.
        
        # Include v
        new_cands = candidates & adj[v] & ~v_bit
        # We should only consider vertices in new_cands that are "after" v in some order?
        # If we don't enforce order, we might find the same clique multiple times.
        # But we stop at the first one. So it's fine.
        # However, if we don't enforce order, the search space is larger.
        # Standard Bron-Kerbosch uses pivot and ordering.
        
        # Let's try a simple DFS with bitsets.
        # To ensure we don't miss anything, we don't strictly need ordering if we just want existence.
        # But to be fast, we should prune.
        
        # Recursive call: include v
        res = search(new_cands, clique + [v], k_remaining - 1)
        if res:
            return res
            
        # Exclude v
        # Remove v from candidates
        new_cands = candidates & ~v_bit
        res = search(new_cands, clique, k_remaining)
        if res:
            return res
            
        return None

    return search(candidates, [], k)

def is_valid(adj, n):
    if find_k4(adj, n):
        return False
    full_mask = (1 << n) - 1
    cadj = [0] * n
    for i in range(n):
        cadj[i] = (full_mask ^ (1 << i)) & ~adj[i]
    if find_k_clique(cadj, 6, n):
        return False
    return True
