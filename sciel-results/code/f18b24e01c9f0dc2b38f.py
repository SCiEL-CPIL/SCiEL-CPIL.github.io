import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_result = None
    best_n = 0

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        if time.time() > deadline - 10:
            break

        min_deg_needed = n - 39

        # Multiple restarts
        found = False
        max_restarts = 100

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            result = _build_c4_free_repair(n, min_deg_needed, restart)
            if result is not None:
                edges, min_deg = result
                if min_deg >= min_deg_needed:
                    if _verify_c4_free(n, edges):
                        if n > best_n:
                            best_n = n
                            best_result = {'n': n, 'edges': edges}
                        found = True
                        break

        if found:
            break

    if best_result is None:
        # Fallback: trivial graph
        best_result = {'n': 39, 'edges': []}

    return best_result

def _build_c4_free_repair(n, min_deg_needed, seed):
    """
    Build a C4-free graph by:
    1. Generating a random dense graph.
    2. Removing edges to eliminate C4s.
    3. Adding edges to boost minimum degree.
    """
    random.seed(seed * 1000003 + 7)
    
    # Target degree slightly above min_deg_needed to allow for removals
    target_deg = min_deg_needed + 2
    p = target_deg / (n - 1)
    
    # Initialize graph
    adj = [set() for _ in range(n)]
    
    # Phase 1: Random initialization
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                adj[u].add(v)
                adj[v].add(u)
                
    # Phase 2: Remove edges to make C4-free
    # We look for C4s and remove edges.
    # A C4 is u-v-w-x-u. u and w share v and x.
    # So for any pair (u, w), if they have >= 2 common neighbors, we have a C4.
    
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for w in range(u + 1, n):
                common = adj[u] & adj[w]
                if len(common) > 1:
                    # Remove an edge incident to u or w to break the C4
                    # Remove edge (u, v) where v is a common neighbor
                    # Choose v randomly or the one with highest degree to balance?
                    # Let's remove (u, v) randomly from common neighbors
                    v = random.choice(list(common))
                    adj[u].remove(v)
                    adj[v].remove(u)
                    changed = True
                    break
            if changed:
                break # Restart scan from beginning as graph changed

    # Check if we already met the requirement
    degrees = [len(adj[i]) for i in range(n)]
    min_deg = min(degrees)
    if min_deg >= min_deg_needed:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return (edges, min_deg)

    # Phase 3: Add edges to boost minimum degree
    # Try to add edges that do not create C4s.
    # An edge (u, v) can be added if |N(u) \cap N(v)| <= 1.
    
    for _ in range(500): # Limit iterations
        if time.time() > deadline - 5:
            break
            
        degrees = [len(adj[i]) for i in range(n)]
        min_deg = min(degrees)
        if min_deg >= min_deg_needed:
            edges = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.append([i, j])
            return (edges, min_deg)
            
        # Find vertex with min degree
        low_v = degrees.index(min_deg)
        
        # Find non-neighbors of low_v
        non_neighbors = [v for v in range(n) if v != low_v and v not in adj[low_v]]
        if not non_neighbors:
            continue
            
        random.shuffle(non_neighbors)
        
        added = False
        for v in non_neighbors:
            common = adj[low_v] & adj[v]
            if len(common) <= 1:
                adj[low_v].add(v)
                adj[v].add(low_v)
                added = True
                break
        
        if not added:
            # Try to swap edges? Or just break and restart
            break

    # Final check
    degrees = [len(adj[i]) for i in range(n)]
    min_deg = min(degrees)
    if min_deg >= min_deg_needed:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return (edges, min_deg)
        
    return None

def _verify_c4_free(n, edges):
    """Verify the graph has no C4."""
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for (u, v) in edges:
        common = adj[u] & adj[v]
        if len(common) >= 2:
            return False
    
    return True

if __name__ == "__main__":
    result = construct()
    print(f"n={result['n']}, edges={len(result['edges'])}")
    if result['n'] > 39:
        n = result['n']
        degs = [0]*n
        for e in result['edges']:
            degs[e[0]] += 1
            degs[e[1]] += 1
        print(f"min_degree={min(degs)}")
        print(f"C4-free: {_verify_c4_free(n, result['edges'])}")
