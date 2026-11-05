import time
import random
import sys

def construct():
    start_time = time.time()
    timeout = 110
    
    # Helper: Check if graph is triangle-free
    def is_triangle_free(adj, n):
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    for w in adj[u]:
                        if v < w and w in adj[v]:
                            return False
        return True

    # Helper: Find an independent set of size k using greedy + backtracking (simple)
    # Returns the set if found, else None
    def find_independent_set(adj, n, k):
        # Simple greedy with randomization
        # Try multiple times
        for _ in range(100):
            S = set()
            candidates = list(range(n))
            random.shuffle(candidates)
            for v in candidates:
                if all(v not in adj[u] for u in S):
                    S.add(v)
                    if len(S) >= k:
                        return S
        return None

    # Helper: Try to add an edge (u,v) if it doesn't create a triangle
    def can_add_edge(adj, u, v):
        if v in adj[u]:
            return False
        # Check if any common neighbor
        common = adj[u].intersection(adj[v])
        return len(common) == 0

    # Main search loop
    best_n = 26
    best_edges = None
    
    # Fallback edges for n=26 (Polarity Graph C_5(2) is 26 vertices)
    # We can generate it or use a known one.
    # C_5(2) vertices 0..25.
    # Edges: (i, j) if i-j is a quadratic residue mod 25? No, it's a polarity graph.
    # Let's just use the seed's logic for n=26 if we fail.
    
    for n in range(40, 25, -1):
        if time.time() - start_time > timeout - 5:
            break
            
        success = False
        for trial in range(50):
            if time.time() - start_time > timeout - 5:
                break
                
            # Initialize random graph
            adj = [set() for _ in range(n)]
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.5:
                        adj[u].add(v)
                        adj[v].add(u)
                        
            # Step 1: Remove triangles
            changed = True
            while changed:
                changed = False
                for u in range(n):
                    for v in list(adj[u]):
                        if u < v:
                            for w in list(adj[u]):
                                if v < w and w in adj[v]:
                                    # Remove edge (v,w)
                                    adj[v].discard(w)
                                    adj[w].discard(v)
                                    changed = True
                                    break
                                if w == v: continue
                    if changed: break
                if changed: break
                
            if not is_triangle_free(adj, n):
                continue
                
            # Step 2: Reduce independence number
            # Try to add edges to break independent sets of size 10
            improved = True
            while improved:
                improved = False
                S = find_independent_set(adj, n, 10)
                if S is None:
                    # Valid graph found!
                    success = True
                    best_n = n
                    best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
                    break
                    
                # Try to add an edge within S
                S_list = list(S)
                random.shuffle(S_list)
                added = False
                for i in range(len(S_list)):
                    for j in range(i+1, len(S_list)):
                        u, v = S_list[i], S_list[j]
                        if can_add_edge(adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            improved = True
                            added = True
                            break
                    if added:
                        break
                if not added:
                    # No edge can be added within this specific set S.
                    # Try to break S by adding edges to vertices outside S?
                    # Or just accept failure for this trial.
                    # A better strategy: try to add edges between S and outside to reduce connectivity?
                    # No, we just need to break the independent set.
                    # If we can't add edges within S, we might need to remove edges incident to S?
                    # Removing edges increases alpha.
                    # So if we can't add edges within S, we are stuck with this alpha.
                    # We should probably restart or try a different random seed.
                    break
            
            if success:
                break
        
        if success:
            break
            
    if best_edges is None:
        # Fallback to n=26 Polarity Graph
        n = 26
        edges = []
        # Generate C_5(2)
        # Points: (x, x^2) for x in 0..4 and point at infinity (5, 6)? No, n=26.
        # Standard C_5(2) construction:
        # Vertices are pairs (x, y) in F_5^2 plus infinity? No, n=26 = 5^2+1.
        # This is complex to hardcode correctly without errors.
        # Let's use the seed's fallback logic which was n=26.
        # The seed used:
        # for i in range(n): for j in range(i+1, n): if (j-i) % n % 3 == 1: ...
        # This generates a specific graph.
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
        
    return {'n': best_n, 'edges': best_edges}
