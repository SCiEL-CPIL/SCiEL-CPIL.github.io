import time
import random

def construct():
    start_time = time.time()
    time_limit = 290  # Safety margin for 300s limit
    
    # 1. Generate Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    adj = [0] * n
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
    
    def get_cliques4():
        cliques = []
        # Find all K4s
        # A K4 is a set of 4 vertices where all pairs are connected.
        # Iterate over combinations of 4 vertices? 37C4 is ~ 65k. That's fine.
        # But we can be smarter or just use combinations.
        from itertools import combinations
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (adj[a] >> b) & 1 and (adj[a] >> c) & 1 and (adj[a] >> d) & 1 and \
               (adj[b] >> c) & 1 and (adj[b] >> d) & 1 and \
               (adj[c] >> d) & 1:
                mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                cliques.append(mask)
        return cliques

    def get_indep4():
        indep = []
        from itertools import combinations
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if not ((adj[a] >> b) & 1) and not ((adj[a] >> c) & 1) and not ((adj[a] >> d) & 1) and \
               not ((adj[b] >> c) & 1) and not ((adj[b] >> d) & 1) and \
               not ((adj[c] >> d) & 1):
                mask = (1 << a) | (1 << b) | (1 << c) | (1 << d)
                indep.append(mask)
        return indep

    k4_masks = get_cliques4()
    ik4_masks = get_indep4()
    
    def try_add_vertex():
        """
        Attempts to add a new vertex by finding a valid adjacency mask.
        Returns True if successful and updates global state, False otherwise.
        """
        nonlocal n, adj, edges, k4_masks, ik4_masks
        
        # We need to find a mask M for the new vertex v (index n)
        # such that:
        # 1. No K4 in k4_masks is a subset of M.
        # 2. No IK4 in ik4_masks is a subset of ~M (within range n).
        
        # Heuristic: Start with a random mask or a mask based on average degree.
        # Let's try a few random initializations and local search.
        
        all_vertices_mask = (1 << n) - 1
        
        best_mask = None
        best_violations = float('inf')
        
        # Try a few random starts
        for _ in range(10):
            # Random mask with density ~ 0.5
            M = 0
            for i in range(n):
                if random.random() < 0.5:
                    M |= (1 << i)
            
            current_M = M
            current_violations = count_violations(current_M, k4_masks, ik4_masks, n)
            
            # Local search: Flip bits to reduce violations
            no_improve = 0
            for step in range(100):
                if current_violations == 0:
                    best_mask = current_M
                    best_violations = 0
                    break
                if best_violations < current_violations:
                    best_mask = current_M
                    best_violations = current_violations
                
                # Try flipping a random bit
                i = random.randint(0, n - 1)
                new_M = current_M ^ (1 << i)
                new_violations = count_violations(new_M, k4_masks, ik4_masks, n)
                
                if new_violations < current_violations:
                    current_M = new_M
                    current_violations = new_violations
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 20:
                        break
            
            if best_violations == 0:
                break
        
        if best_mask is not None and best_violations == 0:
            # Add the vertex
            new_idx = n
            adj.append(0)
            for i in range(n):
                if (best_mask >> i) & 1:
                    adj[i] |= (1 << new_idx)
                    adj[new_idx] |= (1 << i)
                    edges.append((i, new_idx))
            
            n += 1
            # Update cliques and indep sets
            # New K4s involving the new vertex:
            # A K4 involving new vertex v consists of v and a triangle (K3) in N(v).
            # We need to find all triangles in the subgraph induced by N(v).
            # Similarly for IK4s: v and a triangle in Non-N(v).
            
            # This update is O(n^3) roughly. With n~40, it's fast.
            new_k4s = []
            new_ik4s = []
            
            # Find triangles in N(v)
            neighbors = [i for i in range(n-1) if (best_mask >> i) & 1]
            # Map neighbors to indices in the local list? Or just use global indices.
            # To find K3 in neighbors:
            for a in neighbors:
                for b in neighbors:
                    if a >= b: continue
                    if not ((adj[a] >> b) & 1): continue
                    for c in neighbors:
                        if b >= c: continue
                        if not ((adj[b] >> c) & 1): continue
                        if not ((adj[a] >> c) & 1): continue
                        # Triangle a,b,c exists in N(v)
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << (n-1))
                        new_k4s.append(mask)
                        
            # Find triangles in Non-N(v)
            non_neighbors = [i for i in range(n-1) if not ((best_mask >> i) & 1)]
            for a in non_neighbors:
                for b in non_neighbors:
                    if a >= b: continue
                    if (adj[a] >> b) & 1: continue # Must be non-edge
                    for c in non_neighbors:
                        if b >= c: continue
                        if (adj[b] >> c) & 1: continue
                        if (adj[a] >> c) & 1: continue
                        # Triangle a,b,c exists in Non-N(v)
                        mask = (1 << a) | (1 << b) | (1 << c) | (1 << (n-1))
                        new_ik4s.append(mask)
                        
            k4_masks.extend(new_k4s)
            ik4_masks.extend(new_ik4s)
            
            return True
        return False

    def count_violations(M, k4s, ik4s, n):
        v = 0
        # Check K4s in M
        for mask in k4s:
            if (mask & M) == mask:
                v += 1
        # Check IK4s in ~M
        # ~M within n bits is ((1<<n)-1) ^ M
        non_M = ((1 << n) - 1) ^ M
        for mask in ik4s:
            if (mask & non_M) == mask:
                v += 1
        return v

    # Main loop: try to add vertices
    while time.time() - start_time < time_limit:
        if not try_add_vertex():
            # If we can't add with random restarts, break.
            # We could try more sophisticated search, but let's keep it simple.
            break
        
    return n, edges

if __name__ == "__main__":
    n, e = construct()
    print(n, len(e))
