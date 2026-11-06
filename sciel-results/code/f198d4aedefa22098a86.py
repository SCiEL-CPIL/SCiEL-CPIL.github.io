import time
import random
import numpy as np

def construct():
    """
    Constructs a graph with no K5 and no independent 5-set.
    Uses Simulated Annealing to extend the Paley graph on 37 vertices.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # Adjacency matrix as list of bitmasks
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append((i, j))
    
    # Precompute K4s and independent K4s (barK4s) as bitmasks
    # A K4 is a set of 4 vertices forming a clique.
    # We store them as bitmasks of the 4 vertices.
    k4s = []
    bar_k4s = []
    
    # Helper to find all K4s in current graph
    def find_k4s():
        k4_list = []
        bar_k4_list = []
        # Iterate over all 4-sets? No, that's binom(n,4).
        # For n=40, binom(40,4) = 91k. This is fine.
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] & (1 << j)): continue # Must be edge
                for k in range(j+1, n):
                    if not (adj[i] & (1 << k)): continue
                    if not (adj[j] & (1 << k)): continue
                    # i, j, k form a triangle
                    # Check if l connects to i, j, k
                    common = adj[i] & adj[j] & adj[k]
                    # common contains i, j, k themselves? No, adj[i] doesn't have i.
                    # But common might have bits for i,j,k if we weren't careful, but adj[i] has no i.
                    # So common contains vertices connected to all three.
                    # We need l > k to avoid duplicates?
                    # Actually, we just need to find any l.
                    # But we want to store the set.
                    # Let's just iterate l > k.
                    for l in range(k+1, n):
                        if (common & (1 << l)):
                            mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                            k4_list.append(mask)
                            
                    # For bar_k4: i,j,k must be independent? No, they must form an independent set in G.
                    # So NO edges between i,j,k.
                    # Wait, the logic above was for K4.
                    # For barK4, we need 4 vertices with NO edges between them.
                    # Let's do a separate loop for barK4s.
        # Re-do barK4s properly
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j): continue # Must be non-edge
                for k in range(j+1, n):
                    if adj[i] & (1 << k): continue
                    if adj[j] & (1 << k): continue
                    # i, j, k are independent
                    # Need l such that l is not connected to i, j, k
                    # non_common = ~(adj[i] | adj[j] | adj[k]) & ((1 << n) - 1)
                    # But we need l > k
                    for l in range(k+1, n):
                        if not (adj[i] & (1 << l)) and not (adj[j] & (1 << l)) and not (adj[k] & (1 << l)):
                            mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                            bar_k4_list.append(mask)
        return k4_list, bar_k4_list

    k4s, bar_k4s = find_k4s()
    
    def try_extend(target_n):
        """
        Tries to add vertex (target_n - 1) to the graph of size target_n - 1.
        Returns True if successful, and updates adj, edges, k4s, bar_k4s.
        """
        nonlocal adj, edges, k4s, bar_k4s, n
        
        new_v = target_n - 1
        old_n = target_n - 1
        
        # We need to find a set S (neighbors of new_v) such that:
        # 1. No K4 in S
        # 2. No barK4 in complement of S (within old vertices)
        
        # Simulated Annealing
        best_S = None
        best_E = float('inf')
        
        # Initialize S randomly
        S = 0
        for i in range(old_n):
            if random.random() < 0.5:
                S |= (1 << i)
                
        T = 10.0
        T_min = 0.01
        steps = 10000
        
        # Precompute for faster delta calculation?
        # E(S) = sum(1 for k4 in k4s if (k4 & S) == k4) + sum(1 for bk4 in bar_k4s if (bk4 & ~S) == bk4)
        # Note: ~S needs to be masked to old_n bits.
        mask_old = (1 << old_n) - 1
        
        def calc_E(S_mask):
            e = 0
            for k4 in k4s:
                if (k4 & S_mask) == k4:
                    e += 1
            not_S = mask_old & ~S_mask
            for bk4 in bar_k4s:
                if (bk4 & not_S) == bk4:
                    e += 1
            return e
            
        current_E = calc_E(S)
        
        for step in range(steps):
            if current_E == 0:
                return True, S
            
            # Pick a random vertex to flip
            v = random.randint(0, old_n - 1)
            bit = 1 << v
            
            new_S = S ^ bit
            new_E = calc_E(new_S)
            
            delta = new_E - current_E
            if delta < 0 or random.random() < np.exp(-delta / T):
                S = new_S
                current_E = new_E
                
            T *= 0.999 # Cool down
            
            if time.time() > deadline:
                return False, None
                
        return False, None

    # Main loop
    for target_n in range(n + 1, 47):
        if time.time() > deadline:
            break
            
        success = False
        # Try multiple times to extend
        for attempt in range(50):
            if time.time() > deadline:
                break
                
            # We don't actually modify the graph until we are sure?
            # No, if we fail, we just try again with a different random S.
            # The graph state (adj, k4s, etc.) remains the valid graph of size target_n-1.
            # So we just need to find a valid S.
            
            ok, S = try_extend(target_n)
            if ok:
                # Apply the extension
                new_v = target_n - 1
                adj.append(0)
                for i in range(target_n - 1):
                    if S & (1 << i):
                        adj[i] |= (1 << new_v)
                        adj[new_v] |= (1 << i)
                        edges.append((i, new_v))
                        
                # Update k4s and bar_k4s
                # New k4s must include new_v.
                # A new k4 is {new_v, u, w, x} where u,w,x form a triangle in old graph and are connected to new_v.
                # We need to find all triangles in old graph.
                # This might be slow if we do it from scratch.
                # But n is small.
                # Let's just recalculate k4s and bar_k4s from scratch for simplicity and safety.
                n = target_n
                k4s, bar_k4s = find_k4s()
                
                success = True
                break
                
        if not success:
            # Failed to extend
            # We could try to "repair" the current graph to make it easier to extend,
            # but for now, we just stop.
            # Or we can try a different strategy: maybe the current graph is "bad" for extension.
            # We could run a local search on the current graph to find a different valid configuration.
            # But that's complex. Let's just break.
            break

    return n, edges

# Test locally? No, just output.
