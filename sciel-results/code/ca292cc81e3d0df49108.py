"""Simulated Annealing approach for 4-colouring K_n without monochromatic triangles."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def get_seed_colors(n):
        """
        Generate a valid 4-coloring for n vertices.
        Uses the known 32-vertex Greenwood-Gleason coloring as a base.
        For n > 32, adds vertices with random colors that avoid immediate triangles.
        """
        if n <= 32:
            # Standard GF(16) construction for 32 vertices
            # We need a valid 4-coloring of K_32.
            # The standard construction uses a quadratic residue or similar.
            # Actually, the seed program used a specific GF(2^4) construction.
            # Let's replicate that logic or use a simpler valid one.
            # A known valid 4-coloring of K_32 comes from the Paley graph of order 16? No, Paley(16) is 4-colorable?
            # Let's just use the logic from the seed program's initialization which was valid for 32.
            # Seed logic:
            # log, x = {}, 1
            # for e in range(15): log[x] = e; x = mul(x, 2)
            # colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
            
            def mul(a, b):
                r = 0
                for i in range(4):
                    if b >> i & 1:
                        r ^= a << i
                for i in range(7, 3, -1):
                    if r >> i & 1:
                        r ^= 0b10011 << (i - 4)
                return r

            log, x = {}, 1
            for e in range(15):
                log[x] = e
                x = mul(x, 2)
                
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    if i // 16 == j // 16:
                        # Inside the same half (0-15 or 16-31)
                        # The seed used log[(i%16)^(j%16)] % 3
                        # This creates a 3-coloring of K_16? No, K_16 is 3-colorable?
                        # R(3,3,3)=17. So K_16 can be 3-colored.
                        # The seed code: log[(i % 16) ^ (j % 16)] % 3
                        # This is a valid 3-coloring of K_16?
                        # Yes, the Paley graph of order 16 is a (16,5,2,2) strongly regular graph.
                        # Its complement is 3-colorable?
                        # Actually, the seed program worked, so this is a valid 3-coloring of K_16.
                        colors.append(log[(i % 16) ^ (j % 16)] % 3)
                    else:
                        # Between the two halves
                        colors.append(3)
            return colors

        # For n > 32, start with 32 valid vertices, then add more greedily
        base_colors = get_seed_colors(32)
        # Convert to adjacency matrix for easier manipulation
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(32):
            for j in range(i + 1, 32):
                c = base_colors[idx]
                adj[i, j] = c
                adj[j, i] = c
                idx += 1
        
        # Add vertices 32 to n-1 greedily
        # For each new vertex v, choose a color for each edge (v, u) such that no triangle is formed.
        # A triangle is formed if we have edges (u, w), (w, v), (v, u) all with same color.
        # When adding v, we have fixed colors for all (u, w) where u, w < v.
        # For each u < v, we need to choose color c_u.
        # If we choose color c for edge (v, u), we must ensure that for all w < u, 
        # if edge (u, w) has color c, then edge (w, v) cannot have color c.
        # This is a constraint satisfaction problem.
        # Greedy: For each u, pick a random valid color.
        
        # Let's maintain the adjacency matrix.
        # For v in range(32, n):
        #   for u in range(v):
        #     valid_colors = [0,1,2,3]
        #     for w in range(u):
        #       if adj[u, w] == c:
        #         if adj[w, v] is already set to c:
        #           invalid
        #     This is O(n^3). For n=50, 125k ops. Fast enough.
        
        for v in range(32, n):
            for u in range(v):
                valid = []
                for c in range(4):
                    ok = True
                    # Check if any w < u has adj[u,w]==c and adj[w,v]==c
                    # Note: adj[w,v] is 0 for now (uninitialized) or set if w < v?
                    # Wait, we are filling row v. adj[w,v] for w < v is being filled as we go?
                    # No, we fill u from 0 to v-1.
                    # When we are at u, adj[w, v] for w < u has ALREADY been set.
                    # adj[w, v] for w > u is NOT set yet.
                    # So we only need to check w < u.
                    for w in range(u):
                        if adj[u, w] == c and adj[w, v] == c:
                            ok = False
                            break
                    if ok:
                        valid.append(c)
                
                if not valid:
                    # This shouldn't happen if we start from a valid 32-vertex coloring
                    # and n is not too large, but if it does, pick random
                    c = random.randint(0, 3)
                else:
                    # Pick randomly from valid to introduce diversity
                    c = random.choice(valid)
                
                adj[u, v] = c
                adj[v, u] = c
        
        # Flatten
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(int(adj[i, j]))
        return colors

    def count_conflicts(adj):
        """
        Count monochromatic triangles.
        For each color c, count triangles in the graph G_c.
        Number of triangles in G_c = trace(A_c^3) / 6.
        Using numpy matrix multiplication.
        """
        n = adj.shape[0]
        total_conflicts = 0
        for c in range(4):
            # A_c is n x n boolean matrix
            A = (adj == c).astype(np.int8)
            # A^2
            A2 = A @ A
            # trace(A^3) = sum(A2 * A) -- wait, trace(A^3) = sum_{i,j} (A^2)_{ij} A_{ji} = sum(A2 * A.T)
            # Since A is symmetric, A.T = A.
            # So trace(A^3) = np.sum(A2 * A)
            # Number of triangles = trace(A^3) / 6
            tri = np.sum(A2 * A) / 6
            total_conflicts += tri
        return int(total_conflicts)

    def simulate_annealing(n, max_time=100):
        # Initialize
        adj = np.zeros((n, n), dtype=np.int8)
        # Use greedy initialization
        colors = get_seed_colors(n)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i, j] = c
                adj[j, i] = c
                idx += 1
        
        current_conflicts = count_conflicts(adj)
        best_adj = adj.copy()
        best_conflicts = current_conflicts
        
        T = 1.0
        cooling_rate = 0.999
        # Number of steps per second target?
        # Let's just run until time is up.
        
        # Precompute indices for faster access?
        # n is small (<= 50).
        
        while time.time() < deadline - 5:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            old_c = adj[i, j]
            new_c = random.randint(0, 3)
            if old_c == new_c:
                continue
            
            # Change
            adj[i, j] = new_c
            adj[j, i] = new_c
            
            new_conflicts = count_conflicts(adj)
            delta = new_conflicts - current_conflicts
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_conflicts = new_conflicts
                if new_conflicts < best_conflicts:
                    best_conflicts = new_conflicts
                    best_adj = adj.copy()
                    if best_conflicts == 0:
                        break
            else:
                # Revert
                adj[i, j] = old_c
                adj[j, i] = old_c
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
        
        return best_adj, best_conflicts

    # Strategy: Try to find a valid coloring for n=50.
    # If we can't, try n=49, etc.
    # Since the seed program returns 32, and the record is 50, we should aim for 50.
    # If we fail to get 0 conflicts for 50, we return the best we have (which might be invalid? 
    # No, the task says "No colour class may contain a triangle". 
    # If we return an invalid coloring, the score is 0.
    # So we MUST return a valid coloring.
    # The seed program returns a valid 32-vertex coloring.
    # If my SA fails to find 0 conflicts for n=50, I should fall back to a known valid smaller n.
    # Or, I can run SA on n=32 to ensure I have a valid one, then try to extend?
    # Better: Run SA on n=50. If it finds 0 conflicts, return it.
    # If not, try n=49, ..., down to 32.
    # Since SA is stochastic, it might take time.
    # Let's allocate time.
    
    # To guarantee a non-zero score, we MUST return a valid coloring.
    # The seed program's 32-vertex coloring is valid.
    # Let's generate it first as a fallback.
    fallback_adj = None
    fallback_n = 32
    colors_32 = get_seed_colors(32)
    adj_32 = np.zeros((32, 32), dtype=np.int8)
    idx = 0
    for i in range(32):
        for j in range(i + 1, 32):
            c = colors_32[idx]
            adj_32[i, j] = c
            adj_32[j, i] = c
            idx += 1
    if count_conflicts(adj_32) == 0:
        fallback_adj = adj_32
        fallback_n = 32
    
    # Now try to find a valid coloring for larger n.
    # We will try n=50, then 49, etc.
    # But SA is expensive. Let's give it a chance for n=50.
    # If we don't find 0 conflicts in time, we return the fallback.
    
    # Actually, we can try to improve the n=32 coloring? No, it's already optimal for its size.
    # We want to maximize n.
    
    # Let's try n=50 with a time budget.
    # If we find 0 conflicts, great.
    # If not, we return the fallback.
    
    # To be safe, let's try n=50, 49, 48... with decreasing time?
    # Or just try n=50 for 100s. If it fails, return 32.
    # This might score 1.0 (32/32) if we fail, but 50/32 = 1.56 if we succeed.
    
    # Let's optimize the conflict counting.
    # Current count_conflicts is O(n^3) roughly (matrix mult).
    # For n=50, 50^3 = 125,000 ops. Very fast.
    # So we can do many steps.
    
    best_adj = fallback_adj
    best_n = fallback_n
    
    # Try n=50
    target_n = 50
    adj_50, conf_50 = simulate_annealing(target_n, max_time=100)
    if conf_50 == 0:
        best_adj = adj_50
        best_n = target_n
    else:
        # If we didn't find 0 conflicts, maybe we can try a smaller n?
        # But SA on 50 is likely to find a valid one if one exists (and it does, Chung's).
        # If it fails, it's likely due to getting stuck in a local minimum.
        # Let's try a few more times with different seeds?
        # We have ~10s left.
        if time.time() < deadline - 5:
            # Try n=49
            adj_49, conf_49 = simulate_annealing(49, max_time=5)
            if conf_49 == 0:
                best_adj = adj_49
                best_n = 49
            elif time.time() < deadline - 2:
                # Try n=48
                adj_48, conf_48 = simulate_annealing(48, max_time=3)
                if conf_48 == 0:
                    best_adj = adj_48
                    best_n = 48
    
    # Convert best_adj to colors list
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            colors.append(int(best_adj[i, j]))
    
    return {'n': best_n, 'colors': colors}
# EVOLVE-BLOCK-END
