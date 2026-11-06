import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    def verify(n, edges):
        """
        Verify the graph is C4-free and has min degree >= n-39.
        Returns True if valid.
        """
        if n < 1 or n > 47:
            return False
        target_deg = max(0, n - 39)
        
        # Build adjacency matrix
        adj = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            if u < 0 or u >= n or v < 0 or v >= n or u == v:
                return False
            adj[u, v] = 1
            adj[v, u] = 1
        
        # Check min degree
        degrees = adj.sum(axis=1)
        if np.min(degrees) < target_deg:
            return False
            
        # Check C4-free via matrix multiplication
        A = adj.astype(np.int32)
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        if np.any(A2 >= 2):
            return False
            
        return True

    def is_safe_addition(adj, u, v):
        """
        Check if adding edge (u,v) creates a C4.
        This is safe if u and v have fewer than 2 common neighbors.
        """
        common = np.sum(adj[u] & adj[v])
        return common < 2

    def greedy_c4_free(n, target_deg, seed_offset=0):
        """
        Greedy algorithm to construct a C4-free graph with high minimum degree.
        Uses a smart ordering and edge selection heuristic.
        """
        rng = random.Random(seed_offset)
        adj = np.zeros((n, n), dtype=np.int32)
        degrees = np.zeros(n, dtype=np.int32)
        
        # We want to maximize the minimum degree.
        # Strategy: Repeatedly add edges that don't create C4s, prioritizing vertices with lower current degree.
        # To avoid getting stuck in a local minimum of low min-degree, we can use a "potential" function or just greedy.
        
        # Better strategy:
        # 1. Initialize empty graph.
        # 2. While possible, add an edge (u,v) such that:
        #    a. is_safe_addition(u,v) is True
        #    b. u and v are chosen to balance degrees (minimize max degree increase, or target low-degree nodes)
        
        # Let's try a "conflict-avoiding" greedy.
        # Pre-compute all potential edges? No, n is small.
        
        # To help with min degree, we should try to ensure every vertex reaches target_deg.
        # Let's iterate:
        # Find vertex with min degree. Try to add an edge to it from a safe candidate.
        # If we can't add an edge to the min-degree vertex, move to next min.
        # If we get stuck, we might need to restart with different randomization.
        
        # However, simple greedy might get stuck. Let's use a slightly randomized greedy.
        
        # Order vertices randomly? Or keep them fixed but randomize tie-breaking?
        # Let's try:
        # For step in range(max_steps):
        #   Find pair (u,v) not connected, safe to add, and (u,v) has lowest sum of degrees? 
        #   Or (u,v) where min(deg[u], deg[v]) is high? 
        #   Actually, to boost min degree, we want to connect low-degree nodes.
        #   But low-degree nodes might have few safe options.
        
        # Heuristic:
        # 1. Compute all safe pairs.
        # 2. Pick one randomly from the safe pairs that involves a vertex with degree < target_deg (if any exist).
        # 3. If no such pairs, pick random safe pair.
        # 4. Add it.
        # 5. Stop when no safe pairs left or max steps reached.
        
        # This is O(N^3) per step if we recompute safe pairs.
        # N=47, safe pairs ~ O(N^2). Recomputing is fast.
        
        max_steps = n * n * 10
        for _ in range(max_steps):
            # Find all safe pairs
            # We want to prioritize adding edges to vertices with deg < target_deg
            low_deg_vertices = [i for i in range(n) if degrees[i] < target_deg]
            
            candidates = []
            # To speed up, we can just check a subset or use matrix ops
            # But n is small, let's just do a quick check.
            
            # Optimization: Only check pairs involving low-degree vertices first?
            # Or just check all pairs?
            # N=47, ~1100 pairs. Checking each is O(N) bit op. 1100*47 ~ 50k ops.
            # 50k * 2000 steps = 100M ops. Might be too slow in Python loop.
            # Let's use numpy to check safety for all pairs at once?
            # A2 = A @ A gives common neighbors.
            # Safe if A2[i,j] < 2 and A[i,j] == 0.
            
            A = adj
            A2 = A @ A
            np.fill_diagonal(A2, 0)
            
            # Mask: not connected and common neighbors < 2
            mask = (A == 0) & (A2 < 2)
            
            # Get list of candidate pairs
            # We want to bias towards low degree vertices.
            # Score of pair (i,j) = -degrees[i] - degrees[j] (lower degree is better)
            # Or just random selection from candidates.
            
            # Let's create a list of indices
            # To avoid building a huge list every time, we can use np.argwhere
            candidates_idx = np.argwhere(mask)
            
            if len(candidates_idx) == 0:
                break # No more edges can be added
            
            # If we have low-degree vertices, filter candidates to include at least one of them?
            # This might be expensive to filter.
            # Instead, assign weights.
            
            # Simple random pick from candidates
            # But this might ignore low-degree vertices if high-degree ones have more options.
            # Let's do a 2-phase pick:
            # 1. Check if any candidate involves a low-degree vertex.
            if low_deg_vertices:
                low_deg_set = set(low_deg_vertices)
                # Filter candidates
                filtered = []
                for i, j in candidates_idx:
                    if i in low_deg_set or j in low_deg_set:
                        filtered.append((i, j))
                
                if filtered:
                    # Pick randomly from filtered
                    i, j = rng.choice(filtered)
                    adj[i, j] = 1
                    adj[j, i] = 1
                    degrees[i] += 1
                    degrees[j] += 1
                    continue
            
            # Pick randomly from all candidates
            i, j = rng.choice(candidates_idx)
            adj[i, j] = 1
            adj[j, i] = 1
            degrees[i] += 1
            degrees[j] += 1
            
        # Check if we met the requirement
        if np.min(degrees) >= target_deg:
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i,j]]
            if verify(n, edges):
                return {'n': n, 'edges': edges}
                
        return None

    def simulated_annealing(n, target_deg, time_budget):
        """
        Simulated Annealing to find a C4-free graph with min degree >= target_deg.
        """
        if time.time() - start_time > deadline - 5:
            return None
            
        # Initialize with a random graph of moderate density
        p = max(0.05, target_deg / (n - 1) * 1.5)
        
        adj = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i, j] = 1
                    adj[j, i] = 1
                    
        def get_energy(adj):
            A = adj.astype(np.int32)
            A2 = A @ A
            np.fill_diagonal(A2, 0)
            
            c4_count = np.sum(A2 >= 2)
            degrees = A.sum(axis=1)
            deficit = np.sum(np.maximum(0, target_deg - degrees))
            
            energy = c4_count * 1000 + deficit
            return energy, c4_count, deficit

        current_energy, _, _ = get_energy(adj)
        best_energy = current_energy
        best_adj = adj.copy()
        
        T = 100.0
        cooling_rate = 0.995
        min_T = 0.01
        
        while T > min_T:
            if time.time() - start_time > deadline - 5:
                break
                
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            old_val = adj[i, j]
            adj[i, j] = 1 - old_val
            adj[j, i] = adj[i, j]
            
            new_energy, _, _ = get_energy(adj)
            delta_e = new_energy - current_energy
            
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = adj.copy()
            else:
                adj[i, j] = old_val
                adj[j, i] = old_val
                
            T *= cooling_rate
            
        if verify(n, [(i, j) for i in range(n) for j in range(i+1, n) if best_adj[i,j]]):
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if best_adj[i,j]]
            return {'n': n, 'edges': edges}
            
        return None

    # Main logic
    # Try n=47, then 46, etc.
    
    time_alloc = {
        47: 50,
        46: 40,
        45: 20,
        44: 10,
        43: 5
    }
    
    # 1. Try Greedy for all n (fast)
    for n in [47, 46, 45, 44, 43]:
        if time.time() - start_time > deadline - 2:
            break
        target_deg = max(0, n - 39)
        # Try a few seeds
        for seed in range(10):
            if time.time() - start_time > deadline - 2:
                break
            res = greedy_c4_free(n, target_deg, seed)
            if res:
                return res

    # 2. Try Simulated Annealing for high n (slower but more powerful)
    for n in [47, 46, 45]:
        if time.time() - start_time > deadline - 2:
            break
        target_deg = max(0, n - 39)
        res = simulated_annealing(n, target_deg, time_alloc.get(n, 5))
        if res:
            return res

    # Fallback
    return {'n': 39, 'edges': []}
