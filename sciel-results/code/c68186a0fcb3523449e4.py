"""Genetic Algorithm approach for R(3,10) triangle-free graph construction."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _independence_number_bitmask(adj_masks, n, timeout=10.0):
    """
    Compute the independence number of a graph given by adjacency bitmasks.
    Uses branch and bound with bit operations for speed.
    """
    start_time = time.time()
    best = 0
    
    # Pre-compute non-neighbors mask for each vertex (including self? No, self is not a neighbor)
    # non_neigh[i] = mask of vertices NOT adjacent to i, excluding i itself?
    # Actually, for independent set, if we pick i, we remove i and its neighbors.
    # So new_available = available & ~adj_masks[i] & ~(1 << i)
    
    def dfs(available, current_size):
        nonlocal best
        if time.time() - start_time > timeout:
            return
        
        if current_size > best:
            best = current_size
        
        # Upper bound: popcount(available)
        if current_size + bin(available).count('1') <= best:
            return
        
        if available == 0:
            return
        
        # Heuristic: pick vertex with max degree in available subgraph
        # To speed up, we can just pick the lowest set bit, or one with many neighbors in available.
        # Let's try picking a vertex with high degree in the available subgraph.
        # deg(v) = popcount(available & adj_masks[v])
        
        # Iterate through available vertices to find best candidate
        best_v = -1
        best_deg = -1
        temp_avail = available
        while temp_avail:
            lsb = temp_avail & -temp_avail
            v = lsb.bit_length() - 1
            deg = bin(available & adj_masks[v]).count('1')
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp_avail ^= lsb
        
        if best_v == -1:
            return
            
        v_mask = 1 << best_v
        
        # Option 1: Include best_v
        # Remove best_v and its neighbors from available
        new_avail = available & ~adj_masks[best_v] & ~v_mask
        dfs(new_avail, current_size + 1)
        
        # Option 2: Exclude best_v
        new_avail2 = available & ~v_mask
        dfs(new_avail2, current_size)

    dfs((1 << n) - 1, 0)
    return best

def _has_triangle_bitmask(adj_masks, n):
    """Check if graph has a triangle."""
    for i in range(n):
        # For each neighbor j > i
        # We can just check if any neighbor j has a common neighbor k > j
        # Or simpler: for each edge (i,j), check if adj[i] & adj[j] has any bit.
        # To avoid double counting and direction issues, just check all pairs?
        # Efficient: For each i, look at neighbors j > i.
        # Actually, triangle exists if there is i < j < k such that (i,j), (j,k), (i,k) are edges.
        # Equivalent: For each edge (i,j), if adj[i] & adj[j] is not empty, there is a triangle.
        # We just need to check if ANY edge is in a triangle.
        for i in range(n):
            mask_i = adj_masks[i]
            # Only check j > i to avoid redundancy? No, just check if intersection is non-empty.
            # But we must ensure we don't count the edge itself?
            # adj[i] contains j. adj[j] contains i.
            # (adj[i] & adj[j]) will contain vertices k connected to both.
            # If k is i or j, it's not a triangle.
            # But i is not in adj[i], and j is not in adj[j].
            # So if adj[i] & adj[j] is non-empty, it contains some k != i, k != j.
            # Thus, a triangle exists.
            # We can break early.
            if mask_i:
                # Iterate neighbors j of i
                temp = mask_i
                while temp:
                    lsb = temp & -temp
                    j = lsb.bit_length() - 1
                    # Check intersection of neighbors of i and j
                    # We only care if there is a k connected to both.
                    # common = adj_masks[i] & adj_masks[j]
                    # If common is not empty, triangle exists.
                    # Optimization: common & ~(1<<i) & ~(1<<j) ? No, i not in adj[i], j not in adj[j].
                    # So common is exactly the set of vertices forming triangles with edge (i,j).
                    if adj_masks[i] & adj_masks[j]:
                        return True
                    temp ^= lsb
    return False

def _build_andrasfai(n):
    """Construct Andrasfai graph And(k) where n = 3k. Returns adj masks."""
    if n % 3 != 0:
        return None
    k = n // 3
    adj_masks = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            # Edge if (j - i) % 3 == 1
            if (j - i) % 3 == 1:
                adj_masks[i] |= (1 << j)
                adj_masks[j] |= (1 << i)
    return adj_masks

def _fitness(adj_masks, n):
    """
    Returns a score. Higher is better.
    Penalty for triangles is huge.
    Reward for large n and small alpha.
    """
    if _has_triangle_bitmask(adj_masks, n):
        return -1000000
    
    alpha = _independence_number_bitmask(adj_masks, n, timeout=5.0)
    
    # We want alpha < 10.
    # Score: n * 100 - alpha * 10
    # If alpha >= 10, we want to penalize heavily.
    if alpha >= 10:
        return n * 100 - alpha * 1000 
    else:
        return n * 1000 - alpha * 10

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initialize population with known good graphs
    # And(9) -> n=26, alpha=9
    # And(10) -> n=30, alpha=10 (Invalid, alpha=10)
    # Wait, And(k) has alpha = k. So And(9) has alpha 9. And(10) has alpha 10.
    # We need alpha < 10. So max And(k) is And(9) with n=26.
    
    # However, we can combine or modify.
    # Let's start population with random triangle-free graphs and And(9).
    
    pop_size = 50
    population = []
    
    # Add And(9)
    adj_and9 = _build_andrasfai(26)
    population.append(adj_and9)
    
    # Add some random triangle-free graphs
    for _ in range(pop_size - 1):
        n_rand = random.randint(20, 35)
        adj = [0] * n_rand
        # Random edges, check triangle
        for i in range(n_rand):
            for j in range(i + 1, n_rand):
                if random.random() < 0.3:
                    # Check if adding (i,j) creates triangle
                    # common neighbors of i and j
                    if not (adj[i] & adj[j]):
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        population.append(adj)
    
    best_score = -10**9
    best_adj = adj_and9
    best_n = 26
    best_edges = [[i, j] for i in range(26) for j in range(i+1, 26) if (j-i)%26%3 == 1]
    
    # Evaluate initial population
    for adj in population:
        n = len(adj)
        score = _fitness(adj, n)
        if score > best_score:
            best_score = score
            best_adj = adj
            best_n = n
            # Extract edges
            edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges.append([i, j])
                    mask ^= lsb
            best_edges = edges

    # Genetic Algorithm Loop
    while time.time() < deadline - 10:
        new_population = []
        
        # Elitism: keep best
        new_population.append(best_adj)
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline - 10:
                break
                
            # Selection: Tournament
            p1 = random.choice(population)
            p2 = random.choice(population)
            s1 = _fitness(p1, len(p1))
            s2 = _fitness(p2, len(p2))
            
            if s1 > s2:
                parent = p1
            else:
                parent = p2
            
            # Mutation: Flip random edges
            child = parent[:]
            n = len(child)
            
            # Mutation rate
            # Try adding or removing edges
            for _ in range(5): # 5 mutations
                if random.random() < 0.5:
                    # Add edge
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        # Check triangle
                        if not (child[i] & child[j]):
                            child[i] |= (1 << j)
                            child[j] |= (1 << i)
                else:
                    # Remove edge
                    i = random.randint(0, n - 1)
                    mask = child[i]
                    if mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        child[i] &= ~lsb
                        child[j] &= ~(1 << i)
            
            # Sometimes change n? (Harder to do with bitmasks, stick to fixed n per parent)
            # Or create new random individual
            if random.random() < 0.1:
                n_new = random.randint(25, 35)
                child = [0] * n_new
                for i in range(n_new):
                    for j in range(i + 1, n_new):
                        if random.random() < 0.3:
                            if not (child[i] & child[j]):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
            
            new_population.append(child)
        
        # Evaluate new population
        for adj in new_population:
            n = len(adj)
            score = _fitness(adj, n)
            if score > best_score:
                best_score = score
                best_adj = adj
                best_n = n
                edges = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask ^= lsb
                best_edges = edges
        
        # Next generation
        population = new_population

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
