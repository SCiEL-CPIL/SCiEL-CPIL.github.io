import time
import random
import sys
import itertools

# EVOLVE-BLOCK-START

def construct():
    """
    Strategy: Use the known 35-vertex Ramsey graph (published by Exoo) for R(4,6) > 35.
    This is a well-known extremal graph. We can either:
    1. Hard-code the 35-vertex graph (if we have it memorized/derived)
    2. Use a constructive method to build it
    3. Use a fast local search to find a valid 35-vertex graph
    
    The most reliable approach is to use a known valid 35-vertex graph.
    Let me think about what structure might work...
    
    Actually, let me use a different approach: a fast local search that starts from
    a good seed and quickly converges. The key insight is that we need:
    - No K4 in G
    - No independent set of size 6 in G (i.e., no K6 in complement)
    
    Let me implement an efficient local search with proper violation detection.
    
    Alternative: Use the Wagner graph or other known Ramsey graphs as seeds.
    
    Let me try a structured approach:
    - Start with a 29-vertex valid graph (Paley(29) works? Let me check: Paley(29) has 
      clique number? 29 is 1 mod 4, so Paley(29) exists. Its properties...)
    - Actually, let me just use a fast simulated annealing with proper violation counting.
    
    The key is to make the score function fast and the search efficient.
    """
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit, 10s buffer)
    
    def count_violations(adj, n):
        """Count K4s and independent 6-sets using bit operations."""
        mask = (1 << n) - 1
        adj_bar = [(~adj[i]) & mask for i in range(n)]
        
        # Count K4s: For each vertex i, count edges in G[N(i)]
        # Each K4 contributes 12 to the sum (4 vertices * 3 edges among other 3)
        k4_sum = 0
        for i in range(n):
            ni = adj[i]
            ec = 0
            temp = ni
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                ec += (ni & adj[j]).bit_count()
                temp ^= lsb
            k4_sum += ec
        k4_count = k4_sum // 12
        
        # Count independent 6-sets: For each vertex i, count edges in complement[N_bar(i)]
        # Each independent 6-set contributes 60 to the sum (6 vertices * 10 edges among other 5)
        indep_sum = 0
        for i in range(n):
            ni_bar = adj_bar[i]
            ec = 0
            temp = ni_bar
            while temp:
                lsb = temp & -temp
                j = lsb.bit_length() - 1
                ec += (ni_bar & adj_bar[j]).bit_count()
                temp ^= lsb
            indep_sum += ec
        indep_count = indep_sum // 60
        
        return k4_count, indep_count
    
    def is_valid_fast(adj, n):
        """Check if graph has no K4 and no independent 6-set."""
        k4, indep = count_violations(adj, n)
        return k4 == 0 and indep == 0
    
    def random_graph(n, p=0.5):
        """Generate a random graph with edge probability p."""
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    def flip_edge(adj, u, v):
        """Flip the edge between u and v."""
        if adj[u] & (1 << v):
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    # Strategy: Try multiple random restarts with simulated annealing
    best_valid_n = 0
    best_valid_adj = None
    
    # Start with n=35 (the target), then fall back to smaller n if needed
    for target_n in [35, 34, 33, 32, 31, 30, 29]:
        if time.time() > deadline - 5:
            break
        
        n = target_n
        # Try multiple random starts
        found = False
        max_starts = 20
        
        for start_idx in range(max_starts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with a random graph
            adj = random_graph(n, p=0.5)
            cur_k4, cur_indep = count_violations(adj, n)
            cur_score = cur_k4 + cur_indep
            
            if cur_score == 0:
                found = True
                break
            
            # Simulated annealing
            temperature = 2.0
            cooling = 0.99
            no_improve = 0
            best_score = cur_score
            best_adj = [a for a in adj]
            
            while time.time() < deadline - 10:
                if cur_score == 0:
                    found = True
                    break
                
                # Pick a random edge to flip
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                # Flip edge
                flip_edge(adj, u, v)
                new_k4, new_indep = count_violations(adj, n)
                new_score = new_k4 + new_indep
                delta = new_score - cur_score
                
                if delta < 0:
                    # Accept improvement
                    cur_score = new_score
                    no_improve = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_adj = [a for a in adj]
                else:
                    # Accept worsening with probability exp(-delta/T)
                    if delta == 0 or random.random() < pow(2.718281828, -delta / temperature):
                        cur_score = new_score
                        no_improve += 1
                    else:
                        # Revert
                        flip_edge(adj, u, v)
                        no_improve += 1
                
                temperature *= cooling
                if temperature < 0.05:
                    break
                
                if no_improve > 2000:
                    break
            
            if cur_score == 0:
                found = True
                break
            
            # Keep the best found in this restart
            if best_score < best_score_global if 'best_score_global' in dir() else True:
                if 'best_score_global' not in dir():
                    best_score_global = best_score
                    best_adj_global = best_adj
                else:
                    if best_score < best_score_global:
                        best_score_global = best_score
                        best_adj_global = best_adj
        
        if found:
            # Verify and return
            if is_valid_fast(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            else:
                # Double-check
                k4, indep = count_violations(adj, n)
                if k4 == 0 and indep == 0:
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i] & (1 << j):
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
    
    # Fallback: Use a small known valid graph
    # C5 (5-cycle) has no K4 and no independent set of size > 2, so it's valid
    # But we want to maximize n. Let's use a larger known valid graph.
    # Paley(17) is a known valid graph for R(4,6) > 17
    # Actually, let me construct a simple valid graph
    
    # The 9-vertex graph: complete bipartite K_{4,5} complement? No, that has K5.
    # Let's use a Turan-type construction or just return a small valid graph.
    
    # Simple valid graph: complement of C5 is C5, which is valid (no K4, no independent 6)
    # For n=17, Paley(17):
    squares_17 = set()
    for x in range(1, 17):
        squares_17.add((x * x) % 17)
    
    adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares_17:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Verify
    if is_valid_fast(adj, 17):
        edges = []
        for i in range(17):
            for j in range(i + 1, 17):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return {'n': 17, 'edges': edges}
    
    # Ultimate fallback: C5
    edges = [[0,1],[1,2],[2,3],[3,4],[4,0]]
    return {'n': 5, 'edges': edges}

# EVOLVE-BLOCK-END
