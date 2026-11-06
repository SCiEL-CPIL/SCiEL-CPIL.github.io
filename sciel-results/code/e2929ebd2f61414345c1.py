import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def has_independent_set(adj, n, size):
        """
        Returns True if there is an independent set of size 'size'.
        adj is a list of bitsets.
        """
        # candidates: bitset of vertices that can be added
        # We use a recursive search with pruning.
        # To speed up, we can use a greedy heuristic to order vertices or just use bitsets.
        
        def search(candidates, current_size):
            if current_size >= size:
                return True
            if not candidates:
                return False
            if candidates.bit_count() < size - current_size:
                return False
            
            # Pick a vertex. 
            # Heuristic: Pick the vertex with the highest degree in the induced subgraph?
            # Or just the lowest index to be deterministic and fast.
            # For n=40, lowest index is fine.
            v_bit = candidates & -candidates
            v = v_bit.bit_length() - 1
            
            # Option 1: Include v
            # New candidates: remove v and its neighbors
            new_candidates = candidates & ~adj[v] & ~v_bit
            if search(new_candidates, current_size + 1):
                return True
            
            # Option 2: Exclude v
            new_candidates = candidates & ~v_bit
            if search(new_candidates, current_size):
                return True
                
            return False

        full_candidates = (1 << n) - 1
        return search(full_candidates, 0)

    def is_triangle_free_circulant(S, n):
        """
        S is a set of differences in {1, ..., n//2}.
        Checks if the circulant graph defined by S is triangle-free.
        """
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                # The sum a+b creates a potential triangle if the "closing" difference is in S.
                # Vertices 0, a, a+b.
                # Edges: (0,a) diff a, (a, a+b) diff b, (0, a+b) diff a+b.
                # We need to check if the undirected difference of (0, a+b) is in S.
                c = (a + b) % n
                if c == 0:
                    # a+b = n, so 0 and a+b are the same vertex? No, 0 and n are 0.
                    # If a+b = n, then 0 and a+b are the same vertex, so no edge.
                    # But wait, if a+b = n, then the vertices are 0, a, 0. Not a triangle.
                    continue
                undirected_c = min(c, n - c)
                if undirected_c in S:
                    return False
        return True

    def build_graph_from_circulant(S, n):
        adj = [0] * n
        for i in range(n):
            for d in S:
                j1 = (i + d) % n
                j2 = (i - d) % n
                adj[i] |= (1 << j1)
                adj[i] |= (1 << j2)
        return adj

    # Try n from 40 down to 39
    for n in range(40, 38, -1):
        if time.time() > deadline:
            break
            
        max_diff = n // 2
        # We search for a set S of differences.
        # Randomized search over subsets of {1, ..., max_diff}
        
        # To make it more effective, we can start with a random set and modify.
        # Or just try random subsets.
        # Number of subsets is 2^max_diff. For n=40, 2^20 = 1M.
        # We can iterate through them or sample.
        
        # Let's sample random subsets.
        # Bias towards larger S to reduce independence number?
        # But S must be sum-free.
        
        found = False
        # Try 10000 random subsets
        for _ in range(10000):
            if time.time() > deadline:
                break
            
            # Generate random S
            # Probability of including a diff d. 
            # Let's try p=0.5 initially.
            S = set()
            for d in range(1, max_diff + 1):
                if random.random() < 0.5:
                    S.add(d)
            
            # Check triangle-free
            if not is_triangle_free_circulant(S, n):
                continue
            
            # Check independence number
            adj = build_graph_from_circulant(S, n)
            
            # Check if independent set of size 10 exists
            if not has_independent_set(adj, n, 10):
                # Valid graph found!
                edges = []
                for i in range(n):
                    for d in S:
                        j = (i + d) % n
                        if i < j:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        
        # If random search fails, try a more systematic approach?
        # Or just move to smaller n.
        # For n=39, we know it exists.
        # For n=40, it might not exist.
        
    # Fallback: Return a known valid graph for n=26 or similar
    # The seed program returned n=26.
    # We should ensure we return something valid.
    # Let's construct a small valid graph if the search fails.
    # A simple valid graph: Complete graph K_9 is not triangle free.
    # Cycle C_5 is triangle free, alpha=2.
    # We can just return the seed result if we fail to find better.
    # But the seed is n=26.
    
    # Let's try to construct a valid graph for n=39 deterministically if random fails?
    # Or just return the best found.
    
    # If we reach here, we didn't find n=40 or 39 in the time limit.
    # We should return a valid graph.
    # Let's use a simple construction for a smaller n to be safe.
    # E.g., n=26.
    
    n_seed = 26
    adj_seed = [0] * n_seed
    # Use a known valid configuration or just return empty? No, must be valid.
    # The seed program used a specific construction.
    # Let's replicate the seed's fallback logic or just return a small valid graph.
    # Actually, the seed program's `build_seed_graph` was:
    # if (j - i) % n_seed % 3 == 1: add edge.
    # Let's check if that is valid.
    # n=26.
    # It's better to just return the seed result if we fail.
    
    # Re-implementing the seed's fallback to ensure validity
    n_seed = 26
    adj_seed = [0] * n_seed
    for i in range(n_seed):
        for j in range(i+1, n_seed):
            if (j - i) % 3 == 1:
                adj_seed[i] |= (1 << j)
                adj_seed[j] |= (1 << i)
                
    edges = [[i, j] for i in range(n_seed) for j in range(i+1, n_seed) if (adj_seed[i] >> j) & 1]
    return {'n': n_seed, 'edges': edges}
