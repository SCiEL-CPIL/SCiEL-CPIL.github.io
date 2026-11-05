import time
import sys
import random

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def generate_edges(n, p, seed):
        """Generate edges of a random graph with given parameters."""
        edges = set()
        rng = random.Random(seed)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges_set = generate_edges(n, p, seed)
        
        # Build adjacency bitmasks
        # adj[i] is a bitmask of neighbors of i
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # remaining[i] is a bitmask of edges incident to i that are not yet covered
        # Initially, remaining[i] == adj[i]
        remaining = adj[:]
        
        partition = []
        
        # Helper to get list of set bits in a mask
        def get_bits(mask):
            bits = []
            while mask:
                # Get index of lowest set bit
                lsb = mask & -mask
                idx = lsb.bit_length() - 1
                bits.append(idx)
                mask ^= lsb
            return bits

        while True:
            if time.time() - start_time > DEADLINE:
                # Fallback: extract remaining edges as 2-cliques
                # This is slow, so we hope we don't hit this often or with many edges
                # But for safety, we must return *something* valid
                # We can iterate through all pairs
                for u in range(n):
                    mask = remaining[u]
                    # Only look at v > u to avoid duplicates? 
                    # Actually, if we just extract pairs, we need to be careful.
                    # Since we are in fallback, correctness is key.
                    for v in range(u + 1, n):
                        if (remaining[u] >> v) & 1:
                            partition.append([u, v])
                            # We don't need to clear remaining if we are just dumping out
                break

            # Check if any edges remain
            has_edges = False
            for u in range(n):
                if remaining[u]:
                    has_edges = True
                    break
            
            if not has_edges:
                break

            # 1. Find vertex with max degree in remaining graph
            max_deg = -1
            seed_v = -1
            for u in range(n):
                deg = bin(remaining[u]).count('1')
                if deg > max_deg:
                    max_deg = deg
                    seed_v = u
            
            if seed_v == -1:
                break

            # 2. Greedily build a clique containing seed_v
            # Start with seed_v
            clique = [seed_v]
            
            # Candidates for next vertex: neighbors of seed_v in remaining graph
            candidates_mask = remaining[seed_v]
            
            while candidates_mask:
                # Find vertex in candidates with max degree *within candidates*
                # Note: The degree of v in the induced subgraph of candidates is 
                # popcount(remaining[v] & candidates_mask)
                best_v = -1
                best_deg = -1
                
                # Iterate through bits in candidates_mask
                temp_mask = candidates_mask
                while temp_mask:
                    lsb = temp_mask & -temp_mask
                    v = lsb.bit_length() - 1
                    # Count neighbors of v that are in candidates_mask
                    # We use remaining[v] to ensure the edge exists in the remaining graph
                    deg = bin(remaining[v] & candidates_mask).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                    temp_mask ^= lsb
                
                if best_v == -1:
                    break
                
                clique.append(best_v)
                
                # Remove best_v from candidates
                candidates_mask &= ~(1 << best_v)
                
                # Intersect candidates with neighbors of best_v in remaining graph
                candidates_mask &= remaining[best_v]
            
            # 3. Add clique to partition
            partition.append(clique)
            
            # 4. Remove all edges in this clique from remaining graph
            # For each pair in clique, remove the edge
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u = clique[i]
                    v = clique[j]
                    remaining[u] &= ~(1 << v)
                    remaining[v] &= ~(1 << u)
            
            # Optional optimization: Clean up remaining masks?
            # Not strictly necessary, but keeps degrees accurate if we rely on them.
            # The degree calculation `bin(remaining[u]).count('1')` is always accurate
            # because we clear bits as we go.

        return partition

    result1 = solve_instance(32, 0.6, 601)
    if time.time() - start_time > DEADLINE:
        # Fallback for second instance
        edges2 = generate_edges(64, 0.55, 602)
        result2 = [[u, v] for (u, v) in edges2]
        return [result1, result2]
    
    result2 = solve_instance(64, 0.55, 602)
    return [result1, result2]
