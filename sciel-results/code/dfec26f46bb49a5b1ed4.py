"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import math
    
    N = 512
    # 1. Precompute prime factorizations for 1..N
    # factors[i] will be a dict {prime: exponent}
    factors = [None] * (N + 1)
    factors[1] = {}
    for i in range(2, N + 1):
        x = i
        fac = {}
        d = 2
        while d * d <= x:
            while x % d == 0:
                fac[d] = fac.get(d, 0) + 1
                x //= d
            d += 1
        if x > 1:
            fac[x] = fac.get(x, 0) + 1
        factors[i] = fac

    # 2. Build the conflict graph using bitsets
    # neighbors[i] is a bitset of nodes j such that lcm(i, j) > N
    neighbors = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            # Calculate LCM
            # Merge factors
            fac_i = factors[i]
            fac_j = factors[j]
            lcm_val = 1
            # Get all unique primes
            primes = set(fac_i.keys()).union(fac_j.keys())
            for p in primes:
                exp_i = fac_i.get(p, 0)
                exp_j = fac_j.get(p, 0)
                max_exp = max(exp_i, exp_j)
                if max_exp > 0:
                    lcm_val *= (p ** max_exp)
                    if lcm_val > N:
                        break
            
            if lcm_val > N:
                # Conflict
                neighbors[i] |= (1 << j)
                neighbors[j] |= (1 << i)
                
    # 3. Maximum Independent Set using Branch and Bound with Bitsets
    # We represent the set of available candidates as a bitset.
    # We want to find the largest subset of nodes with no edges between them.
    
    best_size = 0
    best_set = 0 # Bitset of the best independent set found
    
    # All nodes 1..N
    all_nodes = (1 << (N + 1)) - 2 # Bit 0 is 0, bits 1..N are 1
    
    def bit_count(x):
        return bin(x).count('1')
        
    def solve(candidates, current_set, current_size):
        nonlocal best_size, best_set
        
        # Pruning: if even adding all candidates doesn't beat best, stop
        if current_size + bit_count(candidates) <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        # Heuristic: pick a node to branch on.
        # A good strategy is to pick a node with high degree in the induced subgraph of candidates.
        # Or simply pick the lowest set bit for simplicity, but heuristics help a lot.
        # Let's pick a node v from candidates.
        # To improve branching, we can order candidates by degree.
        # However, calculating degrees for every step is expensive.
        # Simple heuristic: Pick a random node? No, deterministic is better for reproducibility.
        # Let's pick the node with the smallest index? Or largest?
        # Actually, a common effective heuristic for MIS is to pick a vertex with maximum degree in the candidate set.
        
        # Let's try picking the first available node for now, but we can optimize.
        # To ensure we finish within 60s, we need a strong bound.
        # The bound used above is weak. A stronger bound is Greedy Coloring.
        # Greedy Coloring of the candidate graph (where edges are conflicts) gives an upper bound on the size of the independent set.
        # If we color the candidate graph with k colors, the max independent set size <= k.
        
        # Implementing Greedy Coloring on bitsets is feasible.
        # Let's implement a simple greedy coloring bound.
        
        def greedy_color_bound(cand):
            # Returns an upper bound on the size of MIS in cand
            # We greedily color the graph induced by cand.
            # Colors are assigned 1, 2, ...
            # The number of colors used is an upper bound.
            uncolored = cand
            color = 1
            while uncolored != 0:
                # Pick a node from uncolored
                # To make it fast, just pick the first one
                v_bit = uncolored & (-uncolored)
                v_idx = v_bit.bit_length() - 1
                
                # Remove v from uncolored
                uncolored ^= v_bit
                
                # Remove neighbors of v from uncolored (they can share color v)
                # Wait, in MIS, we want independent set.
                # The bound is on the Clique Size of the Conflict Graph? No.
                # MIS in Conflict Graph = Clique in Compatibility Graph.
                # The bound is on the size of the Independent Set.
                # A standard bound for MIS is the number of colors in a greedy coloring of the *Conflict Graph*?
                # No. If we color the Conflict Graph, adjacent nodes (conflicts) must have different colors.
                # This doesn't directly bound the Independent Set size.
                # The Independent Set in G is a Clique in G_complement.
                # The size of the Max Clique in G_complement is bounded by the number of colors in a greedy coloring of G_complement?
                # Yes, Max Clique <= Chromatic Number.
                # So we should color the *Compatibility Graph* (edges where lcm <= 512).
                # But we built the Conflict Graph.
                # Compatibility neighbors of v = candidates & ~neighbors[v] & ~v_bit.
                
                # Let's just use the simple bound: size <= popcount(candidates).
                # With N=512, and likely high compatibility, the simple bound might be too weak.
                # Let's try a slightly better bound: 
                # If we pick v, we can include v and then solve on (candidates \ {v} \ N(v)).
                # Or exclude v and solve on (candidates \ {v}).
                
                # Let's stick to the simple recursive branch and bound first.
                # If it's too slow, we add coloring.
                # Given N=512, and the structure, it should be fine.
                pass
                
            return 0 # Placeholder, we won't use this complex bound immediately
            
        # Let's refine the branching.
        # Pick a vertex v from candidates.
        # Heuristic: Pick v such that it has the most neighbors in candidates?
        # Let's pick the lowest set bit for simplicity first.
        v_bit = candidates & (-candidates)
        v_idx = v_bit.bit_length() - 1
        
        # Branch 1: Include v
        # New candidates are those in candidates that are not v and not neighbors of v
        new_cands_include = candidates & ~neighbors[v_idx] & ~v_bit
        solve(new_cands_include, current_set | v_bit, current_size + 1)
        
        # Branch 2: Exclude v
        new_cands_exclude = candidates & ~v_bit
        solve(new_cands_exclude, current_set, current_size)

    solve(all_nodes, 0, 0)
    
    # Extract the set from best_set bitset
    result = []
    for i in range(1, N + 1):
        if best_set & (1 << i):
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END
