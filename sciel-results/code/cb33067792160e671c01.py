"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    n = 14
    m = 35
    maximum = 10000
    
    # Helper to compute the objective
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            v = values[i]
            w = values[v] # Wait, values[j]
            w = values[j]
            sums.add(v + w)
            prods.add(v * w)
        return max(len(sums), len(prods))
    
    # Strategy:
    # 1. Values: Powers of 2 are ideal for distinct products (unique factorization).
    #    1, 2, 4, ..., 2^13 = 8192. All <= 10000.
    #    This ensures all 35 edge products are distinct (len(prods) = 35).
    #    Sums of distinct powers of 2 are unique for different pairs of indices?
    #    Yes, if we pick distinct pairs (i,j), the sum 2^i + 2^j is unique to that pair.
    #    So len(sums) will also be 35.
    #    Result: max(35, 35) = 35.
    
    # Can we do better than 35?
    # The minimum possible number of distinct sums is related to the number of edges.
    # If we have m edges, can we have fewer than m distinct sums?
    # Yes, if multiple edges have the same sum.
    # Example: values 1, 10, 11. Edge (0,1) sum 11, Edge (0,2) sum 12, Edge (1,2) sum 21.
    # If we have values 1, 2, 3, 4.
    # Edges: (1,3) sum 4, (2,2) no loop.
    # To minimize distinct sums, we want many edges to share sums.
    # To minimize distinct products, we want many edges to share products.
    
    # Let's try a different value set: Small integers.
    # If values are 1..14, products can collide.
    # 1*12 = 12, 2*6 = 12, 3*4 = 12.
    # Sums: 1+13=14, 2+12=14, 3+11=14, 4+10=14, 5+9=14, 6+8=14.
    
    # Let's use a heuristic search with a tight deadline.
    # Initial candidate: Powers of 2 give 35.
    # Let's try to find a set where distinct sums < 35 or distinct products < 35.
    
    best_score = float('inf')
    best_values = None
    best_edges = None
    
    # Generate a few random candidate structures
    # Focus on values that create collisions.
    
    # Candidate 1: Powers of 2 (Baseline 35)
    vals_p2 = [2**i for i in range(n)]
    # Edges: To minimize sums, pick edges with smallest values?
    # But with powers of 2, all sums are distinct anyway.
    # So score is 35 regardless of edge choice (as long as m=35).
    
    # Candidate 2: Small integers 1..14
    vals_small = list(range(1, n+1))
    # We need to pick 35 edges out of 91 possible.
    # To minimize distinct sums, we should pick edges that share sums.
    # To minimize distinct products, we should pick edges that share products.
    
    # Let's do a greedy/randomized search for edges given a value set.
    # And also search for value sets.
    
    def optimize_edges_for_values(values):
        # Generate all possible edges and their sums/prods
        all_edges = list(combinations(range(n), 2))
        edge_metrics = []
        for i, j in all_edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            edge_metrics.append((i, j, s, p))
        
        # We want to select m edges such that the number of distinct sums and distinct products is minimized.
        # This is a set cover-like problem or just a combinatorial optimization.
        # Heuristic: Group edges by sum. Pick groups with many edges.
        # Group edges by product. Pick groups with many edges.
        
        # Let's try a simple random restart hill climbing on the edge set.
        best_local_score = float('inf')
        best_local_edges = None
        
        # Initial edge set: random 35
        current_edges = set(random.sample(all_edges, m))
        current_sums = set(values[i] + values[j] for i, j in current_edges)
        current_prods = set(values[i] * values[j] for i, j in current_edges)
        current_score = max(len(current_sums), len(current_prods))
        
        if current_score < best_local_score:
            best_local_score = current_score
            best_local_edges = set(current_edges)
            
        # Hill climbing: swap one edge for another
        for _ in range(1000):
            if time.time() > deadline - 10:
                break
                
            # Pick a random edge to remove
            e_out = random.choice(list(current_edges))
            # Pick a random edge to add (not in current)
            possible_in = [e for e in all_edges if e not in current_edges]
            if not possible_in:
                break
            e_in = random.choice(possible_in)
            
            # Calculate new score
            new_edges = set(current_edges)
            new_edges.remove(e_out)
            new_edges.add(e_in)
            
            # Efficient update of sets
            # Remove e_out contributions
            i_out, j_out = e_out
            s_out = values[i_out] + values[j_out]
            p_out = values[i_out] * values[j_out]
            
            # Check if s_out is unique in current set
            # This is O(m) to check, which is fine for m=35.
            # Let's just recompute for simplicity in the inner loop if it's fast enough.
            # 35 edges recompute is very fast.
            
            new_sums = set()
            new_prods = set()
            for i, j in new_edges:
                new_sums.add(values[i] + values[j])
                new_prods.add(values[i] * values[j])
                
            new_score = max(len(new_sums), len(new_prods))
            
            if new_score < current_score:
                current_edges = new_edges
                current_sums = new_sums
                current_prods = new_prods
                current_score = new_score
                if current_score < best_local_score:
                    best_local_score = current_score
                    best_local_edges = set(current_edges)
            
        return best_local_edges, best_local_score

    # Try a few value sets
    value_sets = [
        list(range(1, n+1)),
        [2**i for i in range(n)],
        [i*i for i in range(1, n+1)], # Squares
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # Invalid: distinct positive integers required
    ]
    
    # Ensure distinctness for the sets
    valid_value_sets = []
    for vs in value_sets:
        if len(set(vs)) == n and all(v > 0 and v <= maximum for v in vs):
            valid_value_sets.append(vs)
    
    # Add some random value sets
    for _ in range(20):
        if time.time() > deadline - 20:
            break
        random_vals = random.sample(range(1, maximum+1), n)
        valid_value_sets.append(random_vals)
    
    for vs in valid_value_sets:
        if time.time() > deadline - 5:
            break
            
        edges, score = optimize_edges_for_values(vs)
        if score < best_score:
            best_score = score
            best_values = vs
            best_edges = edges
            
    # If we didn't find anything better than 35, return the powers of 2 solution
    if best_edges is None:
        best_values = [2**i for i in range(n)]
        # Pick any 35 edges, e.g., the first 35 in lexicographic order
        best_edges = set(list(combinations(range(n), 2))[:m])
        best_score = _raw_value(best_values, best_edges)

    # Convert to required format
    return {
        'values': list(best_values),
        'edges': [list(e) for e in best_edges]
    }
# EVOLVE-BLOCK-END
