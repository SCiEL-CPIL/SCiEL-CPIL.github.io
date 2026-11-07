"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin for 90s limit
    
    n = 14
    m = 35
    max_val = 10000
    
    # Precompute all possible edges
    all_edges = list(combinations(range(n), 2))
    
    # Helper to evaluate a candidate (values, edges)
    def evaluate(values, edges):
        if not edges:
            return 0
        v_arr = np.array(values, dtype=np.int64)
        e_arr = np.array(edges, dtype=np.int32)
        
        u_vals = v_arr[e_arr[:, 0]]
        v_vals = v_arr[e_arr[:, 1]]
        
        sums = u_vals + v_vals
        prods = u_vals * v_vals
        
        distinct_sums = len(np.unique(sums))
        distinct_prods = len(np.unique(prods))
        
        return max(distinct_sums, distinct_prods)

    # Strategy:
    # 1. Generate a pool of "good" edge structures. 
    #    A good structure is one that allows for many collisions in sums and products.
    #    For example, if we pick edges such that many pairs share the same sum or product potential.
    #    Alternatively, we can just pick random edge sets but evaluate them with a "potential" metric.
    #    However, the key insight is that for a FIXED edge set, we can optimize the values.
    #    So, let's generate diverse edge sets.
    
    def generate_edge_structures(num_structures):
        structures = []
        # Structure 1: Random
        for _ in range(num_structures // 2):
            structures.append(random.sample(all_edges, m))
        
        # Structure 2: Bipartite-like (to encourage sum collisions if values are symmetric?)
        # Let's try to create structures that are dense in certain areas.
        # Or just rely on the value optimization to find the best fit for random structures.
        
        # To ensure diversity, let's also include some structured ones.
        # E.g., edges within first 7 nodes and last 7 nodes?
        # Actually, random sampling with replacement of "motifs" might be better, but simple random is a good start.
        # Let's add a few "clustered" graphs.
        for _ in range(num_structures // 4):
            # Pick a subset of nodes to be densely connected
            core_nodes = random.sample(range(n), 10)
            core_edges = [e for e in all_edges if e[0] in core_nodes and e[1] in core_nodes]
            if len(core_edges) >= m:
                structures.append(random.sample(core_edges, m))
            else:
                structures.append(random.sample(all_edges, m))
                
        # Fill rest with random
        while len(structures) < num_structures:
            structures.append(random.sample(all_edges, m))
            
        return structures

    def optimize_values_for_edges(edges, max_iter=2000):
        """
        Given a fixed set of edges, find values that minimize max(distinct_sums, distinct_prods).
        Uses Simulated Annealing / Hill Climbing.
        """
        # Initialize values with small integers to encourage collisions
        # Small integers have higher collision probability for sums and products in limited ranges.
        values = random.sample(range(1, 100), n)
        current_score = evaluate(values, edges)
        
        best_values = values.copy()
        best_score = current_score
        
        T = 50.0
        T_min = 1.0
        cooling_rate = 0.995
        
        v_arr = np.array(values, dtype=np.int64)
        e_arr = np.array(edges, dtype=np.int32)
        
        for i in range(max_iter):
            if time.time() > deadline - 2.0:
                break
                
            # Pick a random vertex to perturb
            v_idx = random.randint(0, n - 1)
            old_val = values[v_idx]
            
            # Perturbation: small change
            # Try values in a local range or random small value
            if random.random() < 0.5:
                new_val = old_val + random.randint(-10, 10)
            else:
                new_val = random.randint(1, 200)
                
            if new_val <= 0 or new_val > max_val:
                continue
            if new_val == old_val:
                continue
                
            # Check distinctness
            if new_val in values:
                continue
                
            values[v_idx] = new_val
            
            # Evaluate
            new_score = evaluate(values, edges)
            
            # Acceptance probability
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = values.copy()
            else:
                # Simulated Annealing acceptance
                delta = new_score - current_score
                if T > T_min:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        values[v_idx] = old_val
                else:
                    values[v_idx] = old_val
            
            T *= cooling_rate
            
        return best_values, best_score

    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    # Generate a pool of edge structures
    # We want to try many structures because the landscape of "best values for structure X" varies.
    num_structures = 50
    
    structures = generate_edge_structures(num_structures)
    
    # Shuffle structures to mix quality
    random.shuffle(structures)
    
    # Optimize values for each structure
    # We can allocate time per structure based on remaining time.
    # Let's try 20 structures in depth.
    
    for i, edges in enumerate(structures):
        if time.time() > deadline - 5.0:
            break
            
        # Allocate more iterations if we are early
        remaining_structures = len(structures) - i
        time_per_structure = (deadline - 5.0 - time.time()) / remaining_structures
        # Convert time to iterations roughly. 1 iter is ~1ms? Let's assume 5ms/iter.
        max_iter = int(time_per_structure * 1000 / 5)
        if max_iter < 100:
            max_iter = 100
        if max_iter > 5000:
            max_iter = 5000
            
        values, score = optimize_values_for_edges(edges, max_iter=max_iter)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_values = values.copy()
            best_overall_edges = [list(e) for e in edges]
            
            # Early stop if we find a very good solution
            # Theoretical lower bound:
            # 35 edges. 
            # If all sums are same, distinct_sums=1.
            # If all prods are same, distinct_prods=1.
            # Can we get max(1,1)=1? 
            # v_i + v_j = C for all edges.
            # v_i * v_j = P for all edges.
            # This implies v_i, v_j are roots of x^2 - Cx + P = 0.
            # So only 2 distinct values possible for any connected component.
            # With 14 nodes, if graph is connected, all nodes must be one of 2 values.
            # But values must be DISTINCT. So this is impossible.
            # So distinct values >= 2 is not enough, we have 14 distinct values.
            # So sums and prods will vary.
            # A score of ~20-30 is good. ~15 is great.
            if best_overall_score <= 15:
                break

    # Fallback
    if best_overall_edges is None:
        values = random.sample(range(1, 100), n)
        edges = random.sample(all_edges, m)
        best_overall_values = values
        best_overall_edges = [list(e) for e in edges]
        best_overall_score = evaluate(values, edges)

    res = {
        'values': [int(v) for v in best_overall_values],
        'edges': [[int(u), int(v)] for u, v in best_overall_edges]
    }
    
    return res

# EVOLVE-BLOCK-END
