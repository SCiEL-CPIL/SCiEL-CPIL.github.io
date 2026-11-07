"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations
from collections import defaultdict
import time

def construct():
    """
    Construct 14 distinct positive integers <= 10000 and exactly 35 edges
    to minimize max(number of distinct edge sums, number of distinct edge products).
    
    Strategy:
    1. Generate candidate value sets with high symmetry (many repeated sums/products).
    2. For each set, perform intensive local search to find a 35-edge subgraph 
       that minimizes the max of distinct sums and products.
    """
    
    n = 14
    m = 35
    max_val = 10000
    all_pairs = list(combinations(range(n), 2))
    all_pairs_set = set(all_pairs)
    
    def get_score_info(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods)), sums, prods

    def local_search(values, edges, max_iter=10000, time_limit=10.0):
        """
        Improve solution via local search.
        Moves: 
        1. Swap one edge with another.
        2. Swap two edges with two other edges.
        """
        current_score, current_sums, current_prods = get_score_info(values, edges)
        best_score = current_score
        best_edges = edges[:]
        
        # Precompute sum and prod for all pairs for this value set
        pair_sum = {}
        pair_prod = {}
        for i, j in all_pairs:
            pair_sum[(i, j)] = values[i] + values[j]
            pair_prod[(i, j)] = values[i] * values[j]
        
        start_time = time.time()
        no_improve_count = 0
        max_no_improve = 500
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            if no_improve_count > max_no_improve:
                break

            # Generate candidate moves
            # 1. Single edge swap
            # Pick a random edge in current set, pick a random edge NOT in current set
            if random.random() < 0.7:
                idx = random.randint(0, len(edges) - 1)
                old_edge = edges[idx]
                # Find a candidate to swap in
                # To be efficient, we can just pick random from all_pairs not in edges
                # But we want to be smart: prefer edges with sums/prods already in use?
                # Actually, just random is often fine for diversity, but let's bias towards "good" edges.
                
                # Heuristic: Pick an edge that shares a sum or product with existing edges
                # This increases the chance of reducing distinct counts.
                candidates = []
                for _ in range(50): # Sample 50 random non-edges
                    cand = random.choice(all_pairs)
                    if cand not in set(map(tuple, edges)):
                        candidates.append(cand)
                
                if not candidates:
                    continue
                    
                # Choose candidate that minimizes new distinct items
                best_cand = None
                best_delta = float('inf')
                
                for cand in candidates:
                    s = pair_sum[cand]
                    p = pair_prod[cand]
                    
                    # If we remove old_edge, do we lose its sum/prod?
                    # We need to check if any other edge has that sum/prod.
                    # This is expensive. Let's just estimate.
                    # If cand's sum is in current_sums OR prod is in current_prods, it's likely good.
                    
                    # Simple heuristic:
                    # Score = (1 if s not in current_sums else 0) + (1 if p not in current_prods else 0)
                    # Lower is better.
                    score = (1 if s not in current_sums else 0) + (1 if p not in current_prods else 0)
                    if score < best_delta:
                        best_delta = score
                        best_cand = cand
                
                if best_cand is None:
                    continue

                new_edges = edges[:]
                new_edges[idx] = best_cand
                
                new_score, new_sums, new_prods = get_score_info(values, new_edges)
                
                if new_score < current_score:
                    edges = new_edges
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods
                    no_improve_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = edges[:]
                elif new_score == current_score and random.random() < 0.1:
                    edges = new_edges
                    current_sums = new_sums
                    current_prods = new_prods
                    no_improve_count += 1
                else:
                    no_improve_count += 1

            else:
                # 2. Double edge swap
                if len(edges) < 2:
                    continue
                idx1, idx2 = random.sample(range(len(edges)), 2)
                old_e1 = edges[idx1]
                old_e2 = edges[idx2]
                
                candidates = []
                for _ in range(50):
                    cand = random.choice(all_pairs)
                    if cand not in set(map(tuple, edges)) and cand != old_e1 and cand != old_e2:
                        candidates.append(cand)
                
                if len(candidates) < 2:
                    continue
                    
                # Pick 2 candidates
                # To keep it simple, just pick 2 random from candidates
                cand1, cand2 = random.sample(candidates, 2)
                
                new_edges = edges[:]
                new_edges[idx1] = cand1
                new_edges[idx2] = cand2
                
                new_score, new_sums, new_prods = get_score_info(values, new_edges)
                
                if new_score < current_score:
                    edges = new_edges
                    current_score = new_score
                    current_sums = new_sums
                    current_prods = new_prods
                    no_improve_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_edges = edges[:]
                elif new_score == current_score and random.random() < 0.1:
                    edges = new_edges
                    current_sums = new_sums
                    current_prods = new_prods
                    no_improve_count += 1
                else:
                    no_improve_count += 1

        return best_score, best_edges

    best_global_score = float('inf')
    best_global_values = None
    best_global_edges = None
    
    # Define value sets
    # 1. Small integers (1-14)
    # 2. Arithmetic progression (2, 4, ..., 28)
    # 3. Geometric progression (1, 2, 4, ..., 8192)
    # 4. Mixed small
    # 5. Random small
    value_sets = []
    
    # Set 1: 1..14
    value_sets.append(list(range(1, 15)))
    
    # Set 2: 2..28 even
    value_sets.append([2*i for i in range(1, 15)])
    
    # Set 3: 1, 2, 4, 8...
    value_sets.append([2**i for i in range(14)])
    
    # Set 4: Small random
    for _ in range(10):
        vals = sorted(random.sample(range(1, 50), 14))
        value_sets.append(vals)
        
    # Set 5: Structured small
    # e.g. 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610 (Fibs)
    value_sets.append([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
    
    # Set 6: Powers of 3? 1, 3, 9, 27...
    value_sets.append([3**i for i in range(14)])
    
    # Set 7: Primes
    value_sets.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43])
    
    # Set 8: Squares
    value_sets.append([i*i for i in range(1, 15)])
    
    # Set 9: 10..23
    value_sets.append(list(range(10, 24)))
    
    # Set 10: 1..13, 20
    value_sets.append(list(range(1, 14)) + [20])
    
    # Shuffle value sets
    random.shuffle(value_sets)
    
    time_start = time.time()
    time_budget = 55.0 # Leave 5s for overhead
    
    for values in value_sets:
        if time.time() - time_start > time_budget:
            break
            
        # Ensure distinct and valid
        if len(set(values)) != n or min(values) < 1 or max(values) > max_val:
            continue
            
        # Initial edges: Random 35 edges
        edges = [list(e) for e in random.sample(all_pairs, m)]
        
        # Run local search
        score, improved_edges = local_search(values, edges, max_iter=2000, time_limit=2.0)
        
        if score < best_global_score:
            best_global_score = score
            best_global_values = values[:]
            best_global_edges = improved_edges[:]
            
            # If we hit a very low score, we can stop early
            if best_global_score <= 15:
                break

    # If we didn't find anything good, do a final intense search on the best found
    if best_global_values:
        # Run a longer search on the best candidate
        final_score, final_edges = local_search(best_global_values, best_global_edges, max_iter=5000, time_limit=10.0)
        if final_score < best_global_score:
            best_global_score = final_score
            best_global_edges = final_edges

    # Ensure types are correct
    best_global_values = [int(v) for v in best_global_values]
    best_global_edges = [[int(i), int(j)] for i, j in best_global_edges]
    
    # Validation
    if not best_global_values:
        # Fallback
        best_global_values = list(range(1, 15))
        best_global_edges = [list(e) for e in random.sample(all_pairs, m)]

    return {'values': best_global_values, 'edges': best_global_edges}

# EVOLVE-BLOCK-END
