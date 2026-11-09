"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _raw_score(values, edges):
    """Compute the raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        su = values[u]
        sv = values[v]
        sums.add(su + sv)
        prods.add(su * sv)
    return max(len(sums), len(prods))

def _greedy_edge_selection(values, n, m):
    """
    Greedily select m edges to minimize the score.
    Strategy: Start with empty graph. At each step, add the edge that causes
    the smallest increase in max(distinct_sums, distinct_prods).
    Ties are broken randomly to allow exploration.
    """
    all_edges = list(combinations(range(n), 2))
    random.shuffle(all_edges)
    
    current_sums = set()
    current_prods = set()
    selected_edges = []
    
    for _ in range(m):
        best_edge = None
        best_score = float('inf')
        best_sums = None
        best_prods = None
        
        # We can't just pick the first one because we need to see the impact.
        # However, checking all remaining edges for every addition is O(m^2 * n^2).
        # With n=14, m=35, total edges 91. 
        # 35 * 91 * (cost of set update) is fast enough.
        
        candidates = [e for e in all_edges if e not in set(map(tuple, selected_edges))]
        
        # To speed up, we can sample candidates if the list is large, 
        # but 91 is small. Let's check all.
        
        for e in candidates:
            u, v = e
            new_sum = values[u] + values[v]
            new_prod = values[u] * values[v]
            
            # Estimate score without actually updating sets permanently
            # We need to know if this adds to the sets.
            # Actually, we can just simulate.
            temp_sums = current_sums | {new_sum}
            temp_prods = current_prods | {new_prod}
            
            score = max(len(temp_sums), len(temp_prods))
            
            if score < best_score:
                best_score = score
                best_edge = e
                best_sums = temp_sums
                best_prods = temp_prods
            elif score == best_score and random.random() < 0.1:
                # Random tie-breaking to escape local minima
                best_score = score
                best_edge = e
                best_sums = temp_sums
                best_prods = temp_prods
        
        if best_edge is None:
            # Should not happen if m <= total edges
            break
            
        selected_edges.append(best_edge)
        current_sums = best_sums
        current_prods = best_prods
        
    return [list(e) for e in selected_edges]

def _generate_value_candidates():
    """
    Generate sets of 14 distinct positive integers <= 10000.
    Strategy: Mix small numbers (for sum collisions) and highly composite numbers (for product collisions).
    """
    candidates = []
    
    # Set 1: Small integers with high additive structure (1..14)
    candidates.append(list(range(1, 15)))
    
    # Set 2: Powers of 2 (guarantees unique sums/prods, baseline 35)
    candidates.append([2**i for i in range(14)])
    
    # Set 3: Highly composite numbers (for product collisions)
    # 1, 2, 4, 6, 8, 12, 16, 24, 30, 36, 48, 60, 72, 84
    candidates.append([1, 2, 4, 6, 8, 12, 16, 24, 30, 36, 48, 60, 72, 84])
    
    # Set 4: Random mix of small numbers
    random.seed(42)
    base = list(range(1, 100))
    random.shuffle(base)
    candidates.append(base[:14])
    
    # Set 5: Numbers with many factors (highly composite-ish)
    # 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20
    candidates.append([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20])
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_score = float('inf')
    best_result = None
    
    value_sets = _generate_value_candidates()
    
    for values in value_sets:
        # Ensure types are correct
        values = [int(v) for v in values]
        
        # Check constraints
        if len(set(values)) != 14:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
            
        # Try multiple random restarts of greedy edge selection
        for restart in range(20):
            if time.time() > deadline:
                break
                
            edges = _greedy_edge_selection(values, 14, 35)
            
            # Validate edges
            if len(edges) != 35:
                continue
                
            edge_tuples = set(tuple(sorted(e)) for e in edges)
            if len(edge_tuples) != 35:
                continue
                
            # Check for loops
            if any(u == v for u, v in edges):
                continue
                
            score = _raw_score(values, edges)
            
            if score < best_score:
                best_score = score
                best_result = {
                    'values': values,
                    'edges': edges
                }
                
                # Early exit if we hit a theoretical lower bound?
                # Max distinct sums for 14 nodes is at most 14*13/2 = 91.
                # But we only have 35 edges.
                # If score drops below 20, it's very good.
                if best_score <= 15:
                    break
                    
        if time.time() > deadline:
            break
            
    if best_result is None:
        # Fallback to seed
        return {
            'values': list(range(1, 15)),
            'edges': [list(e) for e in list(combinations(range(14), 2))[:35]]
        }
        
    return best_result
# EVOLVE-BLOCK-END
