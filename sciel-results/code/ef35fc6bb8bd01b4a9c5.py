"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import defaultdict, Counter

_DEADLINE = time.time() + 85  # safety margin before 90s kill

def _score(values, edges):
    """Compute raw score: max(distinct sums, distinct products)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _score_detailed(values, edges):
    """Return (sums_set, prods_set, score)."""
    sums = set()
    prods = set()
    for u, v in edges:
        a = values[u]
        b = values[v]
        sums.add(a + b)
        prods.add(a * b)
    return sums, prods, max(len(sums), len(prods))

def _init_solution(values, n, m):
    """Initialize edges using greedy: prefer edges that share sums/products."""
    all_edges = list(combinations(range(n), 2))
    
    # Precompute (sum, prod) for each edge
    edge_sp = {}
    for e in all_edges:
        u, v = e
        edge_sp[e] = (values[u] + values[v], values[u] * values[v])
    
    # Group edges by sum and product
    sum_groups = defaultdict(list)
    prod_groups = defaultdict(list)
    for e in all_edges:
        s, p = edge_sp[e]
        sum_groups[s].append(e)
        prod_groups[p].append(e)
    
    # Score each edge by how many others share its sum or product
    edge_weight = {}
    for e in all_edges:
        s, p = edge_sp[e]
        edge_weight[e] = len(sum_groups[s]) + len(prod_groups[p])
    
    # Sort by weight descending
    sorted_edges = sorted(all_edges, key=lambda e: -edge_weight[e])
    initial_edges = set(sorted_edges[:m])
    
    return initial_edges, edge_sp, sum_groups, prod_groups

def _anneal(values, n, m, edge_sp, time_budget=20):
    """Simulated annealing on edge selection for fixed values."""
    start_time = time.time()
    
    # Initial solution
    current_edges, _, sum_groups, prod_groups = _init_solution(values, n, m)
    all_edges = list(combinations(range(n), 2))
    
    # Build lookup: for each edge, what's its sum and prod
    # And for each sum/prod value, which edges have it
    
    current_sums_set, current_prods_set, current_score = _score_detailed(values, list(current_edges))
    best_score = current_score
    best_edges = set(current_edges)
    
    # Track counts for fast delta computation
    sum_counts = Counter()
    prod_counts = Counter()
    for e in current_edges:
        s, p = edge_sp[e]
        sum_counts[s] += 1
        prod_counts[p] += 1
    
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    max_iter = 200000
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        if time.time() > _DEADLINE:
            break
        if current_score <= 3:
            break
        
        # Pick a random edge in current set to remove
        edges_list = list(current_edges)
        if not edges_list:
            break
        remove_edge = edges_list[random.randint(0, len(edges_list) - 1)]
        
        # Try to add a random edge not in set
        candidates = [e for e in all_edges if e not in current_edges]
        if not candidates:
            break
        add_edge = candidates[random.randint(0, len(candidates) - 1)]
        
        rs, rp = edge_sp[remove_edge]
        as_, ap = edge_sp[add_edge]
        
        # Compute delta score
        # Removing remove_edge:
        #   sum rs count decreases by 1; if it becomes 0, distinct sums decrease by 1
        #   prod rp count decreases by 1; if it becomes 0, distinct prods decrease by 1
        # Adding add_edge:
        #   sum as count increases by 1; if it was 0, distinct sums increase by 1
        #   prod ap count increases by 1; if it was 0, distinct prods increase by 1
        
        delta_sums = 0
        if sum_counts[rs] == 1:
            delta_sums -= 1  # losing a distinct sum
        if sum_counts[as_] == 0:
            delta_sums += 1  # gaining a distinct sum
        
        delta_prods = 0
        if prod_counts[rp] == 1:
            delta_prods -= 1
        if prod_counts[ap] == 0:
            delta_prods += 1
        
        # New score components
        new_num_sums = len(current_sums_set) + delta_sums
        new_num_prods = len(current_prods_set) + delta_prods
        new_score = max(new_num_sums, new_num_prods)
        
        delta_score = new_score - current_score
        
        # Accept or reject
        if delta_score < 0 or (T > T_min and random.random() < math.exp(-delta_score / T)):
            # Apply move
            current_edges.discard(remove_edge)
            current_edges.add(add_edge)
            
            # Update counts
            sum_counts[rs] -= 1
            if sum_counts[rs] == 0:
                del sum_counts[rs]
            prod_counts[rp] -= 1
            if prod_counts[rp] == 0:
                del prod_counts[rp]
            
            sum_counts[as_] += 1
            prod_counts[ap] += 1
            
            # Update sets
            if sum_counts[rs] == 0:
                current_sums_set.discard(rs)
            if sum_counts[as_] == 1:
                current_sums_set.add(as_)
            
            if prod_counts[rp] == 0:
                current_prods_set.discard(rp)
            if prod_counts[ap] == 1:
                current_prods_set.add(ap)
            
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
        
        # Cool down
        T = max(T_min, T * cooling)
    
    return best_score, best_edges

def _generate_value_candidates(n, max_val=10000):
    """Generate diverse value candidate sets."""
    candidates = []
    
    # Small consecutive
    candidates.append(list(range(1, n + 1)))
    candidates.append(list(range(2, n + 2)))
    
    # Powers of 2 (strong multiplicative structure)
    candidates.append([2**i for i in range(n)])
    
    # Highly composite / smooth numbers
    smooth = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100]
    for start in range(0, min(20, len(smooth) - n + 1)):
        vals = smooth[start:start+n]
        if len(set(vals)) == n and all(1 <= v <= max_val for v in vals):
            candidates.append(vals)
    
    # Mix: small base with multiplicative structure
    candidates.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    candidates.append([1, 2, 4, 8, 16, 3, 6, 12, 24, 5, 10, 20, 40, 80])
    
    # Random with bias toward small numbers
    random.seed(42)
    for trial in range(50):
        # Mix of small random and structured
        if trial % 3 == 0:
            vals = sorted(random.sample(range(1, 30), n))
        elif trial % 3 == 1:
            # Geometric-ish
            base = random.uniform(1.1, 1.5)
            vals = [max(1, int(base**i)) for i in range(n)]
            # Ensure distinct
            seen = set()
            unique_vals = []
            for v in vals:
                while v in seen:
                    v += 1
                seen.add(v)
                unique_vals.append(v)
            vals = sorted(unique_vals)
            if len(vals) != n or any(v > max_val for v in vals):
                continue
        else:
            # Random small
            vals = sorted(random.sample(range(1, 50), n))
        
        if len(set(vals)) == n and all(1 <= v <= max_val for v in vals):
            candidates.append(vals)
    
    # Deduplicate
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    
    return unique_candidates

def construct():
    n = 14
    m = 35
    max_val = 10000
    
    best_score = m
    best_result = None
    
    candidates = _generate_value_candidates(n, max_val)
    
    # Sort candidates by heuristic: prefer sets with more additive/multiplicative structure
    def heuristic(vals):
        # Count how many pairs share sums
        pair_sums = Counter()
        for i in range(n):
            for j in range(i+1, n):
                pair_sums[vals[i] + vals[j]] += 1
        sum_collisions = sum(c*(c-1)//2 for c in pair_sums.values())
        
        # Count how many pairs share products
        pair_prods = Counter()
        for i in range(n):
            for j in range(i+1, n):
                pair_prods[vals[i] * vals[j]] += 1
        prod_collisions = sum(c*(c-1)//2 for c in pair_prods.values())
        
        # Prefer more collisions (more structure)
        return -(sum_collisions + prod_collisions)
    
    candidates.sort(key=heuristic)
    
    # Allocate time budget
    elapsed = time.time() - (_DEADLINE - 85)
    remaining = max(5, 85 - elapsed)
    
    # Try top candidates with more time
    for idx, vals in enumerate(candidates):
        if time.time() > _DEADLINE - 5:
            break
        
        # Allocate more time to better candidates
        if idx < 5:
            time_budget = min(15, remaining / (len(candidates) - idx))
        elif idx < 15:
            time_budget = min(8, remaining / (len(candidates) - idx))
        else:
            time_budget = min(3, remaining / (len(candidates) - idx))
        
        if time_budget < 1:
            time_budget = 1
        
        # Precompute edge info
        all_edges = list(combinations(range(n), 2))
        edge_sp = {}
        for e in all_edges:
            u, v = e
            edge_sp[e] = (vals[u] + vals[v], vals[u] * vals[v])
        
        # Run annealing
        score, edges = _anneal(vals, n, m, edge_sp, time_budget=time_budget)
        
        if score < best_score:
            best_score = score
            best_result = (vals, [list(e) for e in edges])
            if best_score <= 3:
                break
    
    if best_result is None:
        values = list(range(1, n + 1))
        edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    else:
        values, edges = best_result
    
    values = [int(v) for v in values]
    edges = [[int(u), int(v)] for u, v in edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
