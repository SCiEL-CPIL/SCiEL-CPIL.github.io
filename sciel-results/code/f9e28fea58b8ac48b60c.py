"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict, Counter

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    n = 14
    m = 35
    maximum = 10000
    
    all_pairs = list(combinations(range(n), 2))
    
    def _raw_value(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            sums.add(values[i] + values[j])
            prods.add(values[i] * values[j])
        return max(len(sums), len(prods))

    def _greedy_construct(values, time_limit):
        """
        Greedily construct edges to minimize new distinct sums/products.
        Uses frequency-aware scoring.
        """
        # Precompute sums and products for all pairs
        pair_props = {}
        sum_freq = Counter()
        prod_freq = Counter()
        for i, j in all_pairs:
            s = values[i] + values[j]
            p = values[i] * values[j]
            pair_props[(i, j)] = (s, p)
            sum_freq[s] += 1
            prod_freq[p] += 1
        
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        edges = []
        edge_set = set()
        
        # Sort candidates by frequency (prefer edges with common sums/products)
        candidates = sorted(all_pairs, 
                           key=lambda ij: min(sum_freq[pair_props[ij][0]], 
                                              prod_freq[pair_props[ij][1]]),
                           reverse=True)
        
        for _ in range(m):
            if time.time() > time_limit:
                break
            best_edge = None
            best_cost = float('inf')
            
            for i, j in candidates:
                if (i, j) in edge_set or (j, i) in edge_set:
                    continue
                s, p = pair_props[(i, j)]
                # Cost: number of new distinct sums + new distinct products
                cost = 0
                if s not in sum_counts:
                    cost += 1
                if p not in prod_counts:
                    cost += 1
                
                if cost < best_cost:
                    best_cost = cost
                    best_edge = (i, j)
                    if cost == 0:
                        break  # Can't do better
            
            if best_edge is None:
                break
            
            i, j = best_edge
            s, p = pair_props[(i, j)]
            edges.append([i, j])
            edge_set.add((i, j))
            sum_counts[s] += 1
            prod_counts[p] += 1
            
        return edges, max(len(sum_counts), len(prod_counts))

    def _local_search(values, initial_edges, time_limit):
        """
        Local search with simulated annealing to escape local optima.
        """
        best_edges = [list(e) for e in initial_edges]
        edge_set = set(map(tuple, best_edges))
        
        sum_counts = defaultdict(int)
        prod_counts = defaultdict(int)
        
        def add_edge(edge):
            i, j = edge
            s = values[i] + values[j]
            p = values[i] * values[j]
            sum_counts[s] += 1
            prod_counts[p] += 1
            edge_set.add(tuple(edge))
            best_edges.append(edge)
            
        def remove_edge(edge):
            i, j = edge
            s = values[i] + values[j]
            p = values[i] * values[j]
            sum_counts[s] -= 1
            prod_counts[p] -= 1
            if sum_counts[s] == 0:
                del sum_counts[s]
            if prod_counts[p] == 0:
                del prod_counts[p]
            edge_set.remove(tuple(edge))
            best_edges.remove(edge)

        # Initialize counts
        for e in best_edges:
            i, j = e
            s = values[i] + values[j]
            p = values[i] * values[j]
            sum_counts[s] += 1
            prod_counts[p] += 1

        def get_score():
            return max(len(sum_counts), len(prod_counts))
        
        current_score = get_score()
        best_score = current_score
        best_edges_copy = [list(e) for e in best_edges]
        
        # Simulated annealing parameters
        temperature = 2.0
        cooling_rate = 0.995
        min_temperature = 0.01
        
        no_improve_count = 0
        max_no_improve = 2000
        
        while time.time() < time_limit and no_improve_count < max_no_improve:
            # Pick random edge to remove
            idx = random.randint(0, len(best_edges) - 1)
            e_remove = best_edges[idx]
            
            # Pick random non-edge to add
            attempts = 0
            while True:
                i, j = random.choice(all_pairs)
                if (i, j) not in edge_set and (j, i) not in edge_set:
                    break
                attempts += 1
                if attempts > 100:
                    break
            
            if attempts > 100:
                continue
                
            e_add = [i, j]
            
            # Tentative swap
            remove_edge(e_remove)
            add_edge(e_add)
            
            new_score = get_score()
            
            # Accept if better or with probability based on temperature
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_edges_copy = [list(e) for e in best_edges]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Simulated annealing: accept worse solutions with probability
                delta = new_score - current_score
                if delta <= 0 or random.random() < pow(2.718281828, -delta / max(temperature, 0.001)):
                    current_score = new_score
                    no_improve_count += 1
                else:
                    # Revert
                    remove_edge(e_add)
                    add_edge(e_remove)
                    no_improve_count += 1
            
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
        
        return best_edges_copy, best_score

    def _generate_value_sets():
        """Generate diverse candidate value sets prone to collisions."""
        candidates = []
        random.seed(42)
        
        # 1. Arithmetic progressions
        for start in range(1, 15):
            for step in range(1, 8):
                vals = [start + i * step for i in range(n)]
                if max(vals) <= maximum and len(set(vals)) == n:
                    candidates.append(vals)
        
        # 2. Smooth numbers (many factors -> product collisions)
        def is_smooth(num, max_prime):
            for p in [2, 3, 5, 7, 11, 13]:
                if p > max_prime:
                    break
                while num % p == 0:
                    num //= p
            return num == 1
        
        for max_p in [5, 7, 11]:
            smooth = [i for i in range(1, 500) if is_smooth(i, max_p)]
            if len(smooth) >= n:
                for offset in range(0, len(smooth) - n + 1, 3):
                    candidates.append(sorted(smooth[offset:offset+n]))
        
        # 3. Small random sets with structure
        for _ in range(30):
            # Bias towards small numbers for sum collisions
            vals = random.sample(range(1, 60), n)
            candidates.append(sorted(vals))
        
        # 4. Mixed: some small, some medium
        for _ in range(20):
            small = random.sample(range(1, 30), 7)
            medium = random.sample(range(30, 150), 7)
            vals = small + medium
            if len(set(vals)) == n:
                candidates.append(sorted(vals))
        
        # 5. Geometric-like sequences
        for base in range(2, 6):
            for ratio in [1.5, 2, 2.5, 3]:
                vals = []
                v = base
                while len(vals) < n and v <= maximum:
                    if v not in vals:
                        vals.append(int(v))
                    v *= ratio
                if len(vals) == n:
                    candidates.append(sorted(vals))
        
        # 6. Pairs that sum to same values
        for _ in range(20):
            # Generate values in pairs that might create sum collisions
            vals = []
            for _ in range(n // 2):
                a = random.randint(1, 50)
                b = random.randint(1, 50)
                vals.extend([a, b])
            # Add remaining
            while len(vals) < n:
                v = random.randint(1, 50)
                if v not in vals:
                    vals.append(v)
            if len(set(vals)) == n:
                candidates.append(sorted(vals))
        
        # Remove duplicates and validate
        unique_candidates = []
        seen = set()
        for v in candidates:
            if len(set(v)) != n:
                continue
            if any(x <= 0 or x > maximum for x in v):
                continue
            key = tuple(sorted(v))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(sorted(v))
        
        return unique_candidates

    best_result = None
    best_overall_score = float('inf')
    
    candidate_values = _generate_value_sets()
    
    # Sort candidates by a quick heuristic: prefer those with more sum/product collisions
    def candidate_quality(vals):
        sums = Counter()
        prods = Counter()
        for i in range(n):
            for j in range(i+1, n):
                sums[vals[i] + vals[j]] += 1
                prods[vals[i] * vals[j]] += 1
        # Higher max frequency = better collision potential
        return max(max(sums.values()), max(prods.values()))
    
    candidate_values.sort(key=candidate_quality, reverse=True)
    
    # Limit to top candidates to save time
    max_candidates = min(len(candidate_values), 40)
    candidate_values = candidate_values[:max_candidates]
    
    time_per_candidate = (deadline - time.time()) / max(1, len(candidate_values))
    
    for vals in candidate_values:
        if len(set(vals)) != n or any(v <= 0 or v > maximum for v in vals):
            continue
            
        if time.time() > deadline - 8.0:
            break
            
        # 1. Greedy Start
        g_edges, g_score = _greedy_construct(vals, time.time() + 1.0)
        
        # 2. Local Search from Greedy
        ls_time = min(time_per_candidate * 0.5, 5.0)
        ls_edges, ls_score = _local_search(vals, g_edges, time.time() + ls_time)
        
        if ls_score < best_overall_score:
            best_overall_score = ls_score
            best_result = {'values': vals, 'edges': ls_edges}
        
        # 3. Local Search from Random (if time allows)
        if time.time() < deadline - 5.0 and len(candidate_values) <= 20:
            r_edges = [list(e) for e in random.sample(all_pairs, m)]
            ls2_time = min(time_per_candidate * 0.3, 3.0)
            ls2_edges, ls2_score = _local_search(vals, r_edges, time.time() + ls2_time)
            
            if ls2_score < best_overall_score:
                best_overall_score = ls2_score
                best_result = {'values': vals, 'edges': ls2_edges}
                
    if best_result is None:
        vals = list(range(1, n+1))
        edges = [list(e) for e in all_pairs[:m]]
        best_result = {'values': vals, 'edges': edges}
        
    result = {
        'values': [int(v) for v in best_result['values']],
        'edges': [[int(i), int(j)] for i, j in best_result['edges']]
    }
    
    return result

# EVOLVE-BLOCK-END
