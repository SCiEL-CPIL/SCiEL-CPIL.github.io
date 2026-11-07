"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from itertools import combinations

def construct():
    """
    Construct 14 distinct positive integers <= 10000 and exactly 35 edges
    to minimize max(number of distinct edge sums, number of distinct edge products).
    
    Strategy:
    1. Use structured initial values (arithmetic progression) to encourage sum collisions.
    2. Initialize edges by selecting those with the smallest sums to minimize distinct sums.
    3. Perform local search to minimize the maximum of distinct sums and products.
    """
    
    n = 14
    m = 35
    max_val = 10000
    
    def score(values, edges):
        sums = set()
        prods = set()
        for i, j in edges:
            s = values[i] + values[j]
            p = values[i] * values[j]
            sums.add(s)
            prods.add(p)
        return max(len(sums), len(prods))

    def get_initial_solution(start=1, step=1):
        # Values: arithmetic progression
        vals = [start + i * step for i in range(n)]
        # Ensure distinct and <= 10000
        if max(vals) > max_val:
            # Scale down or adjust
            scale = max_val / max(vals)
            vals = [int(v * scale) for v in vals]
            # Ensure distinct
            seen = set()
            new_vals = []
            for v in vals:
                if v not in seen and v > 0:
                    new_vals.append(v)
                    seen.add(v)
            if len(new_vals) < n:
                # Fallback to random
                return random.sample(range(1, max_val), n), []
            vals = new_vals
        
        # Calculate all edges with their sums
        all_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                s = vals[i] + vals[j]
                all_edges.append((s, i, j))
        
        # Sort by sum to pick edges with smallest sums first
        all_edges.sort(key=lambda x: x[0])
        
        # Pick first 35 edges
        edges = []
        for _, i, j in all_edges[:m]:
            edges.append([i, j])
            
        return vals, edges

    def local_search(values, edges, max_iters=1000):
        current_score = score(values, edges)
        best_score = current_score
        best_values = values[:]
        best_edges = edges[:]
        
        all_possible_edges = list(combinations(range(n), 2))
        
        for iteration in range(max_iters):
            # Choose a random move
            # 1. Swap an edge
            if edges:
                rem_idx = random.randint(0, len(edges) - 1)
                rem_e = edges[rem_idx]
                rem_e_tuple = tuple(sorted(rem_e))
                
                # Find candidate edges to add
                # Prefer edges that share sum or product with current edges
                current_sums = set(values[i] + values[j] for i, j in edges)
                current_prods = set(values[i] * values[j] for i, j in edges)
                
                candidates = []
                for e in all_possible_edges:
                    if e == rem_e_tuple:
                        continue
                    if any(tuple(sorted(e)) == tuple(sorted(existing)) for existing in edges):
                        continue
                    s = values[e[0]] + values[e[1]]
                    p = values[e[0]] * values[e[1]]
                    # Heuristic: prefer if sum or prod is already used
                    weight = 0
                    if s in current_sums: weight += 1
                    if p in current_prods: weight += 1
                    candidates.append((weight, e))
                
                if not candidates:
                    continue
                
                # Pick best candidate or random from top
                candidates.sort(key=lambda x: -x[0])
                # 50% chance pick top, 50% random
                if random.random() < 0.5:
                    _, new_e = candidates[0]
                else:
                    _, new_e = random.choice(candidates[:10])
                
                new_edges = edges[:]
                new_edges[rem_idx] = list(new_e)
                
                new_score = score(values, new_edges)
                
                if new_score < current_score:
                    values = values[:] # No change to values
                    edges = new_edges
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_values = values[:]
                        best_edges = edges[:]
                        
            # 2. Swap values
            elif random.random() < 0.2:
                i, j = random.sample(range(n), 2)
                new_values = values[:]
                new_values[i], new_values[j] = new_values[j], new_values[i]
                
                # Check if values are still distinct and positive
                if len(set(new_values)) != n or min(new_values) <= 0 or max(new_values) > max_val:
                    continue
                    
                new_score = score(new_values, edges)
                
                if new_score < current_score:
                    values = new_values
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_values = values[:]
                        best_edges = edges[:]
                        
            if best_score <= 12: # Optimistic lower bound
                break
                
        return best_score, best_values, best_edges

    best_overall_score = float('inf')
    best_overall_values = None
    best_overall_edges = None
    
    # Try different starting configurations
    configs = [
        (1, 1),    # 1..14
        (1, 2),    # 1,3,5..27
        (2, 1),    # 2..15
        (5, 1),    # 5..18
        (10, 1),   # 10..23
        (1, 3),    # 1,4,7..
        (1, 5),    # 1,6,11..
    ]
    
    for start, step in configs:
        vals, edges = get_initial_solution(start, step)
        if not edges:
            continue
        s, v, e = local_search(vals, edges, max_iters=200)
        if s < best_overall_score:
            best_overall_score = s
            best_overall_values = v
            best_overall_edges = e
            
        if best_overall_score <= 12:
            break
            
    # If no good result, fallback to random search
    if best_overall_values is None or best_overall_score > 20:
        for _ in range(10):
            vals = random.sample(range(1, 1000), n)
            # Sort to keep structure somewhat
            vals.sort()
            # Create initial edges based on smallest sums
            all_edges = [(vals[i] + vals[j], i, j) for i in range(n) for j in range(i+1, n)]
            all_edges.sort()
            edges = [[i, j] for _, i, j in all_edges[:m]]
            s, v, e = local_search(vals, edges, max_iters=500)
            if s < best_overall_score:
                best_overall_score = s
                best_overall_values = v
                best_overall_edges = e

    return {'values': best_overall_values, 'edges': best_overall_edges}

# EVOLVE-BLOCK-END
