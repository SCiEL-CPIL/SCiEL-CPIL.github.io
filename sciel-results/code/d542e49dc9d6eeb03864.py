import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin
    
    n = 14
    m = 35
    
    all_edges = list(combinations(range(n), 2))
    
    def evaluate(values, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(values[u] + values[v])
            prods.add(values[u] * values[v])
        return max(len(sums), len(prods))
    
    def get_min_sum_graph():
        """
        Construct a graph with 35 edges that minimizes the number of distinct index sums u+v.
        Sums in [9, 15] provide the highest capacity.
        """
        # Group edges by sum
        edges_by_sum = {}
        for u, v in all_edges:
            s = u + v
            if s not in edges_by_sum:
                edges_by_sum[s] = []
            edges_by_sum[s].append((u, v))
        
        # Prioritize sums 9 to 15
        priority_sums = [13, 12, 14, 11, 15, 10, 16, 9, 17, 8, 18, 7, 19, 6, 20, 5, 21, 4, 22, 3, 23, 2, 24, 1, 25]
        
        selected_edges = []
        for s in priority_sums:
            if s in edges_by_sum:
                for e in edges_by_sum[s]:
                    if len(selected_edges) < m:
                        selected_edges.append(e)
                    else:
                        break
            if len(selected_edges) >= m:
                break
        
        # If we still don't have enough (should not happen), fill with random
        if len(selected_edges) < m:
            remaining = [e for e in all_edges if e not in selected_edges]
            random.shuffle(remaining)
            selected_edges.extend(remaining[:m - len(selected_edges)])
        
        return selected_edges[:m]

    def get_random_graph():
        edges = random.sample(all_edges, m)
        return edges

    def simulated_annealing(values, edges, time_limit):
        current_score = evaluate(values, edges)
        best_score = current_score
        best_values = list(values)
        
        T = 1000.0
        cooling = 0.995
        
        while time.time() < time_limit:
            # Generate neighbor
            i = random.randint(0, n - 1)
            old_val = values[i]
            
            # Strategy: mix of local perturbations and jumps
            r = random.random()
            if r < 0.5:
                # Small perturbation
                delta = random.randint(-10, 10)
                new_val = old_val + delta
            elif r < 0.8:
                # Medium jump
                new_val = random.randint(1, 10000)
            else:
                # Structure preserving: try to match another value's ratio or sum
                j = random.randint(0, n - 1)
                if i != j:
                    # Try to make v_i + v_j collide with an existing sum or product?
                    # Simple heuristic: swap values
                    new_val = values[j]
                    values[j] = old_val
                    # Check validity (distinctness)
                    # Swapping keeps set of values same, so distinctness is preserved.
                    # Evaluate
                    new_score = evaluate(values, edges)
                    if new_score < best_score:
                        best_score = new_score
                        best_values = list(values)
                    # Decide accept/reject for current state
                    if new_score < current_score:
                        current_score = new_score
                    else:
                        # Revert swap
                        values[i], values[j] = values[j], values[i]
                        # Do not update current_score
                        T *= cooling
                        continue
                
                # If swap was rejected or not performed, proceed with new_val logic?
                # Actually, swap is a complete move. If rejected, we revert and continue.
                # If accepted, we continue.
                # The logic above handles swap separately.
                # Let's simplify: just do perturbations.
                pass
            
            # Ensure new_val is valid
            if new_val < 1 or new_val > 10000:
                new_val = random.randint(1, 10000)
            
            # Check distinctness
            if any(new_val == values[k] for k in range(n) if k != i):
                # If collision, try to adjust
                # Simple retry
                new_val = random.randint(1, 10000)
                if any(new_val == values[k] for k in range(n) if k != i):
                    T *= cooling
                    continue
            
            values[i] = new_val
            new_score = evaluate(values, edges)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_values = list(values)
            else:
                # Metropolis criterion
                if random.random() < math.exp(-(new_score - current_score) / T):
                    current_score = new_score
                else:
                    values[i] = old_val
            
            T *= cooling
        
        return best_values, best_score

    best_overall_score = float('inf')
    best_result = None
    
    # Strategy 1: Min Sum Graph with AP/GP/Random starts
    graph_min_sum = get_min_sum_graph()
    
    initial_strategies = [
        list(range(1, n + 1)),
        [2 ** i for i in range(n)],
        [3 ** i for i in range(n) if 3**i <= 10000] + [10000 - i for i in range(n - len([3**i for i in range(n) if 3**i <= 10000]))], # Fallback
        random.sample(range(1, 10001), n),
        [i * 100 for i in range(1, n + 1)],
        [i * 50 for i in range(1, n + 1)],
    ]
    
    # Clean up initial strategies to ensure distinct and <= 10000
    valid_initials = []
    for s in initial_strategies:
        if len(set(s)) == n and all(1 <= v <= 10000 for v in s):
            valid_initials.append(s)
    
    if not valid_initials:
        valid_initials = [list(range(1, n + 1))]
    
    # Run SA on Min Sum Graph
    for init_vals in valid_initials:
        if time.time() > deadline - 5:
            break
        vals, score = simulated_annealing(list(init_vals), graph_min_sum, deadline - 5)
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': vals, 'edges': graph_min_sum}
    
    # Strategy 2: Try a few other graphs
    # Maybe a graph with fewer distinct products for AP?
    # Random graphs might have better product structures?
    for _ in range(5):
        if time.time() > deadline - 5:
            break
        g = get_random_graph()
        vals, score = simulated_annealing(list(range(1, n + 1)), g, deadline - 5)
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': vals, 'edges': g}

    # Final Polish: Local search on the best result found
    if best_result:
        vals = list(best_result['values'])
        edges = list(best_result['edges'])
        
        # Try to improve by swapping edges
        edges_set = set(tuple(e) for e in edges)
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            i = random.randint(0, m - 1)
            new_edge = random.choice(all_edges)
            if tuple(new_edge) in edges_set:
                continue
            
            old_edge = edges[i]
            edges[i] = new_edge
            edges_set.discard(tuple(old_edge))
            edges_set.add(tuple(new_edge))
            
            new_score = evaluate(vals, edges)
            if new_score < best_overall_score:
                best_overall_score = new_score
                best_result = {'values': vals, 'edges': edges}
            else:
                edges[i] = old_edge
                edges_set.discard(tuple(new_edge))
                edges_set.add(tuple(old_edge))
        
        # Try to improve values one last time
        vals, score = simulated_annealing(vals, edges, deadline - 2)
        if score < best_overall_score:
            best_overall_score = score
            best_result = {'values': vals, 'edges': edges}

    if best_result is None:
        best_result = {
            'values': list(range(1, n + 1)),
            'edges': graph_min_sum
        }
    
    values = [int(v) for v in best_result['values']]
    edges = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return {'values': values, 'edges': edges}
