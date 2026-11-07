import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 sec safety margin before 90s kill
    
    n = 14
    target_edges = 35
    all_edges = list(combinations(range(n), 2))
    
    # We will optimize a score that is equivalent to the raw value but easier to work with.
    # Raw = max(distinct_sums, distinct_products).
    # To minimize this, we want to maximize collisions.
    # Let S be the set of sums, P be the set of products.
    # We want |S| small and |P| small.
    
    # Initial setup
    # Start with a random graph and random values
    # Or start with a structured graph (e.g. random 35 edges) and structured values
    
    def get_initial_state():
        # Random distinct values
        vals = random.sample(range(1, 10001), n)
        # Random edges
        edges = set(random.sample(all_edges, target_edges))
        return vals, edges

    def evaluate(vals, edges):
        sums = set()
        prods = set()
        for u, v in edges:
            sums.add(vals[u] + vals[v])
            prods.add(vals[u] * vals[v])
        return max(len(sums), len(prods))

    # Precompute edge indices for faster access? 
    # Actually, with n=14, 91 edges, and 35 selected, iterating is fast.
    
    best_raw = float('inf')
    best_state = None
    
    # Run multiple restarts
    # Budget: 85 seconds. 
    # Let's do 10 restarts, each with a time limit.
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Mix of strategies for initialization
        if r % 2 == 0:
            # Structured values: Arithmetic progression
            start = random.randint(1, 100)
            step = random.randint(1, 10)
            vals = [start + i * step for i in range(n)]
            # Ensure distinct and <= 10000
            if len(set(vals)) < n or max(vals) > 10000:
                 vals = random.sample(range(1, 10001), n)
            
            # Structured edges: Random
            edges = set(random.sample(all_edges, target_edges))
        else:
            # Random values
            vals = random.sample(range(1, 10001), n)
            # Greedy edges for these values?
            # Let's just use random edges for simplicity in SA start, 
            # or a quick greedy pass.
            edge_data = [(u, v, vals[u]+vals[v], vals[u]*vals[v]) for u, v in all_edges]
            edge_data.sort(key=lambda x: x[2]) # Sort by sum
            edges = set((e[0], e[1]) for e in edge_data[:target_edges])
        
        current_raw = evaluate(vals, edges)
        if current_raw < best_raw:
            best_raw = current_raw
            best_state = (vals[:], edges.copy())
            
        # Simulated Annealing
        temp = 100.0
        cooling = 0.995
        min_temp = 0.1
        
        local_deadline = start_time + time_per_restart
        
        while time.time() < local_deadline and temp > min_temp:
            # Choose move
            move_type = random.random()
            
            new_vals = vals[:]
            new_edges = edges.copy()
            
            if move_type < 0.4:
                # Perturb Value
                idx = random.randint(0, n-1)
                # Random change
                if random.random() < 0.5:
                    new_vals[idx] = random.randint(1, 10000)
                else:
                    # Small step
                    step = random.randint(-50, 50)
                    new_val = new_vals[idx] + step
                    if 1 <= new_val <= 10000:
                        new_vals[idx] = new_val
            
            elif move_type < 0.8:
                # Swap Edge
                if edges:
                    e_in = random.choice(list(edges))
                    # Find an edge not in set
                    # To avoid O(N) list comp on every step, we can keep a list of non-edges?
                    # 91 total, 35 in. 56 out.
                    non_edges = [e for e in all_edges if e not in edges]
                    e_out = random.choice(non_edges)
                    new_edges.remove(e_in)
                    new_edges.add(e_out)
                    
            else:
                # Swap Values (permute indices)
                i, j = random.sample(range(n), 2)
                new_vals[i], new_vals[j] = new_vals[j], new_vals[i]
            
            # Validate distinctness for values
            if len(set(new_vals)) < n:
                continue
                
            new_raw = evaluate(new_vals, new_edges)
            
            # Acceptance
            if new_raw <= current_raw:
                current_raw = new_raw
                vals = new_vals
                edges = new_edges
                if current_raw < best_raw:
                    best_raw = current_raw
                    best_state = (vals[:], edges.copy())
            else:
                # Delta is positive (worse)
                delta = new_raw - current_raw
                # Probability of accepting worse move
                # exp(-delta / temp)
                if random.random() < pow(2.71828, -delta / temp):
                    current_raw = new_raw
                    vals = new_vals
                    edges = new_edges
            
            temp *= cooling

    if best_state:
        vals, edges = best_state
        # Ensure types
        final_vals = [int(v) for v in vals]
        final_edges = [[int(u), int(v)] for u, v in edges]
        return {'values': final_vals, 'edges': final_edges}
    else:
        # Fallback
        vals = list(range(1, 15))
        edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
        return {'values': vals, 'edges': edges}

if __name__ == "__main__":
    res = construct()
    # Quick check
    vals = res['values']
    edges = res['edges']
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(vals[u] + vals[v])
        prods.add(vals[u] * vals[v])
    print(f"Raw: {max(len(sums), len(prods))}")
    print(f"Distinct sums: {len(sums)}, Distinct prods: {len(prods)}")
