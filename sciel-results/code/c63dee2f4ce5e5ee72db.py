"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations
from collections import Counter

P = {'n': 14, 'm': 35, 'maximum': 10000}

def _raw_value(values, edges):
    """Compute the objective: max(#distinct sums, #distinct products)."""
    sums = set()
    prods = set()
    for i, j in edges:
        v = values[i]
        w = values[j]
        sums.add(v + w)
        prods.add(v * w)
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    n = P['n']
    m = P['m']
    maximum = P['maximum']
    
    best_raw = float('inf')
    best_values = None
    best_edges = None
    
    random.seed(42)
    
    all_pairs = list(combinations(range(n), 2))
    
    def try_value_set(values):
        """Given 14 values, find 35 edges minimizing the objective using structured init + SA."""
        # Precompute all pair (i,j) with i<j, and their sum and product
        edge_sums = [0] * len(all_pairs)
        edge_prods = [0] * len(all_pairs)
        for idx, (i, j) in enumerate(all_pairs):
            edge_sums[idx] = values[i] + values[j]
            edge_prods[idx] = values[i] * values[j]
        
        def get_score(selected_set):
            sums = set()
            prods = set()
            for idx in selected_set:
                sums.add(edge_sums[idx])
                prods.add(edge_prods[idx])
            return max(len(sums), len(prods))
        
        best_local_score = float('inf')
        best_local_edges = None
        
        # Multiple restarts with diverse initializations
        num_restarts = 50
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart % 5 == 0:
                # Random
                selected_indices = set(random.sample(range(len(all_pairs)), m))
            elif restart % 5 == 1:
                # Smallest sums
                sorted_by_sum = sorted(range(len(all_pairs)), key=lambda x: edge_sums[x])
                selected_indices = set(sorted_by_sum[:m])
            elif restart % 5 == 2:
                # Largest sums
                sorted_by_sum = sorted(range(len(all_pairs)), key=lambda x: edge_sums[x])
                selected_indices = set(sorted_by_sum[-m:])
            elif restart % 5 == 3:
                # Median sums
                sorted_by_sum = sorted(range(len(all_pairs)), key=lambda x: edge_sums[x])
                mid = len(sorted_by_sum) // 2
                selected_indices = set(sorted_by_sum[mid - m//2 : mid + m//2 + 1])
            else:
                # Greedy by collision potential
                sum_counts = Counter(edge_sums)
                prod_counts = Counter(edge_prods)
                sorted_by_pop = sorted(range(len(all_pairs)), 
                                       key=lambda x: (sum_counts[edge_sums[x]], prod_counts[edge_prods[x]]), 
                                       reverse=True)
                selected_indices = set(sorted_by_pop[:m])
            
            current_score = get_score(selected_indices)
            if current_score < best_local_score:
                best_local_score = current_score
                best_local_edges = set(selected_indices)
            
            # Simulated Annealing
            T = 10.0
            cooling = 0.97
            steps = 2000
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                
                # Pick a random edge to remove
                remove_idx = random.choice(list(selected_indices))
                # Pick a random edge to add (not currently selected)
                add_idx = random.choice(range(len(all_pairs)))
                while add_idx in selected_indices:
                    add_idx = random.choice(range(len(all_pairs)))
                
                new_selected = set(selected_indices)
                new_selected.remove(remove_idx)
                new_selected.add(add_idx)
                
                new_score = get_score(new_selected)
                
                if new_score < current_score:
                    selected_indices = new_selected
                    current_score = new_score
                    if current_score < best_local_score:
                        best_local_score = current_score
                        best_local_edges = set(selected_indices)
                else:
                    # Accept with probability
                    delta = new_score - current_score
                    if delta > 0:
                        p = math.exp(-delta / T)
                        if random.random() < p:
                            selected_indices = new_selected
                            current_score = new_score
                    else:
                        # If new_score == current_score, delta=0, p=1
                        selected_indices = new_selected
                        current_score = new_score
                        
                T *= cooling
                
                # Early stop if very good
                if best_local_score <= 8:
                    break
        
        if best_local_edges is not None:
            edges = [list(all_pairs[idx]) for idx in best_local_edges]
            raw = best_local_score
            return raw, edges
    
    # Generate diverse value sets and try them
    num_attempts = 100
    
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Generate a value set with emphasis on collision structures
        if attempt % 7 == 0:
            # Pairs that sum to same constant: a, b, c, d, ... where a+d=b+c=const
            # Use symmetric pairs around a center
            center = random.randint(20, 500)
            offsets = random.sample(range(1, center), 7)
            values = []
            for off in offsets:
                values.append(center - off)
                values.append(center + off)
            values = list(set(values))
            while len(values) < n:
                values.append(random.randint(1, maximum))
            values = list(set(values))[:n]
        elif attempt % 7 == 1:
            # Arithmetic progression
            a = random.randint(1, 50)
            d = random.randint(1, 30)
            values = [a + i*d for i in range(n)]
            if max(values) > maximum or min(values) < 1:
                values = list(range(1, n+1))
        elif attempt % 7 == 2:
            # Small clustered numbers
            base = random.randint(1, 20)
            values = [base + random.randint(0, 30) for _ in range(n)]
            values = list(set(values))
            while len(values) < n:
                values.append(random.randint(1, maximum))
            values = list(set(values))[:n]
        elif attempt % 7 == 3:
            # Random distinct small
            values = random.sample(range(1, 200), n)
        elif attempt % 7 == 4:
            # Mix small and medium
            small = random.sample(range(1, 50), n//2)
            medium = random.sample(range(50, 500), n - n//2)
            values = small + medium
            if len(set(values)) < n:
                values = random.sample(range(1, maximum+1), n)
        elif attempt % 7 == 5:
            # Dense small sets
            start = random.randint(1, 10)
            values = list(range(start, start + n))
        else:
            # Random
            values = random.sample(range(1, maximum+1), n)
        
        # Ensure valid
        values = list(set(values))
        if len(values) < n:
            continue
        values = values[:n]
        if any(v < 1 or v > maximum for v in values):
            continue
        
        raw, edges = try_value_set(values)
        
        if raw < best_raw:
            best_raw = raw
            best_values = values
            best_edges = edges
    
    # Fallback: always valid solution
    if best_values is None:
        best_values = list(range(1, n+1))
        best_edges = [list(e) for e in list(combinations(range(n), 2))[:m]]
    
    # Ensure types are correct
    best_values = [int(v) for v in best_values]
    best_edges = [[int(i), int(j)] for i, j in best_edges]
    
    return {'values': best_values, 'edges': best_edges}

# EVOLVE-BLOCK-END
