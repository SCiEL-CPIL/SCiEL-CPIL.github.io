import time
import sys
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_constraint_indices(n):
    """Build index arrays for quad constraints (4 triples each) and quint constraints (10 triples each)."""
    triples = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Quad constraints: each 4-set has 4 triples
    quad_constraints = []
    for quad in combinations(range(n), 4):
        ts = [
            (quad[0], quad[1], quad[2]),
            (quad[0], quad[1], quad[3]),
            (quad[0], quad[2], quad[3]),
            (quad[1], quad[2], quad[3]),
        ]
        quad_constraints.append([triple_to_idx[t] for t in ts])
    
    # Quint constraints: each 5-set has 10 triples
    quint_constraints = []
    for quint in combinations(range(n), 5):
        ts = []
        for i in range(5):
            for j in range(i + 1, 5):
                for k in range(j + 1, 5):
                    ts.append((quint[i], quint[j], quint[k]))
        quint_constraints.append([triple_to_idx[t] for t in ts])
    
    return np.array(quad_constraints, dtype=np.int32), np.array(quint_constraints, dtype=np.int32)

def _count_violations_vec(colors, quad_idx, quint_idx):
    """Count violations using vectorized NumPy operations."""
    quad_colors = colors[quad_idx]
    red_viol = np.all(quad_colors == 0, axis=1).sum()
    
    quint_colors = colors[quint_idx]
    blue_viol = np.all(quint_colors == 1, axis=1).sum()
    
    return int(red_viol + blue_viol)

def _violation_driven_search(colors, quad_idx, quint_idx, max_iters, deadline):
    """Refine a coloring using violation-driven local search."""
    colors = colors.copy()
    num_triples = len(colors)
    cur_viol = _count_violations_vec(colors, quad_idx, quint_idx)
    
    if cur_viol == 0:
        return colors, 0
        
    rng = np.random.RandomState(42)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if cur_viol == 0:
            break
            
        # Find violated constraints
        quad_colors = colors[quad_idx]
        red_viol_mask = np.all(quad_colors == 0, axis=1)
        red_viol_indices = np.where(red_viol_mask)[0]
        
        quint_colors = colors[quint_idx]
        blue_viol_mask = np.all(quint_colors == 1, axis=1)
        blue_viol_indices = np.where(blue_viol_mask)[0]
        
        # Pick a random violation
        use_red = False
        if len(red_viol_indices) > 0:
            if len(blue_viol_indices) == 0:
                use_red = True
            else:
                use_red = rng.random() < 0.5
                
        if use_red:
            viol_idx = red_viol_indices[rng.randint(0, len(red_viol_indices))]
            triple_indices_in_viol = quad_idx[viol_idx]
            triple_to_flip = triple_indices_in_viol[rng.randint(0, 4)]
            target_color = 1 
        else:
            if len(blue_viol_indices) == 0:
                break
            viol_idx = blue_viol_indices[rng.randint(0, len(blue_viol_indices))]
            triple_indices_in_viol = quint_idx[viol_idx]
            triple_to_flip = triple_indices_in_viol[rng.randint(0, 10)]
            target_color = 0
            
        old_color = colors[triple_to_flip]
        colors[triple_to_flip] = target_color
        
        new_viol = _count_violations_vec(colors, quad_idx, quint_idx)
        
        if new_viol < cur_viol:
            cur_viol = new_viol
        else:
            colors[triple_to_flip] = old_color
            
    return colors, cur_viol

def _genetic_search(n, deadline):
    """Run a genetic algorithm to find a valid coloring."""
    quad_idx, quint_idx = _build_constraint_indices(n)
    num_triples = n * (n - 1) * (n - 2) // 6
    
    pop_size = 40
    rng = np.random.RandomState(123)
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        c = rng.randint(0, 2, size=num_triples).astype(np.int8)
        population.append(c)
        
    best_viol = float('inf')
    best_colors = None
    
    while time.time() < deadline:
        # Evaluate fitness
        fitnesses = []
        for c in population:
            v = _count_violations_vec(c, quad_idx, quint_idx)
            fitnesses.append(v)
            
        min_viol = min(fitnesses)
        if min_viol < best_viol:
            best_viol = min_viol
            best_colors = population[fitnesses.index(min_viol)].copy()
            
        if best_viol == 0:
            break
            
        # Selection and Crossover
        new_population = []
        
        # Elitism
        sorted_idx = np.argsort(fitnesses)
        for i in range(2):
            new_population.append(population[sorted_idx[i]].copy())
            
        while len(new_population) < pop_size:
            # Tournament selection
            i1, i2 = rng.randint(0, pop_size, 2)
            p1 = population[i1] if fitnesses[i1] <= fitnesses[i2] else population[i2]
            
            i3, i4 = rng.randint(0, pop_size, 2)
            p2 = population[i3] if fitnesses[i3] <= fitnesses[i4] else population[i4]
            
            # Uniform crossover
            mask = rng.randint(0, 2, size=num_triples).astype(np.int8)
            child = np.where(mask == 1, p1, p2).astype(np.int8)
            
            # Mutation
            mut_rate = 0.01
            mut_mask = (rng.random(size=num_triples) < mut_rate).astype(np.int8)
            child = child ^ mut_mask
            
            new_population.append(child)
            
        population = new_population
        
        # Local search on best few
        if time.time() < deadline - 2:
            sorted_idx = np.argsort(fitnesses)
            for i in range(2):
                idx = sorted_idx[i]
                c, v = _violation_driven_search(population[idx], quad_idx, quint_idx, 100, time.time() + 1)
                population[idx] = c
                if v < best_viol:
                    best_viol = v
                    best_colors = c.copy()
                    if best_viol == 0:
                        break
                        
    if best_viol == 0:
        return best_colors.tolist()
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    for n in range(34, 9, -1):
        if time.time() > deadline - 5:
            break
            
        result = _genetic_search(n, deadline)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback
    n = 9
    colors = [0] * (n*(n-1)*(n-2)//6)
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
