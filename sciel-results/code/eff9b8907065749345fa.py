"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import defaultdict

P = {'n': 14, 'm': 35, 'maximum': 10000}
N = P['n']
M = P['m']
MAX_VAL = P['maximum']

def _raw_value(values, edges):
    """Compute the raw objective value."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _all_pairs(n):
    """Return all possible edges as a list of (i, j) tuples."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _compute_scores(values):
    """Precompute sum and product scores for all edges."""
    all_edges = _all_pairs(N)
    sum_scores = {}
    prod_scores = {}
    for i, j in all_edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sum_scores[(i, j)] = s
        prod_scores[(i, j)] = p
    return all_edges, sum_scores, prod_scores

def _evaluate(values, edges):
    """Evaluate the objective for given values and edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        sums.add(values[i] + values[j])
        prods.add(values[i] * values[j])
    return max(len(sums), len(prods))

def _construct_initial_values():
    """Construct initial values with high collision structure."""
    # Use a geometric-like sequence to encourage product collisions
    # and arithmetic structure for sum collisions
    values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    return values

def _simulated_annealing_edges(values, all_edges, sum_scores, prod_scores, deadline):
    """Find good edge set via simulated annealing."""
    best_edges = None
    best_val = M + 1
    
    # Start with random edge set
    edges = random.sample(all_edges, M)
    cur_val = _evaluate(values, edges)
    best_edges = edges[:]
    best_val = cur_val
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    while T > T_min and time.time() < deadline:
        # Propose move: swap one edge
        i = random.randint(0, M - 1)
        j = random.randint(0, len(all_edges) - 1)
        old_edge = edges[i]
        new_edge = all_edges[j]
        
        if old_edge == new_edge:
            continue
        
        # Compute new objective efficiently
        # Remove old edge, add new edge
        new_edges = edges[:]
        new_edges[i] = new_edge
        
        # Check for duplicates
        if len(set(map(tuple, new_edges))) != M:
            continue
        
        new_val = _evaluate(values, new_edges)
        
        if new_val < cur_val or random.random() < min(1.0, (cur_val - new_val) / T if T > 0 else 0):
            edges = new_edges
            cur_val = new_val
            if cur_val < best_val:
                best_val = cur_val
                best_edges = edges[:]
        
        T *= cooling
    
    return best_edges, best_val

def _local_search_edges(values, all_edges, sum_scores, prod_scores, deadline):
    """Local search: try swapping edges to improve."""
    edges = random.sample(all_edges, M)
    cur_val = _evaluate(values, edges)
    best_edges = edges[:]
    best_val = cur_val
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(M):
            for j in range(len(all_edges)):
                new_edge = all_edges[j]
                if new_edge == edges[i]:
                    continue
                
                new_edges = edges[:]
                new_edges[i] = new_edge
                
                if len(set(map(tuple, new_edges))) != M:
                    continue
                
                new_val = _evaluate(values, new_edges)
                if new_val < cur_val:
                    edges = new_edges
                    cur_val = new_val
                    improved = True
                    if cur_val < best_val:
                        best_val = cur_val
                        best_edges = edges[:]
                    break
    
    return best_edges, best_val

def construct():
    deadline = time.time() + 55.0  # Safety margin
    
    # Start with structured values
    values = _construct_initial_values()
    all_edges, sum_scores, prod_scores = _compute_scores(values)
    
    best_result = None
    best_score = M + 1
    
    # Try multiple random restarts with different value sets
    attempts = 0
    while time.time() < deadline and attempts < 20:
        if time.time() > deadline - 5:
            break
        
        # Generate value set
        if attempts == 0:
            vals = _construct_initial_values()
        elif attempts % 3 == 1:
            # Random values with some structure
            base = random.randint(1, 50)
            ratio = random.uniform(1.2, 2.0)
            vals = []
            v = base
            for _ in range(N):
                vals.append(int(v))
                v *= ratio
            # Ensure distinct and within bounds
            vals = sorted(set(min(v, MAX_VAL) for v in vals))
            while len(vals) < N:
                vals.append(random.randint(1, MAX_VAL))
            vals = list(dict.fromkeys(vals))[:N]
            random.shuffle(vals)
        else:
            # Random values
            vals = random.sample(range(1, MAX_VAL + 1), N)
        
        if len(set(vals)) != N:
            continue
        
        # Find good edges
        edges, score = _simulated_annealing_edges(vals, all_edges, sum_scores, prod_scores, deadline)
        
        if score < best_score:
            best_score = score
            best_result = (vals, edges)
        
        attempts += 1
    
    if best_result is None:
        # Fallback
        vals = list(range(1, N + 1))
        edges = [list(e) for e in list(combinations(range(N), 2))[:M]]
        best_result = (vals, edges)
    
    vals, edges = best_result
    return {'values': [int(v) for v in vals], 'edges': [[int(i), int(j)] for i, j in edges]}
# EVOLVE-BLOCK-END
