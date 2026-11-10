"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}
DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _score(values, edges):
    """Compute max(distinct_sums, distinct_products) for given values and edges."""
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))

def _valid_edges(n, m):
    """Generate all possible edges, return list of m of them (initially first m)."""
    all_edges = list(combinations(range(n), 2))
    return [list(e) for e in all_edges[:m]]

def _greedy_edges(values, n, m):
    """Greedy: pick edges that minimize growth in distinct sums and products."""
    sums = set()
    prods = set()
    edges = []
    all_pairs = list(combinations(range(n), 2))
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if len(edges) >= m:
            break
        a, b = values[u], values[v]
        s = a + b
        p = a * b
        # Count how many new distinct values this edge adds
        new_s = 1 if s not in sums else 0
        new_p = 1 if p not in prods else 0
        # Prefer edges that add few new distinct values
        cost = new_s + new_p
        edges.append((cost, u, v))
    
    edges.sort(key=lambda x: x[0])
    return [list(e[1:]) for e in edges[:m]]

def _local_search(values, edges, time_limit):
    """Local search: try swapping values and changing edges."""
    n = P['n']
    m = P['m']
    best_score = _score(values, edges)
    best_values = list(values)
    best_edges = [list(e) for e in edges]
    start = time.time()
    
    all_edges_set = set(tuple(e) for e in edges)
    
    while time.time() - start < time_limit:
        improved = False
        
        # Try swapping two values
        for i in range(n):
            for j in range(i+1, n):
                new_values = list(values)
                new_values[i], new_values[j] = new_values[j], new_values[i]
                s = _score(new_values, edges)
                if s < best_score:
                    values = new_values
                    best_values = list(values)
                    best_score = s
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        
        # Try changing one edge: remove one edge, add a different one
        for idx in range(m):
            old_u, old_v = edges[idx]
            for new_u in range(n):
                for new_v in range(new_u+1, n):
                    if (new_u, new_v) == (old_u, old_v) or (new_u, new_v) == (old_v, old_u):
                        continue
                    # Check no duplicate
                    if (new_u, new_v) in all_edges_set or (new_v, new_u) in all_edges_set:
                        continue
                    new_edges = [list(e) for e in edges]
                    new_edges[idx] = [new_u, new_v]
                    s = _score(values, new_edges)
                    if s < best_score:
                        edges = new_edges
                        best_edges = [list(e) for e in edges]
                        best_score = s
                        # Update edge set
                        all_edges_set.discard(tuple(sorted((old_u, old_v))))
                        all_edges_set.add(tuple(sorted((new_u, new_v))))
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_values, best_edges, best_score

def _initial_configurations():
    """Generate diverse initial configurations."""
    n = P['n']
    m = P['m']
    configs = []
    
    # Config 1: Powers of 2
    vals1 = [2**i for i in range(n)]
    edges1 = _greedy_edges(vals1, n, m)
    configs.append((list(vals1), edges1))
    
    # Config 2: Mixed multiplicative structure
    vals2 = [1, 2, 4, 8, 16, 3, 6, 12, 24, 48, 96, 192, 384, 768]
    edges2 = _greedy_edges(vals2, n, m)
    configs.append((list(vals2), edges2))
    
    # Config 3: Small consecutive values
    vals3 = list(range(1, n+1))
    edges3 = _greedy_edges(vals3, n, m)
    configs.append((list(vals3), edges3))
    
    # Config 4: Powers of 2 with some variation
    vals4 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    edges4 = _greedy_edges(vals4, n, m)
    configs.append((list(vals4), edges4))
    
    # Config 5: Random values with multiplicative bias
    random.seed(42)
    vals5 = [random.randint(1, 100) for _ in range(n)]
    # Ensure distinct
    while len(set(vals5)) < n:
        vals5[random.randint(0, n-1)] = random.randint(1, 100)
    edges5 = _greedy_edges(vals5, n, m)
    configs.append((list(vals5), edges5))
    
    return configs

def construct():
    start_time = time.time()
    
    configs = _initial_configurations()
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    for vals, edges in configs:
        if time.time() - start_time > DEADLINE - 5:
            break
        # Validate
        if len(set(vals)) != P['n']:
            continue
        if any(v < 1 or v > P['maximum'] for v in vals):
            continue
        if len(edges) != P['m']:
            continue
        # Check no loops or duplicates
        edge_set = set()
        valid = True
        for e in edges:
            if e[0] == e[1]:
                valid = False
                break
            key = tuple(sorted(e))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        if not valid:
            continue
        
        s = _score(vals, edges)
        if s < best_score:
            best_score = s
            best_values = list(vals)
            best_edges = [list(e) for e in edges]
        
        # Local search
        if time.time() - start_time > DEADLINE - 10:
            break
        v2, e2, s2 = _local_search(vals, edges, time_limit=min(10, DEADLINE - (time.time() - start_time) - 5))
        if s2 < best_score:
            best_score = s2
            best_values = v2
            best_edges = e2
    
    if best_values is None:
        # Fallback
        best_values = list(range(1, P['n']+1))
        best_edges = _valid_edges(P['n'], P['m'])
    
    # Final validation and type conversion
    best_values = [int(v) for v in best_values]
    best_edges = [[int(u), int(v)] for u, v in best_edges]
    
    return {'values': best_values, 'edges': best_edges}
# EVOLVE-BLOCK-END
