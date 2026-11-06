"""Improved program for hypergraph Ramsey R(4,5;3) colouring search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

DEADLINE = 140.0  # seconds; keep 10s safety margin under 150s wall-clock

def _triple_indices(n):
    """Return list of (a,b,c) triples in lexicographic order."""
    return list(itertools.combinations(range(n), 3))

def _build_k4_sets(n):
    """For each 4-set of vertices, return the 4 triple indices that form it."""
    k4_sets = []
    triples = list(itertools.combinations(range(n), 3))
    triple_idx = {t: i for i, t in enumerate(triples)}
    for quad in itertools.combinations(range(n), 4):
        indices = []
        for i in range(4):
            t = tuple(quad[j] for j in range(4) if j != i)
            indices.append(triple_idx[t])
        k4_sets.append(tuple(indices))
    return k4_sets

def _build_k5_sets(n):
    """For each 5-set of vertices, return the 10 triple indices that form it."""
    k5_sets = []
    triples = list(itertools.combinations(range(n), 3))
    triple_idx = {t: i for i, t in enumerate(triples)}
    for quint in itertools.combinations(range(n), 5):
        indices = []
        for tri in itertools.combinations(quint, 3):
            indices.append(triple_idx[tri])
        k5_sets.append(tuple(indices))
    return k5_sets

def _count_violations(colors, k4_sets, k5_sets):
    """Count red K4 violations and blue K5 violations."""
    red_viol = 0
    for indices in k4_sets:
        # Check if all 4 are red (0)
        if all(colors[i] == 0 for i in indices):
            red_viol += 1
    blue_viol = 0
    for indices in k5_sets:
        # Check if all 10 are blue (1)
        if all(colors[i] == 1 for i in indices):
            blue_viol += 1
    return red_viol + blue_viol

def _is_valid(colors, k4_sets, k5_sets):
    return _count_violations(colors, k4_sets, k5_sets) == 0

def _random_colors(n):
    """Random initial coloring."""
    m = len(list(itertools.combinations(range(n), 3)))
    return np.random.randint(0, 2, size=m)

def _structured_init(n, seed_type='mod3'):
    """Structured initial coloring using modular arithmetic."""
    triples = list(itertools.combinations(range(n), 3))
    if seed_type == 'mod3':
        return np.array([0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples])
    elif seed_type == 'mod2':
        return np.array([0 if (a + b + c) % 2 == 0 else 1 for a, b, c in triples])
    elif seed_type == 'xor':
        return np.array([0 if ((a ^ b ^ c) & 1) == 0 else 1 for a, b, c in triples])
    else:
        return np.array([0 if (a * b + c) % 3 == 0 else 1 for a, b, c in triples])

def _simulated_annealing(colors, k4_sets, k5_sets, triples, n, start_time, max_time, rng):
    """Simulated annealing to minimize violations."""
    current = colors.copy()
    current_cost = _count_violations(current, k4_sets, k5_sets)
    best = current.copy()
    best_cost = current_cost
    
    m = len(triples)
    
    # Temperature schedule
    T = 1.0
    cooling = 0.999
    
    while time.time() - start_time < max_time:
        # Pick a random triple to flip
        idx = rng.randint(0, m - 1)
        old_val = current[idx]
        new_val = 1 - old_val
        current[idx] = new_val
        
        # Compute cost change efficiently
        new_cost = _count_violations(current, k4_sets, k5_sets)
        delta = new_cost - current_cost
        
        if delta <= 0 or rng.random() < np.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best = current.copy()
                if best_cost == 0:
                    break
        else:
            current[idx] = old_val
        
        T *= cooling
        if T < 0.01:
            T = 0.01
    
    return best, best_cost

def _greedy_flip(colors, k4_sets, k5_sets, triples, n, start_time, max_time, rng):
    """Greedy hill climbing: flip single triples to reduce violations."""
    current = colors.copy()
    current_cost = _count_violations(current, k4_sets, k5_sets)
    best = current.copy()
    best_cost = current_cost
    
    m = len(triples)
    improved = True
    while improved and time.time() - start_time < max_time:
        improved = False
        # Shuffle order for diversity
        order = rng.permutation(m)
        for idx in order:
            old_val = current[idx]
            new_val = 1 - old_val
            current[idx] = new_val
            new_cost = _count_violations(current, k4_sets, k5_sets)
            if new_cost < current_cost:
                current_cost = new_cost
                improved = True
                if current_cost < best_cost:
                    best_cost = current_cost
                    best = current.copy()
                    if best_cost == 0:
                        return best, 0
            else:
                current[idx] = old_val
        # If no improvement in full pass, try random perturbations
        if not improved:
            # Perturb a few random triples
            for _ in range(3):
                idx = rng.randint(0, m - 1)
                current[idx] = 1 - current[idx]
            current_cost = _count_violations(current, k4_sets, k5_sets)
            improved = True
    
    return best, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 10  # 10s safety margin
    
    rng = np.random.RandomState(42)
    
    best_n = 0
    best_colors = None
    
    # Try increasing n from small to large
    # The published construction has n=34, so aim for that
    # Start with structured inits and use SA + greedy
    
    for n in [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]:
        if time.time() > deadline:
            break
        
        triples = list(itertools.combinations(range(n), 3))
        m = len(triples)
        k4_sets = _build_k4_sets(n)
        k5_sets = _build_k5_sets(n)
        
        n_k4 = len(k4_sets)
        n_k5 = len(k5_sets)
        
        # Multiple restarts with different strategies
        max_time_for_n = min(30.0, max(5.0, (deadline - time.time()) * 0.5))
        
        found_valid = False
        best_local = None
        best_local_cost = float('inf')
        
        # Try different initializations
        inits = [
            ('mod3', lambda: _structured_init(n, 'mod3')),
            ('mod2', lambda: _structured_init(n, 'mod2')),
            ('xor', lambda: _structured_init(n, 'xor')),
            ('rand1', lambda: _random_colors(n)),
            ('rand2', lambda: _random_colors(n)),
        ]
        
        for init_name, init_func in inits:
            if time.time() > deadline:
                break
            
            colors = init_func()
            cost = _count_violations(colors, k4_sets, k5_sets)
            
            if cost == 0:
                found_valid = True
                best_n = n
                best_colors = colors.tolist()
                break
            
            # Run simulated annealing
            sa_time = min(10.0, max(2.0, (deadline - time.time()) * 0.3))
            sa_colors, sa_cost = _simulated_annealing(
                colors, k4_sets, k5_sets, triples, n, time.time(), sa_time, rng
            )
            
            if sa_cost == 0:
                found_valid = True
                best_n = n
                best_colors = sa_colors.tolist()
                break
            
            if sa_cost < best_local_cost:
                best_local_cost = sa_cost
                best_local = sa_colors
            
            # Run greedy from SA result
            greedy_time = min(8.0, max(1.0, (deadline - time.time()) * 0.2))
            g_colors, g_cost = _greedy_flip(
                sa_colors, k4_sets, k5_sets, triples, n, time.time(), greedy_time, rng
            )
            
            if g_cost == 0:
                found_valid = True
                best_n = n
                best_colors = g_colors.tolist()
                break
            
            if g_cost < best_local_cost:
                best_local_cost = g_cost
                best_local = g_colors
        
        if found_valid:
            # Try to go higher
            continue
        else:
            # Didn't find valid for this n, but keep going to try larger
            # Actually, if we can't find valid for n, we likely can't for larger
            # But let's try a couple more with more time
            pass
    
    # If we found a valid coloring, return it
    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback: return the best we found (even if invalid, but we should try hard)
    # Actually, the task says maximize n with valid coloring. If we can't find valid,
    # we should return the largest n where we found valid. If none, return small valid.
    
    # Let's do a final quick search for small n that we know is feasible
    for n in [9, 10, 11, 12]:
        if time.time() > deadline:
            break
        triples = list(itertools.combinations(range(n), 3))
        k4_sets = _build_k4_sets(n)
        k5_sets = _build_k5_sets(n)
        
        for _ in range(20):
            colors = _random_colors(n)
            cost = _count_violations(colors, k4_sets, k5_sets)
            if cost == 0:
                return {'n': n, 'colors': colors.tolist()}
            
            # Quick SA
            t0 = time.time()
            sa_colors, sa_cost = _simulated_annealing(
                colors, k4_sets, k5_sets, triples, n, t0, 1.0, rng
            )
            if sa_cost == 0:
                return {'n': n, 'colors': sa_colors.tolist()}
    
    # Ultimate fallback: n=1 trivially valid
    return {'n': 1, 'colors': []}
# EVOLVE-BLOCK-END
