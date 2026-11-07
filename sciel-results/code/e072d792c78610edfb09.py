"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

N = 14
M = 35
MAX_VAL = 10000
DEADLINE = 85.0  # safety margin from 90s wall-clock

def _all_edges():
    return list(combinations(range(N), 2))

def _raw_score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        sums.add(values[u] + values[v])
        prods.add(values[u] * values[v])
    return max(len(sums), len(prods))

def _compute_scores(values):
    """Precompute sum and product for all pairs."""
    n = len(values)
    sums = [[0]*n for _ in range(n)]
    prods = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            sums[i][j] = sums[j][i] = values[i] + values[j]
            prods[i][j] = prods[j][i] = values[i] * values[j]
    return sums, prods

def _score_edge_set(values, edge_set, sums, prods):
    """Compute raw score for a given edge set."""
    s_set = set()
    p_set = set()
    for (u, v) in edge_set:
        s_set.add(sums[u][v])
        p_set.add(prods[u][v])
    return max(len(s_set), len(p_set))

def _greedy_edge_select(values, sums, prods):
    """Greedy: pick edges that minimize marginal increase in distinct sums/prods."""
    all_edges = _all_edges()
    chosen = set()
    s_counts = {}
    p_counts = {}
    s_set = set()
    p_set = set()
    
    for _ in range(M):
        best_edge = None
        best_score = float('inf')
        for e in all_edges:
            if e in chosen:
                continue
            u, v = e
            s_val = sums[u][v]
            p_val = prods[u][v]
            # Marginal increase
            ms = 0 if s_val in s_set else 1
            mp = 0 if p_val in p_set else 1
            new_s = len(s_set) + ms
            new_p = len(p_set) + mp
            score = max(new_s, new_p)
            if score < best_score:
                best_score = score
                best_edge = e
        if best_edge is None:
            break
        chosen.add(best_edge)
        u, v = best_edge
        s_set.add(sums[u][v])
        p_set.add(prods[u][v])
    
    return chosen

def _simulated_annealing(values, sums, prods, start_edges, t_start, t_end, n_iter, deadline):
    """SA on edge swaps to minimize raw score."""
    current_edges = set(start_edges)
    current_score = _score_edge_set(values, current_edges, sums, prods)
    best_edges = set(current_edges)
    best_score = current_score
    
    all_edges = _all_edges()
    edge_list = list(all_edges)
    
    t = t_start
    cooling = (t_end / t_start) ** (1.0 / max(n_iter, 1))
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        # Pick a random edge to remove and a random edge to add
        if len(current_edges) < M:
            # Add a random edge
            candidates = [e for e in edge_list if e not in current_edges]
            if not candidates:
                break
            e_add = random.choice(candidates)
            new_edges = current_edges | {e_add}
        else:
            # Swap: remove one, add one
            e_remove = random.choice(list(current_edges))
            candidates = [e for e in edge_list if e not in current_edges]
            if not candidates:
                break
            e_add = random.choice(candidates)
            new_edges = (current_edges - {e_remove}) | {e_add}
        
        new_score = _score_edge_set(values, new_edges, sums, prods)
        
        if new_score < current_score:
            current_edges = new_edges
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_edges = set(current_edges)
        else:
            delta = new_score - current_score
            if delta > 0:
                accept_prob = math.exp(-delta / max(t, 1e-10))
                if random.random() < accept_prob:
                    current_edges = new_edges
                    current_score = new_score
        
        t *= cooling
    
    return best_edges, best_score

def _local_search_edges(values, sums, prods, edge_set, deadline):
    """First-improvement local search via edge swaps."""
    current_edges = set(edge_set)
    current_score = _score_edge_set(values, current_edges, sums, prods)
    all_edges = _all_edges()
    edge_list = list(all_edges)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for e_remove in list(current_edges):
            if time.time() > deadline:
                break
            for e_add in edge_list:
                if e_add in current_edges:
                    continue
                new_edges = (current_edges - {e_remove}) | {e_add}
                new_score = _score_edge_set(values, new_edges, sums, prods)
                if new_score < current_score:
                    current_edges = new_edges
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
    
    return current_edges, current_score

def _generate_values_strategy(strategy_id, rng):
    """Generate 14 distinct positive integers <= 10000 using different strategies."""
    n = N
    
    if strategy_id == 0:
        # Consecutive integers
        return list(range(1, n+1))
    elif strategy_id == 1:
        # Small primes
        primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43]
        return primes[:n]
    elif strategy_id == 2:
        # Powers of 2 (small)
        return [2**i for i in range(n)]
    elif strategy_id == 3:
        # Fibonacci-like
        vals = [1, 2]
        while len(vals) < n:
            vals.append(vals[-1] + vals[-2])
        return vals[:n]
    elif strategy_id == 4:
        # Random small values with some structure
        base = rng.randint(1, 5)
        step = rng.randint(1, 3)
        vals = [base + i*step for i in range(n)]
        # Ensure distinct and <= 10000
        if max(vals) > MAX_VAL or len(set(vals)) < n:
            return list(range(1, n+1))
        return vals
    elif strategy_id == 5:
        # Mix of small and medium
        small = list(range(1, n//2+1))
        large = [rng.randint(n//2+1, 20) for _ in range(n - n//2)]
        vals = small + large
        rng.shuffle(vals)
        # Ensure distinct
        if len(set(vals)) < n:
            return list(range(1, n+1))
        return vals
    elif strategy_id == 6:
        # Arithmetic progression with random start/step
        a = rng.randint(1, 10)
        d = rng.randint(1, 5)
        vals = [a + i*d for i in range(n)]
        if max(vals) > MAX_VAL or len(set(vals)) < n:
            return list(range(1, n+1))
        return vals
    elif strategy_id == 7:
        # Random permutation of 1..n with scaling
        vals = list(range(1, n+1))
        rng.shuffle(vals)
        return vals
    elif strategy_id == 8:
        # Geometric-like
        ratio = rng.randint(2, 3)
        vals = []
        v = rng.randint(1, 3)
        while len(vals) < n and v <= MAX_VAL:
            vals.append(v)
            v *= ratio
        while len(vals) < n:
            vals.append(rng.randint(1, MAX_VAL))
        # Ensure distinct
        seen = set()
        unique = []
        for v in vals:
            if v not in seen and 1 <= v <= MAX_VAL:
                seen.add(v)
                unique.append(v)
        while len(unique) < n:
            v = rng.randint(1, MAX_VAL)
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return unique[:n]
    elif strategy_id == 9:
        # Twin primes / close pairs
        vals = []
        v = 2
        while len(vals) < n and v <= MAX_VAL:
            vals.append(v)
            v += rng.randint(1, 3)
        while len(vals) < n:
            v = rng.randint(1, MAX_VAL)
            if v not in vals:
                vals.append(v)
        return vals[:n]
    else:
        return list(range(1, n+1))

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_values = None
    best_edges = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    strategies = list(range(10))
    rng.shuffle(strategies)
    
    for strat_id in strategies:
        if time.time() > deadline - 5.0:
            break
        
        values = _generate_values_strategy(strat_id, rng)
        
        # Validate
        if len(values) != N or len(set(values)) != N:
            continue
        if any(not isinstance(v, int) or v < 1 or v > MAX_VAL for v in values):
            continue
        
        sums, prods = _compute_scores(values)
        
        # Greedy edge selection
        greedy_edges = _greedy_edge_select(values, sums, prods)
        greedy_score = _score_edge_set(values, greedy_edges, sums, prods)
        
        if greedy_score < best_score:
            best_score = greedy_score
            best_values = list(values)
            best_edges = set(greedy_edges)
        
        # Local search from greedy
        if time.time() > deadline - 10.0:
            break
        ls_edges, ls_score = _local_search_edges(values, sums, prods, greedy_edges, deadline - 5.0)
        if ls_score < best_score:
            best_score = ls_score
            best_values = list(values)
            best_edges = set(ls_edges)
        
        # SA with multiple restarts
        if time.time() > deadline - 10.0:
            break
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
            # Perturb greedy solution
            perturbed = set(greedy_edges)
            for _ in range(rng.randint(5, 15)):
                e_remove = rng.choice(list(perturbed))
                candidates = [e for e in _all_edges() if e not in perturbed]
                if candidates:
                    e_add = rng.choice(candidates)
                    perturbed = (perturbed - {e_remove}) | {e_add}
            
            sa_edges, sa_score = _simulated_annealing(
                values, sums, prods, perturbed,
                t_start=1.0, t_end=0.01,
                n_iter=2000,
                deadline=deadline - 5.0
            )
            if sa_score < best_score:
                best_score = sa_score
                best_values = list(values)
                best_edges = set(sa_edges)
    
    # Final local search on best
    if time.time() < deadline - 2.0 and best_values is not None and best_edges is not None:
        sums, prods = _compute_scores(best_values)
        final_edges, final_score = _local_search_edges(best_values, sums, prods, best_edges, deadline - 1.0)
        if final_score < best_score:
            best_score = final_score
            best_edges = set(final_edges)
    
    # Ensure valid output
    if best_values is None or best_edges is None:
        best_values = list(range(1, N+1))
        all_e = _all_edges()
        best_edges = set(all_e[:M])
    
    # Convert to proper types
    edges_list = [[u, v] for u, v in sorted(best_edges)]
    values_list = [int(v) for v in best_values]
    
    return {'values': values_list, 'edges': edges_list}

# EVOLVE-BLOCK-END
