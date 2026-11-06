"""Construct a graph with no regular induced subgraph on >= 6 vertices."""
import time
import random
from itertools import combinations

_START_TIME = time.time()
DEADLINE = 125.0

def _elapsed():
    return time.time() - _START_TIME

def _is_regular_subset_fast(subset, adj):
    n = len(subset)
    degs = [0] * n
    for i in range(n):
        u = subset[i]
        for j in range(n):
            if i != j:
                degs[i] += (adj[u] >> subset[j]) & 1
    first = degs[0]
    for d in degs[1:]:
        if d != first:
            return False
    return True

def _count_violations_6(n, adj, six_subs):
    count = 0
    for sub in six_subs:
        if _is_regular_subset_fast(sub, adj):
            count += 1
    return count

def _has_violation_full(n, adj):
    for size in range(6, n + 1):
        for sub in combinations(range(n), size):
            if _is_regular_subset_fast(sub, adj):
                return True
    return False

def _edges_from_adj(adj, n):
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

def _ga_search(n, pop_size=200, max_gen=500, time_budget=50):
    deadline = _START_TIME + time_budget
    rng = random.Random(42)
    
    six_subs = list(combinations(range(n), 6))
    
    # Initialize population
    pop = []
    for _ in range(pop_size):
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        pop.append(adj)
    
    best_adj = None
    best_score = float('inf')
    
    for gen in range(max_gen):
        if _elapsed() > deadline:
            break
        
        # Evaluate
        scored = []
        for adj in pop:
            score = _count_violations_6(n, adj, six_subs)
            scored.append((score, adj))
        
        scored.sort(key=lambda x: x[0])
        best_score, best_adj = scored[0]
        
        if best_score == 0:
            if not _has_violation_full(n, best_adj):
                return best_adj
        
        # Create new population
        new_pop = []
        
        # Elitism: keep top 20%
        elite_count = max(10, pop_size // 5)
        for i in range(elite_count):
            new_pop.append(scored[i][1][:])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            p1 = scored[rng.randrange(min(5, len(scored)))]
            p2 = scored[rng.randrange(min(5, len(scored)))]
            parent1 = p1[1] if p1[0] < p2[0] else p2[1]
            parent2 = scored[rng.randrange(min(10, len(scored)))]
            
            # Crossover
            child = [0] * n
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < 0.5:
                        child[i] |= (parent1[i] >> j) & 1
                        child[j] |= (parent1[i] >> j) & 1
                    else:
                        child[i] |= (parent2[i] >> j) & 1
                        child[j] |= (parent2[i] >> j) & 1
            
            # Mutation
            for i in range(n):
                for j in range(i+1, n):
                    if rng.random() < 0.05:
                        if (child[i] >> j) & 1:
                            child[i] &= ~(1 << j)
                            child[j] &= ~(1 << i)
                        else:
                            child[i] |= (1 << j)
                            child[j] |= (1 << i)
            
            new_pop.append(child)
        
        pop = new_pop
    
    return best_adj if best_score == 0 else None

def construct():
    global _START_TIME, DEADLINE
    _START_TIME = time.time()
    DEADLINE = _START_TIME + 125.0
    
    for target_n in [22, 21, 20, 19]:
        if _elapsed() > DEADLINE - 10:
            break
        budget = min(40, DEADLINE - _elapsed() - 10)
        if budget < 10:
            break
        adj = _ga_search(target_n, time_budget=budget)
        if adj is not None:
            edges = _edges_from_adj(adj, target_n)
            return {'n': target_n, 'edges': edges}
    
    # Fallback
    for target_n in [18, 17, 16, 15]:
        if _elapsed() > DEADLINE - 5:
            break
        for seed in range(50000):
            if _elapsed() > DEADLINE - 5:
                break
            rng = random.Random(seed)
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if not _has_violation_full(target_n, adj):
                edges = _edges_from_adj(adj, target_n)
                return {'n': target_n, 'edges': edges}
    
    return {'n': 5, 'edges': []}
