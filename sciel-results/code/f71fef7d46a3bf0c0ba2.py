"""Solve dense odd-order graphs avoiding equal-degree path endpoints.

Validity: no path s-u-v-t (s!=t) has deg(s)==deg(t).
Equivalent: for every vertex u, all neighbors of u have distinct degrees.
Strategy: simulated annealing on edge set + greedy completion.
"""
import time
import random

N = 15
DEADLINE = 85.0  # safety margin under 90s kill


def _is_valid(adj, n):
    """Check: for every u, neighbors have distinct degrees."""
    deg = [len(adj[i]) for i in range(n)]
    for u in range(n):
        seen = set()
        for v in adj[u]:
            d = deg[v]
            if d in seen:
                return False
            seen.add(d)
    return True


def _violation_count(adj, n):
    """Count number of (u, pair-of-equal-degree-neighbors) violations."""
    deg = [len(adj[i]) for i in range(n)]
    count = 0
    for u in range(n):
        seen = set()
        for v in adj[u]:
            d = deg[v]
            if d in seen:
                count += 1
            else:
                seen.add(d)
    return count


def _edge_count(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2


def _random_adj(n, seed, p=0.5):
    """Generate a random graph with edge probability p."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i].add(j)
                adj[j].add(i)
    return adj


def _simulated_annealing(adj, n, seed, max_iter, t_start, t_end):
    """Simulated annealing to minimize violations while maximizing edges."""
    rng = random.Random(seed)
    adj = [set(s) for s in adj]
    current_viol = _violation_count(adj, n)
    current_edges = _edge_count(adj, n)
    best_adj = [set(s) for s in adj]
    best_score = current_edges * 1000 - current_viol  # maximize edges, minimize violations
    
    # Temperature schedule
    for it in range(max_iter):
        t = t_start * (t_end / t_start) ** (it / max_iter)
        
        # Generate a move: add or remove a random edge
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        if j in adj[i]:
            # Remove edge
            adj[i].discard(j)
            adj[j].discard(i)
            new_viol = _violation_count(adj, n)
            new_edges = _edge_count(adj, n)
        else:
            # Add edge
            adj[i].add(j)
            adj[j].add(i)
            new_viol = _violation_count(adj, n)
            new_edges = _edge_count(adj, n)
        
        new_score = new_edges * 1000 - new_viol
        delta = new_score - (current_edges * 1000 - current_viol)
        
        if delta >= 0 or rng.random() < _exp_neg_delta(delta, t):
            # Accept
            current_viol = new_viol
            current_edges = new_edges
            if new_score > best_score:
                best_score = new_score
                best_adj = [set(s) for s in adj]
        else:
            # Reject: revert
            if j in adj[i]:
                adj[i].discard(j)
                adj[j].discard(i)
            else:
                adj[i].add(j)
                adj[j].add(i)
    
    return best_adj


def _exp_neg_delta(delta, t):
    """Compute exp(-delta/t) safely for delta <= 0."""
    if delta >= 0:
        return 1.0
    if t <= 0:
        return 0.0
    x = -delta / t
    if x > 50:
        return 0.0
    import math
    return math.exp(-x)


def _greedy_complete(adj, n, max_rounds=50):
    """Try to add any missing edge that keeps the graph valid."""
    adj = [set(s) for s in adj]
    improved = True
    rounds = 0
    while improved and rounds < max_rounds:
        improved = False
        rounds += 1
        for i in range(n):
            for j in range(i + 1, n):
                if j not in adj[i]:
                    adj[i].add(j)
                    adj[j].add(i)
                    if _is_valid(adj, n):
                        improved = True
                    else:
                        adj[i].discard(j)
                        adj[j].discard(i)
    return adj


def _local_search_swap(adj, n, seed, max_iter=2000):
    """Local search: try edge swaps (remove one, add another) to increase edge count while staying valid."""
    rng = random.Random(seed)
    adj = [set(s) for s in adj]
    current_edges = _edge_count(adj, n)
    best_adj = [set(s) for s in adj]
    best_edges = current_edges
    
    for _ in range(max_iter):
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        # Try swapping: remove edge (a,b) and add edge (i,j) if i,j not adjacent
        if j in adj[i]:
            continue
        # Find an edge to remove
        candidates = []
        for a in range(n):
            for b in adj[a]:
                if a < b:
                    candidates.append((a, b))
        if not candidates:
            continue
        a, b = rng.choice(candidates)
        
        # Remove (a,b), add (i,j)
        adj[a].discard(b)
        adj[b].discard(a)
        adj[i].add(j)
        adj[j].add(i)
        
        if _is_valid(adj, n):
            new_edges = _edge_count(adj, n)
            if new_edges > best_edges:
                best_edges = new_edges
                best_adj = [set(s) for s in adj]
        else:
            # Revert
            adj[a].add(b)
            adj[b].add(a)
            adj[i].discard(j)
            adj[j].discard(i)
    
    return best_adj


def construct():
    start = time.time()
    
    # Baseline: star graph (14 edges, always valid)
    star = [set() for _ in range(N)]
    for v in range(1, N):
        star[0].add(v)
        star[v].add(0)
    best_adj = star
    best_count = 14
    
    # Phase 1: Simulated annealing from various random starts
    sa_configs = [
        (0.5, 5000, 1.0, 0.01),
        (0.6, 5000, 1.0, 0.01),
        (0.4, 5000, 1.0, 0.01),
        (0.7, 5000, 1.0, 0.01),
        (0.5, 8000, 2.0, 0.005),
        (0.3, 8000, 1.5, 0.005),
    ]
    
    sa_seed = 42
    for p, max_iter, t_start, t_end in sa_configs:
        if time.time() - start > DEADLINE - 30:
            break
        for trial in range(3):
            if time.time() - start > DEADLINE - 30:
                break
            init = _random_adj(N, sa_seed, p)
            sa_seed += 1
            result = _simulated_annealing(init, N, sa_seed, max_iter, t_start, t_end)
            sa_seed += 1
            
            # Greedy complete
            result = _greedy_complete(result, N)
            
            cnt = _edge_count(result, N)
            if cnt > best_count and _is_valid(result, N):
                best_count = cnt
                best_adj = result
    
    # Phase 2: Local search with swaps from best
    if time.time() - start < DEADLINE - 15:
        for seed in range(5):
            if time.time() - start > DEADLINE - 15:
                break
            ls_result = _local_search_swap(best_adj, N, 1000 + seed, max_iter=3000)
            ls_result = _greedy_complete(ls_result, N)
            cnt = _edge_count(ls_result, N)
            if cnt > best_count and _is_valid(ls_result, N):
                best_count = cnt
                best_adj = ls_result
    
    # Phase 3: More SA restarts from the best solution perturbed
    if time.time() - start < DEADLINE - 10:
        for trial in range(10):
            if time.time() - start > DEADLINE - 10:
                break
            # Perturb best: randomly remove 1-3 edges
            perturbed = [set(s) for s in best_adj]
            rng = random.Random(2000 + trial)
            for _ in range(rng.randint(1, 3)):
                all_edges = []
                for i in range(N):
                    for j in perturbed[i]:
                        if i < j:
                            all_edges.append((i, j))
                if all_edges:
                    a, b = rng.choice(all_edges)
                    perturbed[a].discard(b)
                    perturbed[b].discard(a)
            
            result = _simulated_annealing(perturbed, N, 3000 + trial, 3000, 1.0, 0.01)
            result = _greedy_complete(result, N)
            
            cnt = _edge_count(result, N)
            if cnt > best_count and _is_valid(result, N):
                best_count = cnt
                best_adj = result
    
    # Final greedy completion
    best_adj = _greedy_complete(best_adj, N)
    final_count = _edge_count(best_adj, N)
    if final_count > best_count and _is_valid(best_adj, N):
        best_count = final_count
    
    # Convert to list of [u,v] with proper types
    edges = []
    for i in range(N):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
    
    # Final validity check
    adj_final = [set() for _ in range(N)]
    for e in edges:
        adj_final[e[0]].add(e[1])
        adj_final[e[1]].add(e[0])
    if not _is_valid(adj_final, N):
        # Fallback to star
        edges = [[0, v] for v in range(1, N)]
    
    return edges
