"""Combined approach: degree-sequence guided construction with bitmask validity."""
import time
import random
import itertools

N = 15
DEADLINE = 55.0  # Safety margin under the 90s limit

def _adjacency(edges):
    adj = [0] * N
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _degrees(adj):
    return [bin(adj[i]).count('1') for i in range(N)]

def _is_valid(adj, deg):
    # For every edge (u, v) with u < v, check all simple 3-edge paths s-u-v-t
    for u in range(N):
        au = adj[u]
        for v in range(u + 1, N):
            if not (au >> v) & 1:
                continue
            av = adj[v]
            # s is neighbor of u, s != v
            for s in range(N):
                if s == v or s == u:
                    continue
                if not (au >> s) & 1:
                    continue
                ds = deg[s]
                # t is neighbor of v, t != u, t != s
                for t in range(N):
                    if t == u or t == s:
                        continue
                    if not (av >> t) & 1:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def _valid(edges):
    adj = _adjacency(edges)
    deg = _degrees(adj)
    return _is_valid(adj, deg)

def _greedy_add(order, start_edges):
    edges = list(start_edges)
    present = set()
    for u, v in edges:
        present.add((min(u, v), max(u, v)))
    for (u, v) in order:
        key = (min(u, v), max(u, v))
        if key in present:
            continue
        trial = edges + [list(key)]
        if _valid(trial):
            edges = trial
            present.add(key)
    return edges

def _erdos_gallai(degs):
    """Check if degree sequence is graphical using Erdős–Gallai theorem."""
    n = len(degs)
    if sum(degs) % 2 != 0:
        return False
    if any(d < 0 or d > n - 1 for d in degs):
        return False
    s = sorted(degs, reverse=True)
    total = sum(s)
    for k in range(1, n + 1):
        left = sum(s[:k])
        right = k * (k - 1) + sum(min(d, k) for d in s[k:])
        if left > right:
            return False
    return True

def _try_degree_partition(target_degs, rng, t0):
    """Try to construct a graph with the given degree sequence that is valid."""
    # Generate candidate edges ordered by how they help satisfy degrees
    # Use a randomized greedy approach
    best = []
    best_len = 0
    
    for attempt in range(50):
        if time.time() - t0 > DEADLINE:
            break
        
        # Shuffle edge order
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        rng.shuffle(all_edges)
        
        # Greedy: add edge if it keeps the partial graph "degree-compatible"
        # We track current degrees and try to approach target_degs
        cur_degs = [0] * N
        edges = []
        adj = [0] * N
        
        for u, v in all_edges:
            if time.time() - t0 > DEADLINE:
                break
            # Check if adding this edge keeps us on track
            # Simple heuristic: prefer edges between vertices with lower current degree
            # relative to target
            if cur_degs[u] >= target_degs[u] or cur_degs[v] >= target_degs[v]:
                continue
            
            # Try adding
            trial_adj = adj[:]
            trial_adj[u] |= (1 << v)
            trial_adj[v] |= (1 << u)
            trial_degs = cur_degs[:]
            trial_degs[u] += 1
            trial_degs[v] += 1
            
            if _is_valid(trial_adj, trial_degs):
                edges.append([u, v])
                adj = trial_adj
                cur_degs = trial_degs
        
        if len(edges) > best_len:
            best_len = len(edges)
            best = edges
    
    return best, best_len

def construct():
    t0 = time.time()
    rng = random.Random(20240607)
    
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Strong baseline: K_{7,8} has 56 edges and is valid
    k78 = [[a, b] for a in range(7) for b in range(7, 15)]
    best = k78
    best_len = len(k78)
    
    def consider(cand):
        nonlocal best, best_len
        if len(cand) > best_len:
            best_len = len(cand)
            best = cand
    
    # Phase 1: Greedy from K_{7,8} with multiple random orders
    for _ in range(20):
        if time.time() - t0 > DEADLINE:
            break
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy_add(order, k78)
        consider(cand)
    
    # Phase 2: Try different degree sequences using Erdős–Gallai
    # We want high-degree vertices but with enough variation to avoid equal-degree path endpoints
    # Try various target degree sequences
    base_degs = [8] * 7 + [7] * 8  # K_{7,8} degrees
    
    # Try to find better degree sequences
    candidate_degs = []
    
    # Generate random graphical degree sequences with high total
    for _ in range(200):
        if time.time() - t0 > DEADLINE:
            break
        # Start with high degrees, then perturb
        degs = [8] * 7 + [7] * 8
        # Randomly increase some degrees
        for _ in range(5):
            i = rng.randint(0, N - 1)
            if degs[i] < N - 1:
                degs[i] += 1
        # Randomly decrease some
        for _ in range(3):
            i = rng.randint(0, N - 1)
            if degs[i] > 1:
                degs[i] -= 1
        
        if _erdos_gallai(degs):
            candidate_degs.append(degs)
    
    # Try constructing graphs for promising degree sequences
    for degs in candidate_degs[:50]:
        if time.time() - t0 > DEADLINE:
            break
        # Only try if the sum is higher than current best
        if sum(degs) // 2 <= best_len:
            continue
        cand, cl = _try_degree_partition(degs, rng, t0)
        consider(cand)
    
    # Phase 3: Local search - remove an edge, re-greedy
    for _ in range(30):
        if time.time() - t0 > DEADLINE:
            break
        base = best[:]
        if not base:
            continue
        idx = rng.randint(0, len(base) - 1)
        reduced = base[:idx] + base[idx + 1:]
        
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy_add(order, reduced)
        consider(cand)
        
        if time.time() - t0 < DEADLINE:
            order2 = all_edges[:]
            rng.shuffle(order2)
            cand2 = _greedy_add(order2, cand)
            consider(cand2)
    
    # Phase 4: Random restarts with repair from dense random graphs
    for _ in range(30):
        if time.time() - t0 > DEADLINE:
            break
        
        adj = [0] * N
        edges = []
        for u in range(N):
            for v in range(u + 1, N):
                if rng.random() < 0.85:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edges.append([u, v])
        
        # Repair: remove edges that cause violations
        deg = _degrees(adj)
        for _ in range(100):
            if time.time() - t0 > DEADLINE:
                break
            if _is_valid(adj, deg):
                break
            # Find a violating edge and remove it
            found = False
            for u in range(N):
                if found:
                    break
                au = adj[u]
                for v in range(u + 1, N):
                    if not (au >> v) & 1:
                        continue
                    av = adj[v]
                    for s in range(N):
                        if s == u or s == v or not (au >> s) & 1:
                            continue
                        for t in range(N):
                            if t == u or t == s or not (av >> t) & 1:
                                continue
                            if deg[s] == deg[t]:
                                # Remove edge (u, v)
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                                deg[u] -= 1
                                deg[v] -= 1
                                edges = [[u2, v2] for u2 in range(N) for v2 in range(u2 + 1, N) if (adj[u2] >> v2) & 1]
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        
        consider(edges)
    
    # Ensure return type is correct
    result = [[int(u), int(v)] for u, v in best]
    return result
