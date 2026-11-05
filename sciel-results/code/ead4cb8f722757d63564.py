import time
import math
import random
import numpy as np
from collections import deque

_START_TIME = time.time()
_TIMEOUT = 145  # wall-clock budget (seconds)

def _deadline_exceeded():
    return (time.time() - _START_TIME) > _TIMEOUT

def _verify_girth(edges, n, g):
    """Check no cycle shorter than g using non-backtracking walks from every vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # visited states: (prev, curr)
        dist = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb, 1))
            dist[(start, nb)] = 1
        while queue:
            prev, curr, d = queue.popleft()
            if d >= g:
                continue
            for nb2 in adj[curr]:
                if nb2 == prev:
                    continue
                state = (curr, nb2)
                if state in dist:
                    if d + 1 < g:
                        return False
                else:
                    dist[state] = d + 1
                    queue.append((curr, nb2, d + 1))
    return True

def _check_simple_and_regular(edges, n, k):
    if len(edges) != n * k // 2:
        return False
    seen = set()
    deg = [0] * n
    for u, v in edges:
        if u == v:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            return False
    return True

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def construct_lw(k, g):
    """Lazebnik-Ustimenko-Woldar CD(d,q) construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct_abs(k, g):
    """Alon-Babai-Suzuki construction on Heisenberg group H_q. Requires even k."""
    if k % 2 != 0:
        return None

    if g % 2 == 0:
        mb = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        mb = 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))

    q = 2
    while True:
        if _is_prime(q) and q ** 3 >= mb:
            break
        q += 1

    if k == 4:
        gens = [
            (1, 0, 0), (q - 1, 0, 0),
            (0, 1, 0), (0, q - 1, 1)
        ]
    elif k == 6:
        gens = [
            (1, 0, 0), (q - 1, 0, 0),
            (0, 1, 0), (0, q - 1, 1),
            (0, 0, 1), (0, 0, q - 1)
        ]
    else:
        return None

    n = q ** 3
    edges = []

    def idx(x, y, z):
        return (x * q + y) * q + z

    for x in range(q):
        for y in range(q):
            for z in range(q):
                u = idx(x, y, z)
                for gx, gy, gz in gens:
                    vx = (x + gx) % q
                    vy = (y + gy) % q
                    vz = (z + gz + x * gy) % q
                    v = idx(vx, vy, vz)
                    if u < v:
                        edges.append((u, v))

    return n, edges

def _generate_random_k_regular(n, k):
    """Generate a random k-regular graph using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    edges = set()
    adj = [[] for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            # Self-loop, retry or swap
            # Simple retry: find another stub
            found = False
            for j in range(i+2, len(stubs)):
                if stubs[j] != u:
                    v = stubs[j]
                    stubs[j] = stubs[i+1]
                    found = True
                    break
            if not found:
                return None
        if u < v:
            if (u, v) in edges:
                return None # Multi-edge
            edges.add((u, v))
            adj[u].append(v)
            adj[v].append(u)
        else:
            if (v, u) in edges:
                return None
            edges.add((v, u))
            adj[v].append(u)
            adj[u].append(v)
            
    return list(edges)

def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g. Returns list of vertices or None."""
    # BFS from a random node
    start = random.randint(0, n-1)
    dist = {start: 0}
    parent = {start: None}
    queue = deque([start])
    
    while queue:
        u = queue.popleft()
        if dist[u] >= g - 1:
            continue
        for v in adj[u]:
            if v == parent.get(u, -1):
                continue
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
            else:
                # Found a cycle
                # Cycle length is dist[u] + dist[v] + 1? 
                # Actually, if we hit a visited node, we need to trace back.
                # This simple BFS finds *a* cycle, but we need to ensure it's < g.
                # If dist[u] + 1 + dist[v] < g, we have a short cycle.
                # But BFS tree edges don't necessarily form the shortest cycle.
                # For simplicity, let's just return a path if we detect a back-edge to an ancestor.
                # This heuristic is weak. 
                # Better: just check if girth is < g using the full verifier on a sample?
                # No, let's use a specific "girth finder" that returns the cycle.
                pass
    return None

def _optimize_girth(edges, n, k, g, time_limit=5.0):
    """
    Try to increase the girth of a given k-regular graph using 2-swaps.
    Returns (edges, success).
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute adjacency sets for fast lookup
    adj_set = [set(adj[i]) for i in range(n)]
    
    t_start = time.time()
    current_edges = list(edges)
    edge_set = set(current_edges)
    
    # Helper to get girth (approximate or exact)
    # For speed, we'll just check if girth >= g using _verify_girth periodically
    # Or we can count short cycles.
    
    # Let's just do random 2-swaps and check girth every N steps.
    # A 2-swap: remove (u,v) and (x,y), add (u,x) and (v,y) or (u,y) and (v,x).
    
    steps = 0
    while time.time() - t_start < time_limit:
        if _deadline_exceeded():
            break
            
        # Pick two random edges
        e1_idx = random.randint(0, len(current_edges)-1)
        e2_idx = random.randint(0, len(current_edges)-1)
        if e1_idx == e2_idx:
            continue
            
        u, v = current_edges[e1_idx]
        x, y = current_edges[e2_idx]
        
        # Ensure u,v,x,y are distinct
        if len({u, v, x, y}) < 4:
            continue
            
        # Try swap 1: (u,x) and (v,y)
        # Check if edges already exist
        if (min(u,x), max(u,x)) not in edge_set and (min(v,y), max(v,y)) not in edge_set:
            # Perform swap
            edge_set.discard((min(u,v), max(u,v)))
            edge_set.discard((min(x,y), max(x,y)))
            edge_set.add((min(u,x), max(u,x)))
            edge_set.add((min(v,y), max(v,y)))
            
            # Update adj
            adj[u].remove(v); adj[v].remove(u)
            adj[x].remove(y); adj[y].remove(x)
            
            adj[u].append(x); adj[x].append(u)
            adj[v].append(y); adj[y].append(v)
            adj_set[u].discard(v); adj_set[v].discard(u)
            adj_set[x].discard(y); adj_set[y].discard(x)
            adj_set[u].add(x); adj_set[x].add(u)
            adj_set[v].add(y); adj_set[y].add(v)
            
            current_edges[e1_idx] = (min(u,x), max(u,x))
            current_edges[e2_idx] = (min(v,y), max(v,y))
            
            steps += 1
            if steps % 100 == 0:
                if _verify_girth(current_edges, n, g):
                    return current_edges, True

    return current_edges, False

def construct(k, g):
    """Main entry point: construct k-regular graph with girth >= g, minimize n."""
    t0 = time.time()
    
    # 1. Get baseline from LW
    n_lw, edges_lw = construct_lw(k, g)
    best_n = n_lw
    best_edges = edges_lw
    
    # 2. Calculate Moore Bound
    if g % 2 == 0:
        mb = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        mb = 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))
        
    # 3. Try to find a smaller n
    # We scan n from mb to best_n - 1
    # For each n, we try to construct a valid graph.
    # Strategy: Random init + Local Search to fix girth.
    
    # To be efficient, we'll try a few candidate n values.
    # If k=5, g=9, mb=426, n_lw=1250.
    # We can't try all 426..1249.
    # We'll try a subset: e.g., mb, mb+50, mb+100, ... up to best_n.
    # Or just try to optimize the LW graph? No, LW is fixed size.
    
    # Let's try to construct graphs of size n for a few candidate n.
    # We prioritize smaller n.
    
    # Determine step size based on time budget
    # If we have 100s left, and we want to try 10 candidates, 10s each.
    
    candidates = []
    # Add some candidates between mb and best_n
    if best_n > mb:
        # Try a logarithmic or linear spread
        num_candidates = 5
        for i in range(num_candidates):
            # Exponential spacing towards mb? Or linear?
            # Let's try values close to mb first, as that's the target.
            # But random graphs at mb are very likely to have short cycles.
            # Maybe start a bit higher?
            # Let's just pick a few: mb, mb + (best_n-mb)//2, etc.
            pass
            
    # Actually, a better heuristic:
    # Start with the LW graph.
    # It is valid.
    # Can we reduce n?
    # No, we can't easily reduce n with local moves.
    
    # So we must construct from scratch for smaller n.
    
    # Let's try n = mb + 10, mb + 20, ...
    # But generating random k-regular graphs is fast.
    # The bottleneck is the girth optimization.
    
    # We will try to construct a graph of size n_target.
    # If we succeed, we return it.
    
    # To maximize chance of success, we should pick n_target such that
    # the graph is "dense" enough to allow girth g.
    
    # Let's try a few specific n values.
    # For k=5, g=9, known best is 1116.
    # Let's try to beat 1250.
    
    # We'll iterate n from max(mb, 400) to best_n - 1.
    # But we can't iterate all.
    
    # Strategy:
    # 1. Try n = best_n - 10, best_n - 50, best_n - 100...
    # 2. For each n, run a short optimization.
    
    # Actually, let's just try to optimize the LW graph's girth?
    # No, it already has girth >= 9.
    
    # Let's try to construct a graph of size n = 1000.
    # If we can get a valid 1000-vertex graph, we win.
    
    # We'll pick a few target sizes.
    targets = []
    if best_n > 500:
        # Try to halve the size?
        # Try a range.
        step = max(10, (best_n - mb) // 10)
        for n_try in range(max(mb, 100), best_n, step):
            targets.append(n_try)
            
    # Shuffle targets? No, start from small to large?
    # Small n is harder. Large n is easier to satisfy girth.
    # We want the *smallest* valid n.
    # So we should try small n first, but give them less time?
    # Or try large n first to ensure we have *some* improvement over LW?
    # If we find a valid graph at n=1100, that's better than 1250.
    # If we find one at n=1000, even better.
    
    # Let's try from best_n downwards.
    # If we find a valid graph at n, we can stop?
    # No, we want the *minimum* n.
    # But finding the *absolute* minimum is hard.
    # Finding *any* n < best_n is an improvement.
    
    # Let's try a few n values below best_n.
    # We'll allocate time to each.
    
    remaining_time = _TIMEOUT - (time.time() - t0)
    if remaining_time < 10:
        return best_n, best_edges
        
    # Pick 3 candidate sizes: best_n - 10%, best_n - 20%, best_n - 30%
    # And maybe one near mb.
    
    candidates = []
    for frac in [0.1, 0.2, 0.3, 0.5]:
        n_cand = int(best_n * (1 - frac))
        if n_cand > mb:
            candidates.append(n_cand)
            
    # Add mb + 10
    candidates.append(mb + 10)
    
    # Sort candidates
    candidates = sorted(set(candidates))
    
    # Time per candidate
    time_per_cand = remaining_time / len(candidates)
    
    for n_cand in candidates:
        if _deadline_exceeded():
            break
            
        # Try to construct a graph of size n_cand
        # 1. Generate random k-regular graph
        # 2. Optimize girth
        
        # Retry loop for generation
        edges_cand = None
        for _ in range(10):
            edges_cand = _generate_random_k_regular(n_cand, k)
            if edges_cand:
                break
                
        if not edges_cand:
            continue
            
        # Check if it's already valid (unlikely for small n)
        if _verify_girth(edges_cand, n_cand, g):
            best_n = n_cand
            best_edges = edges_cand
            break # Found a very good one, stop? Or keep looking for smaller?
            # Since we are scanning arbitrary candidates, we should keep looking for smaller.
            # But we are scanning a specific list.
            
        # Optimize
        time_left_for_opt = time_per_cand - (time.time() - t0 - (time.time() - t0)) # just use time_per_cand
        # Actually, just pass a time limit
        edges_opt, success = _optimize_girth(edges_cand, n_cand, k, g, time_limit=max(1.0, time_per_cand/2))
        
        if success:
            best_n = n_cand
            best_edges = edges_opt
            # Don't break, try to find smaller n?
            # But our candidates are fixed.
            # If we want to be greedy, we could try smaller n next.
            # But our list is sorted ascending. So we already tried smaller ones.
            # Wait, `candidates` is sorted ascending.
            # So if we succeed at a large n, we might have failed at smaller ones.
            # We should continue to check if we can do even smaller?
            # No, we already tried smaller ones in the loop (since sorted).
            # So if we succeed at n_cand, it's the best so far among tried.
            # But we might have skipped some.
            # Let's just update best_n if n_cand < best_n.
            
    return best_n, best_edges
