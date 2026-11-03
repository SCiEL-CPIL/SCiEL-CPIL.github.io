"""Portfolio SEARCH: try multiple explicit families, keep best (smallest) valid graph."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    deadline = time.time() + 115.0  # safety margin before 150s kill
    rng = random.Random(12345)

    def check_deadline():
        return time.time() > deadline

    # --- Utility: validate graph (simple, k-regular, girth >= g) ---
    def validate(n, edges, k, g):
        if n <= 0:
            return False
        adj = [[] for _ in range(n)]
        seen = set()
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return False
            key = (min(u,v), max(u,v))
            if key in seen:
                return False
            seen.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for u in range(n):
            if len(adj[u]) != k:
                return False
        
        # Optimized Girth Check
        for s in range(n):
            dist = {}
            q = deque()
            for nb in adj[s]:
                dist[(nb, s)] = 1
                q.append((nb, s))
            
            while q:
                u, prev = q.popleft()
                d = dist[(u, prev)]
                if d >= g:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    nd = d + 1
                    if v == s:
                        if nd < g:
                            return False
                        continue
                    key = (v, u)
                    if key not in dist or dist[key] > nd:
                        dist[key] = nd
                        q.append((v, u))
        return True

    def try_graph(n, edges, k, g):
        if not validate(n, edges, k, g):
            return None
        return n, edges

    # --- Family 1: Lazebnik-Ustimenko-Woldar (Fallback) ---
    def luw_family(q=None, d=None):
        if q is None:
            q = k
        if d is None:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q
            sub = lambda x,y: (x-y)%q
            mul = lambda x,y: x*y%q
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        
        start = (0, (None,)+(0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # --- Family 2: Random Regular Graph with Girth Repair ---
    def generate_random_regular(n, k, rng):
        """Generate a random k-regular graph using configuration model + 2-switches."""
        if n * k % 2 != 0:
            return None
        if k >= n:
            return None
        for attempt in range(50):
            half_edges = []
            for v in range(n):
                half_edges.extend([v] * k)
            rng.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if not valid:
                continue
            has_multi = False
            for u in range(n):
                if len(adj[u]) != len(set(adj[u])):
                    has_multi = True
                    break
            if has_multi:
                continue
            # Randomize with 2-switches
            for _ in range(n * 2):
                a = rng.randint(0, n-1)
                b = rng.choice(adj[a])
                c = rng.randint(0, n-1)
                if c == a or c == b:
                    continue
                d = rng.choice(adj[c])
                if d == a or d == b:
                    continue
                if b in adj[c] or d in adj[a]:
                    continue
                # Switch (a,b) and (c,d) -> (a,d) and (c,b)
                adj[a].remove(b)
                adj[b].remove(a)
                adj[c].remove(d)
                adj[d].remove(c)
                
                adj[a].append(d)
                adj[d].append(a)
                adj[c].append(b)
                adj[b].append(c)
            
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
        return None

    def find_short_cycle(adj, g):
        """Find a cycle of length < g. Returns list of nodes in cycle or None."""
        n = len(adj)
        for s in range(n):
            # BFS for non-backtracking walks
            dist = {}
            q = deque()
            for nb in adj[s]:
                dist[(nb, s)] = [s, nb]
                q.append((nb, s))
            
            while q:
                u, prev = q.popleft()
                path = dist[(u, prev)]
                d = len(path)
                if d >= g:
                    continue
                for v in adj[u]:
                    if v == prev:
                        continue
                    if v == s:
                        if d + 1 < g:
                            return path + [s]
                        continue
                    key = (v, u)
                    if key not in dist:
                        dist[key] = path + [v]
                        q.append((v, u))
        return None

    def repair_girth(n, edges, k, g, max_iters=1000):
        """Try to remove short cycles using 2-switches with adaptive strategy."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Track best state (fewest short cycles)
        best_adj = [list(x) for x in adj]
        best_cycle_count = float('inf')
        stuck_rounds = 0
        
        for it in range(max_iters):
            if check_deadline():
                break
            
            cycle = find_short_cycle(adj, g)
            if cycle is None:
                break
            
            L = len(cycle)
            cycle_nodes = cycle[:-1]
            found_switch = False
            
            # Try all pairs of non-adjacent edges in the cycle
            for i in range(L):
                for j in range(i+2, L):
                    if j == i+1 or (j+1)%L == i:
                        continue
                    
                    u1 = cycle_nodes[i]
                    v1 = cycle_nodes[(i+1)%L]
                    u2 = cycle_nodes[j]
                    v2 = cycle_nodes[(j+1)%L]
                    
                    # New edges: (u1, v2) and (u2, v1)
                    if v2 in adj[u1] or v1 in adj[u2]:
                        continue
                    
                    # Perform switch
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    
                    adj[u1].append(v2)
                    adj[v2].append(u1)
                    adj[u2].append(v1)
                    adj[v1].append(u2)
                    
                    found_switch = True
                    break
                if found_switch:
                    break
            
            if not found_switch:
                stuck_rounds += 1
                # More aggressive random restarts when stuck
                n_random = min(50 + stuck_rounds * 10, 200)
                for _ in range(n_random):
                    if check_deadline():
                        break
                    a = rng.randint(0, n-1)
                    b = rng.choice(adj[a])
                    c = rng.randint(0, n-1)
                    if c == a or c == b: continue
                    d = rng.choice(adj[c])
                    if d == a or d == b: continue
                    if b in adj[c] or d in adj[a]: continue
                    adj[a].remove(b); adj[b].remove(a)
                    adj[c].remove(d); adj[d].remove(c)
                    adj[a].append(d); adj[d].append(a)
                    adj[c].append(b); adj[b].append(c)
            else:
                stuck_rounds = 0
        
        # Final check
        if find_short_cycle(adj, g) is None:
            new_edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        new_edges.append((u, v))
            return new_edges
        return None

    best = None
    best_n = float('inf')

    def consider(n, edges):
        nonlocal best, best_n
        if n < best_n:
            result = try_graph(n, edges, k, g)
            if result is not None:
                best = result
                best_n = n
        if check_deadline():
            raise TimeoutError

    # --- Run Search ---
    
    # 1. Calculate Moore Bound
    if g % 2 == 1:
        r = (g-1)//2
        moore = 1 + k * sum((k-1)**i for i in range(r))
    else:
        r = g//2
        moore = k * sum((k-1)**i for i in range(r))
        
    # Fallback LUW first
    try:
        n_luw, e_luw = luw_family()
        consider(n_luw, e_luw)
    except Exception:
        pass

    # Search for better - focus on promising n ranges
    # For k=5, g=9: known best is 1116, lower bound 428
    # Try n values that are more likely to succeed
    max_n_to_try = 1200
    
    # Generate candidate n values: start from reasonable range
    # Skip very small n (near Moore bound) as they're nearly impossible
    # Focus on n where random regular graphs can be repaired
    # Prioritize sizes that are multiples of small numbers (better for regular graphs)
    candidate_ns = []
    min_n = max(moore, 450)
    
    # First pass: try promising sizes (multiples of 10, 20, 50)
    for n in range(min_n, max_n_to_try + 1):
        if n * k % 2 == 0 and k < n:
            # Score by "niceness" - multiples of larger numbers first
            score = 0
            for div in [100, 50, 20, 10, 5]:
                if n % div == 0:
                    score += div
            candidate_ns.append((-score, n))
    
    # Sort by score (descending) then by n (ascending)
    candidate_ns.sort()
    
    # Try candidates in priority order
    for _, n in candidate_ns:
        if check_deadline():
            break
        
        # Adaptive restarts: more for smaller n (harder), fewer for larger
        if n < 700:
            n_restarts = 8
        elif n < 900:
            n_restarts = 5
        else:
            n_restarts = 3
        
        for _restart in range(n_restarts):
            if check_deadline():
                break
            res = generate_random_regular(n, k, rng)
            if res is None:
                continue
            n_gen, edges_gen = res
            # Repair with adaptive iteration count
            max_iter = 800 if n < 700 else 500
            repaired = repair_girth(n_gen, edges_gen, k, g, max_iters=max_iter)
            if repaired is not None:
                consider(n_gen, repaired)
                if best is not None and best_n == n_gen:
                    break
            # If we found a valid one, move to next n

    if best is not None:
        return best

    # Fallback: return LUW
    n, e = luw_family()
    return n, e

# EVOLVE-BLOCK-END
