import time
from collections import deque
import random

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

N = P['n']
BASE = set()
for e in P['edges']:
    u, v = e
    BASE.add((min(u,v), max(u,v)))

def build_adj(added):
    adj = [set() for _ in range(N)]
    for (u,v) in BASE:
        adj[u].add(v); adj[v].add(u)
    for (u,v) in added:
        adj[u].add(v); adj[v].add(u)
    return adj

def is_triangle_free(adj):
    for u in range(N):
        for v in adj[u]:
            if u < v:
                for w in adj[u]:
                    if w != v and w in adj[v]:
                        return False
    return True

def all_pairs_dist(adj):
    """Return dict mapping (s,v) with s<v to distance, or -1 if unreachable."""
    dists = {}
    for s in range(N):
        dist = [-1]*N
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u]+1
                    dq.append(v)
        for v in range(s+1, N):
            dists[(s,v)] = dist[v]
    return dists

def bad_pairs_count(adj):
    """Count pairs with distance > 4 or unreachable."""
    cnt = 0
    for s in range(N):
        dist = [-1]*N
        dist[s] = 0
        dq = deque([s])
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u]+1
                    dq.append(v)
        for v in range(s+1, N):
            if dist[v] > 4:
                cnt += 1
    return cnt

def is_valid_solution(added):
    adj = build_adj(added)
    # Check connected
    seen = [False]*N
    dq = deque([0])
    seen[0] = True
    cnt = 1
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                cnt += 1
                dq.append(v)
    if cnt != N:
        return False
    if not is_triangle_free(adj):
        return False
    return bad_pairs_count(adj) == 0

def same_parity_candidates():
    """Edges connecting vertices of the same parity - safe from immediate triangles in path graph."""
    cands = []
    for u in range(N):
        for v in range(u+1, N):
            if (u,v) not in BASE and u % 2 == v % 2:
                cands.append((u,v))
    return cands

def greedy_set_cover(candidates, initial_added):
    """Greedy set-cover: pick edge that reduces bad-pair count most."""
    added = list(initial_added)
    added_set = set(added)
    
    if is_valid_solution(added):
        return added
    
    while time.time() < deadline_global:
        bad_cnt = bad_pairs_count(build_adj(added))
        if bad_cnt == 0:
            break
        
        best_edge = None
        best_bad = bad_cnt
        
        for e in candidates:
            if e in added_set:
                continue
            trial = added + [e]
            adj = build_adj(trial)
            # Quick triangle check
            u, v = e
            if not is_triangle_free(adj):
                continue
            b = bad_pairs_count(adj)
            if b < best_bad:
                best_bad = b
                best_edge = e
                if best_bad == 0:
                    break
        
        if best_edge is None:
            break
        added.append(best_edge)
        added_set.add(best_edge)
    
    return added

def local_search_removal(cur):
    """Try removing each edge; keep if still valid."""
    improved = True
    while improved and time.time() < deadline_global:
        improved = False
        for i in range(len(cur)):
            if time.time() >= deadline_global:
                break
            trial = cur[:i] + cur[i+1:]
            if is_valid_solution(trial):
                cur = trial
                improved = True
                break
    return cur

def local_search_swap(cur, candidates):
    """Try replacing one edge with another to maintain validity."""
    cur_set = set(cur)
    improved = True
    while improved and time.time() < deadline_global:
        improved = False
        for i in range(len(cur)):
            if time.time() >= deadline_global:
                break
            removed = cur[i]
            trial_base = cur[:i] + cur[i+1:]
            for e in candidates:
                if e in cur_set or e == removed:
                    continue
                trial = trial_base + [e]
                if is_valid_solution(trial):
                    cur = trial
                    cur_set = set(cur)
                    improved = True
                    break
            if improved:
                break
    return cur

def solve():
    global deadline_global
    deadline_global = time.time() + 75
    
    cands = same_parity_candidates()
    best_solution = None
    best_count = float('inf')
    
    # Phase 1: Greedy set-cover
    sol = greedy_set_cover(cands, [])
    if is_valid_solution(sol):
        best_solution = local_search_removal(sol)
        best_count = len(best_solution)
    
    # Phase 2: Randomized restarts with different orderings
    for restart in range(100):
        if time.time() >= deadline_global:
            break
        random.seed(restart * 7919)
        
        # Shuffle candidates for diversity
        shuffled = list(cands)
        random.shuffle(shuffled)
        
        added = []
        added_set = set()
        
        while time.time() < deadline_global:
            bad_cnt = bad_pairs_count(build_adj(added))
            if bad_cnt == 0:
                break
            
            # Score candidates by how many bad pairs they fix
            scored = []
            for e in shuffled:
                if e in added_set:
                    continue
                trial = added + [e]
                adj = build_adj(trial)
                if not is_triangle_free(adj):
                    continue
                b = bad_pairs_count(adj)
                scored.append((b, e))
            
            if not scored:
                break
            
            scored.sort()
            # Pick from top-k with some randomness
            k = min(3, len(scored))
            _, best_edge = random.choice(scored[:k])
            added.append(best_edge)
            added_set.add(best_edge)
        
        if is_valid_solution(added):
            cur = local_search_removal(added)
            if len(cur) < best_count:
                best_count = len(cur)
                best_solution = cur
    
    # Phase 3: Try to improve with swap-based local search
    if best_solution is not None and time.time() < deadline_global:
        improved = True
        while improved and time.time() < deadline_global:
            improved = False
            cur = list(best_solution)
            cur = local_search_swap(cur, cands)
            if len(cur) < best_count:
                best_count = len(cur)
                best_solution = cur
                improved = True
    
    if best_solution is None:
        return []
    return [[int(u), int(v)] for (u,v) in best_solution]

def construct():
    result = solve()
    return result
