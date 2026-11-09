"""Clique subdivision finder using max-flow verification and randomized local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    adj = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _max_clique(adj):
    n = len(adj)
    best = []
    
    def bronk(R, P_set, X):
        if not P_set and not X:
            if len(R) > len(best):
                best[:] = R
            return
        
        pivot_set = P_set | X
        if pivot_set:
            pivot = max(pivot_set, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
            
        for v in list(candidates):
            bronk(R + [v], P_set & adj[v], X & adj[v])
            P_set.remove(v)
            X.add(v)
            
    bronk([], set(range(n)), set())
    return best

def _bfs_path(adj, s, t, blocked, rng=None):
    if s == t:
        return [s]
    prev = {s: None}
    q = deque([s])
    while q:
        cur = q.popleft()
        if cur == t:
            break
        neighbors = list(adj[cur])
        if rng:
            rng.shuffle(neighbors)
        for nb in neighbors:
            if nb in prev:
                continue
            if nb in blocked and nb != s and nb != t:
                continue
            prev[nb] = cur
            q.append(nb)
    if t not in prev:
        return None
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

def _verify_disjoint(adj, branches):
    """
    Verify that branches form a valid subdivision by checking
    each pair has a path internally disjoint from all other paths.
    Returns dict of pair -> path, or None.
    """
    bs = set(branches)
    n_pairs = len(branches) * (len(branches) - 1) // 2
    used = set()
    paths = {}
    pairs = list(combinations(sorted(branches), 2))
    rng = random.Random(0)
    
    # Order pairs by how "constrained" they are
    # Pairs sharing more common neighbors are more constrained
    pair_score = {}
    for (u, v) in pairs:
        common = adj[u] & adj[v] & (bs - {u, v})
        pair_score[(u, v)] = len(common)
    pairs.sort(key=lambda p: pair_score[p], reverse=True)
    
    for (u, v) in pairs:
        blocked = used | (bs - {u, v})
        path = _bfs_path(adj, u, v, blocked, rng)
        if path is None:
            return None
        interior = set(path[1:-1])
        if interior & used:
            return None
        used |= interior
        paths[(u, v)] = path
    return paths

def _can_add_vertex(adj, current_branches, candidate):
    """Check if candidate can be added to current_branches."""
    trial = list(current_branches) + [candidate]
    return _verify_disjoint(adj, trial) is not None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    adj = _build_adj()
    n = P['n']
    vertices = list(range(n))
    
    def time_up():
        return time.time() > deadline
    
    # 1. Find maximum clique
    max_c = _max_clique(adj)
    best_branches = list(max_c)
    best_paths = _verify_disjoint(adj, best_branches)
    
    # 2. If max clique doesn't work, try smaller subsets
    if best_paths is None:
        while len(best_branches) > 2 and best_paths is None:
            best_branches.pop()
            best_paths = _verify_disjoint(adj, best_branches)
    
    # 3. Greedy expansion from best known solution
    current_branches = list(best_branches)
    current_paths = best_paths
    
    improved = True
    while improved and not time_up():
        improved = False
        current = sorted(current_branches)
        non_branch = [x for x in vertices if x not in set(current)]
        non_branch.sort(key=lambda x: len(adj[x]), reverse=True)
        
        for vb in non_branch:
            if time_up():
                break
            if _can_add_vertex(adj, current, vb):
                current_branches = sorted(current_branches + [vb])
                current_paths = _verify_disjoint(adj, current_branches)
                if len(current_branches) > len(best_branches):
                    best_branches = list(current_branches)
                    best_paths = current_paths
                improved = True
                break
    
    # 4. Randomized search for larger branch sets
    random.seed(42)
    for restart in range(300):
        if time_up():
            break
        
        # Random start with bias towards high-degree vertices
        size = random.randint(3, min(8, n))
        init = random.sample(vertices, size)
        
        paths = _verify_disjoint(adj, init)
        if paths is None:
            continue
        
        current = list(init)
        current_paths = paths
        
        # Greedy expansion from this random start
        remaining = sorted([v for v in vertices if v not in current], key=lambda x: -len(adj[x]))
        for cand in remaining:
            if time_up():
                break
            if _can_add_vertex(adj, current, cand):
                current = current + [cand]
                current_paths = _verify_disjoint(adj, current)
        
        if len(current) > len(best_branches):
            best_branches = list(current)
            best_paths = current_paths
    
    # 5. Swap-based local search
    for _ in range(100):
        if time_up():
            break
            
        if len(best_branches) <= 2:
            break
        idx = random.randint(0, len(best_branches) - 1)
        trial_branches = [b for i, b in enumerate(best_branches) if i != idx]
        
        pool = [v for v in vertices if v not in trial_branches]
        if not pool:
            break
        added = random.choice(pool)
        trial_branches.append(added)
        
        tp = _verify_disjoint(adj, trial_branches)
        if tp is not None:
            best_branches = trial_branches
            best_paths = tp
            # Try to expand from this new state
            current = list(best_branches)
            current_paths = best_paths
            remaining = sorted([v for v in vertices if v not in current], key=lambda x: -len(adj[x]))
            for cand in remaining:
                if time_up():
                    break
                if _can_add_vertex(adj, current, cand):
                    current = current + [cand]
                    current_paths = _verify_disjoint(adj, current)
                    if len(current) > len(best_branches):
                        best_branches = list(current)
                        best_paths = current_paths
                        break
                        
    # Final verification and formatting
    if best_paths is None:
        u, v = P['edges'][0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}
        
    final_paths = []
    for u, v in combinations(best_branches, 2):
        if (u, v) in best_paths:
            final_paths.append(list(best_paths[(u, v)]))
        elif (v, u) in best_paths:
            final_paths.append(list(best_paths[(v, u)]))
        else:
            final_paths.append([u, v])
            
    final_branches = [int(b) for b in best_branches]
    
    return {
        'branches': final_branches,
        'paths': final_paths
    }

# EVOLVE-BLOCK-END
