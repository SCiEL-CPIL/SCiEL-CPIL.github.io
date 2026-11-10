# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
from collections import deque
import scipy.optimize as so
import numpy as np
from scipy.sparse import coo_matrix

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

# Precompute adjacency
N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)
adj_list = [sorted(adj[i]) for i in range(N)]

DEADLINE = time.time() + 50

def time_left():
    return DEADLINE - time.time()

def find_disjoint_paths(branches, n_branches):
    """
    Given a list of branch vertices, find one simple path for each unordered pair
    such that all path interiors are pairwise disjoint and avoid all branches.
    Uses incremental BFS with branch-vertex removal.
    Returns dict of (i,j) -> path, or None if impossible.
    """
    if n_branches < 2:
        return None
    branch_set = set(branches)
    pairs = list(combinations(range(n_branches), 2))
    def pair_key(p):
        u, v = branches[p[0]], branches[p[1]]
        return 0 if v in adj[u] else 1
    pairs.sort(key=pair_key)
    
    used_interior = set()
    paths = {}
    
    for (i, j) in pairs:
        s, t = branches[i], branches[j]
        parent = {}
        visited = {s}
        q = deque([s])
        found = False
        while q:
            u = q.popleft()
            if u == t:
                found = True
                break
            for w in adj_list[u]:
                if w in visited:
                    continue
                if w in branch_set and w != t:
                    continue
                if w in used_interior:
                    continue
                visited.add(w)
                parent[w] = u
                q.append(w)
        if not found:
            return None
        path = [t]
        cur = t
        while cur != s:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        for v in path[1:-1]:
            used_interior.add(v)
        paths[(i, j)] = path
    
    result = {}
    for (i, j) in combinations(range(n_branches), 2):
        result[(i, j)] = paths[(i, j)]
    return result

def check_feasibility_ilp(branches):
    """
    Uses MILP to check if there are vertex-disjoint paths for all pairs of branches.
    Returns a dict of paths if feasible, else None.
    """
    n_branches = len(branches)
    if n_branches < 2:
        return None
        
    pairs = list(combinations(range(n_branches), 2))
    num_pairs = len(pairs)
    
    m = len(P['edges'])
    
    dir_edges = []
    for i, (u, v) in enumerate(P['edges']):
        dir_edges.append((u, v))
        dir_edges.append((v, u))
        
    M = len(dir_edges)
    n_vars = M * num_pairs
    
    c = np.ones(n_vars)
    
    rows = []
    cols = []
    vals = []
    lbs = []
    ubs = []
    
    def add_constraint(row_coeffs, lb, ub):
        r = len(lbs)
        for ci, v in row_coeffs:
            rows.append(r)
            cols.append(ci)
            vals.append(v)
        lbs.append(lb)
        ubs.append(ub)
        
    In = [[] for _ in range(N)]
    Out = [[] for _ in range(N)]
    for idx, (u, v) in enumerate(dir_edges):
        Out[u].append(idx)
        In[v].append(idx)
        
    for p_idx, (i, j) in enumerate(pairs):
        s = branches[i]
        t = branches[j]
        for v in range(N):
            coeffs = []
            for e in In[v]:
                coeffs.append((e * num_pairs + p_idx, 1))
            for e in Out[v]:
                coeffs.append((e * num_pairs + p_idx, -1))
                
            if v == s:
                lb, ub = 1, 1
            elif v == t:
                lb, ub = -1, -1
            else:
                lb, ub = 0, 0
            
            add_constraint(coeffs, lb, ub)
            
    branch_set = set(branches)
    for v in range(N):
        if v in branch_set:
            continue
        coeffs = []
        for p_idx in range(num_pairs):
            for e in In[v]:
                coeffs.append((e * num_pairs + p_idx, 1))
        add_constraint(coeffs, 0, 1)
        
    for b in branches:
        for p_idx, (i, j) in enumerate(pairs):
            s = branches[i]
            t = branches[j]
            if b == s or b == t:
                continue
            coeffs_in = []
            for e in In[b]:
                coeffs_in.append((e * num_pairs + p_idx, 1))
            add_constraint(coeffs_in, 0, 0)
            
            coeffs_out = []
            for e in Out[b]:
                coeffs_out.append((e * num_pairs + p_idx, 1))
            add_constraint(coeffs_out, 0, 0)
            
    if not rows:
        A = coo_matrix(([], ([], [])), shape=(0, n_vars)).tocsr()
    else:
        A = coo_matrix((vals, (rows, cols)), shape=(len(lbs), n_vars)).tocsr()
    
    integrality = np.ones(n_vars)
    bounds = [(0, 1)] * n_vars
    
    res = so.milp(c, constraints=so.LinearConstraint(A, lbs, ubs), integrality=integrality, bounds=bounds)
    
    if res.status != 0:
        return None
        
    paths = {}
    for p_idx, (i, j) in enumerate(pairs):
        used_edges = []
        for e_idx in range(M):
            if res.x[e_idx * num_pairs + p_idx] > 0.5:
                used_edges.append(dir_edges[e_idx])
                
        s = branches[i]
        t = branches[j]
        used_adj = {u: [] for u, v in used_edges}
        for u, v in used_edges:
            if v not in used_adj:
                used_adj[v] = []
            used_adj[u].append(v)
            
        parent = {s: None}
        q = deque([s])
        found = False
        while q:
            u = q.popleft()
            if u == t:
                found = True
                break
            for v in used_adj.get(u, []):
                if v not in parent:
                    parent[v] = u
                    q.append(v)
        if not found:
            return None
        path = [t]
        cur = t
        while cur != s:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        paths[(i, j)] = path
        
    return paths

def try_branch_set(branches):
    """Try to route disjoint paths for the given branch set."""
    n = len(branches)
    # First try fast BFS
    result = find_disjoint_paths(branches, n)
    if result is not None:
        path_list = []
        for (i, j) in combinations(range(n), 2):
            path_list.append(result[(i, j)])
        return path_list
        
    # If BFS fails, try ILP
    if time_left() < 1:
        return None
        
    ilp_result = check_feasibility_ilp(branches)
    if ilp_result is not None:
        path_list = []
        for (i, j) in combinations(range(n), 2):
            path_list.append(ilp_result[(i, j)])
        return path_list
        
    return None

def greedy_build_branches(max_k, deadline_check=None):
    deg = [len(adj[i]) for i in range(N)]
    order = sorted(range(N), key=lambda x: -deg[x])
    
    branches = []
    for candidate in order:
        if len(branches) >= max_k:
            break
        if deadline_check and deadline_check() < 2:
            break
        trial = branches + [candidate]
        paths = try_branch_set(trial)
        if paths is not None:
            branches = trial
            if len(branches) >= max_k:
                break
    
    if len(branches) < 2:
        return None
    paths = try_branch_set(branches)
    if paths is None:
        return None
    return branches, paths

def local_search(branches, paths, deadline_check=None):
    n = len(branches)
    best = (branches, paths)
    
    for candidate in range(N):
        if candidate in branches:
            continue
        if deadline_check and deadline_check() < 2:
            break
        trial = list(branches) + [candidate]
        trial_paths = try_branch_set(trial)
        if trial_paths is not None:
            return trial, trial_paths
            
    for i in range(len(branches)):
        for candidate in range(N):
            if candidate in branches:
                continue
            if deadline_check and deadline_check() < 1:
                break
            trial = list(branches)
            trial[i] = candidate
            trial_paths = try_branch_set(trial)
            if trial_paths is not None:
                extended, ext_paths = local_search(trial, trial_paths, deadline_check)
                if extended is not None and len(extended) > len(branches):
                    return extended, ext_paths
    return best

def search_max_branches():
    def deadline_check():
        return time_left()
    
    best_branches = None
    best_paths = None
    best_count = 1
    
    u, v = P['edges'][0]
    fallback = ([u, v], [[u, v]])
    
    result = greedy_build_branches(N, deadline_check)
    if result is not None:
        b, p = result
        if len(b) > best_count:
            best_count = len(b)
            best_branches = b
            best_paths = p
    
    if time_left() > 10:
        for (u, v) in P['edges']:
            if time_left() < 5:
                break
            trial = [u, v]
            paths = try_branch_set(trial)
            if paths is None:
                continue
            extended, ext_paths = local_search(trial, paths, deadline_check)
            if extended is not None and len(extended) > best_count:
                best_count = len(extended)
                best_branches = extended
                best_paths = ext_paths
    
    import random
    if time_left() > 5:
        rng = random.Random(42)
        for _ in range(200):
            if time_left() < 3:
                break
            u = rng.randint(0, N-1)
            v = rng.choice(adj_list[u])
            trial = [u, v]
            paths = try_branch_set(trial)
            if paths is None:
                continue
            extended, ext_paths = local_search(trial, paths, deadline_check)
            if extended is not None and len(extended) > best_count:
                best_count = len(extended)
                best_branches = extended
                best_paths = ext_paths
                
    if best_branches is not None and time_left() > 3:
        extended, ext_paths = local_search(best_branches, best_paths, deadline_check)
        if extended is not None and len(extended) > best_count:
            best_count = len(extended)
            best_branches = extended
            best_paths = ext_paths
            
    if best_branches is None:
        return fallback
        
    return best_branches, best_paths

def construct():
    global DEADLINE
    DEADLINE = time.time() + 50
    
    branches, paths = search_max_branches()
    
    branches_out = [int(b) for b in branches]
    paths_out = [[int(v) for v in p] for p in paths]
    
    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END
