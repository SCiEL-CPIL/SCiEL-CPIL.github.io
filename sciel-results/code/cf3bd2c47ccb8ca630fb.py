import random
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment

def _build_graph():
    n = 100
    rng = random.Random(748)
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 1.8 / 100:
                edges.add((i, j))
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return n, adj

N, ADJ = _build_graph()

def _connected_components():
    visited = [False] * N
    comps = []
    for i in range(N):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in ADJ[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comps.append(comp)
    return comps

COMPS = _connected_components()

def _longest_path_via_mwm(comp):
    n = len(comp)
    if n == 0:
        return []
    if n == 1:
        return [comp[0]]
    if n == 2:
        if comp[0] in ADJ[comp[1]]:
            return [comp[0], comp[1]]
        else:
            return [comp[0]]
    
    # Map local indices
    idx = {v: i for i, v in enumerate(comp)}
    # Build bipartite graph cost matrix for Hungarian
    # Size 2n x 2n
    # We want to maximize matching size (or weight)
    # linear_sum_assignment minimizes cost, so we use -1 for valid edges, large cost for invalid
    INF_COST = 10**9
    cost = np.full((2*n, 2*n), INF_COST, dtype=float)
    
    for u in comp:
        for v in ADJ[u]:
            if v in idx:
                i = idx[u]
                j = idx[v]
                # Edge u_L -> v_R
                cost[i][n + j] = -1
                # Edge v_L -> u_R (since undirected)
                cost[j][n + i] = -1
                # Self loops? 
                # To allow endpoints, we can allow u_L -> u_R?
                # If we match u_L to u_R, it means u is an endpoint?
                # No, in the bipartite matching for path cover:
                # A match (u_L, v_R) corresponds to edge (u,v) in path.
                # If u is start, it has no incoming edge, so u_L is unmatched?
                # Or u_L is matched to a "dummy"?
                # Standard reduction:
                # To find a path cover, we match L to R.
                # Unmatched nodes in L are starts. Unmatched nodes in R are ends.
                # To allow a node to be both start and end (isolated), we don't match it.
                # To allow a node to be internal, it must be matched in L (outgoing) and in R (incoming).
                # So we just find a maximum cardinality matching.
                # We do NOT need self loops for the basic path cover.
                # However, if we want to allow a vertex to be an endpoint, we just leave its L or R side unmatched.
                # The Hungarian algorithm finds a maximum weight matching.
                # We set cost -1 for valid edges.
                # We should also allow "matching" a vertex to itself to represent it being an endpoint?
                # No, if we don't match u_L, u is a start. If we don't match u_R, u is an end.
                # So we just maximize the number of edges.
                
    # Run Hungarian
    row_ind, col_ind = linear_sum_assignment(cost)
    
    # Extract edges
    # An edge is (u, v) if row_ind[k] = i (u_L) and col_ind[k] = n + j (v_R)
    # And cost[i][n+j] == -1
    edges_in_path = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r][c] == -1:
            u = r
            v = c - n
            if u != v: # Skip self loops if any (we didn't add them)
                edges_in_path.add((u, v))
            else:
                # If we added self loops, handle them
                pass
    
    # Reconstruct paths from edges
    # Build adjacency list of the path cover
    path_adj = defaultdict(list)
    in_deg = {i: 0 for i in range(n)}
    out_deg = {i: 0 for i in range(n)}
    for u, v in edges_in_path:
        path_adj[u].append(v)
        out_deg[u] += 1
        in_deg[v] += 1
        
    # Find starts (in_deg == 0)
    starts = [i for i in range(n) if in_deg[i] == 0]
    
    best_path = []
    for s in starts:
        path = [s]
        curr = s
        while curr in path_adj and path_adj[curr]:
            curr = path_adj[curr][0]
            path.append(curr)
        if len(path) > len(best_path):
            best_path = path
            
    # If no starts found (all cycles), pick any node and traverse
    if not best_path:
        for i in range(n):
            if out_deg[i] > 0:
                path = [i]
                curr = i
                visited = {i}
                while curr in path_adj and path_adj[curr]:
                    curr = path_adj[curr][0]
                    if curr in visited:
                        break
                    path.append(curr)
                    visited.add(curr)
                if len(path) > len(best_path):
                    best_path = path
                break

    # Convert local indices to global
    return [comp[i] for i in best_path]

def _longest_path_heuristic(comp):
    n = len(comp)
    if n == 0: return []
    if n == 1: return [comp[0]]
    
    idx = {v: i for i, v in enumerate(comp)}
    local_adj = [set() for _ in range(n)]
    for u in comp:
        for v in ADJ[u]:
            if v in idx:
                local_adj[idx[u]].add(idx[v])
                
    best_path = [0]
    for start in range(n):
        # Greedy DFS
        stack = [(start, [start])]
        while stack:
            curr, path = stack.pop()
            if len(path) > len(best_path):
                best_path = path[:]
            for nxt in local_adj[curr]:
                if nxt not in path:
                    stack.append((nxt, path + [nxt]))
                    
    return [comp[i] for i in best_path]

component_paths = []
for comp in COMPS:
    # Use MWM approach
    p1 = _longest_path_via_mwm(comp)
    # Use Heuristic approach
    p2 = _longest_path_heuristic(comp)
    # Pick the longer one
    best = p1 if len(p1) >= len(p2) else p2
    if best:
        component_paths.append(best)

if component_paths:
    BEST_PATH = max(component_paths, key=len)
else:
    BEST_PATH = [0]

def construct():
    return list(BEST_PATH)
