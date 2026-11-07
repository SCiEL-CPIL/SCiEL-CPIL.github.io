"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _bfs_path(adj, src, dst, blocked):
    """BFS shortest path from src to dst avoiding blocked vertices as interiors."""
    if src == dst:
        return [src]
    parent = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        if cur == dst:
            path = []
            node = dst
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path
        for nb in adj[cur]:
            if nb not in parent:
                if nb == dst or (nb not in blocked and nb != src):
                    parent[nb] = cur
                    q.append(nb)
    return None

def construct():
    deadline = time.time() + 55
    adj = [set() for _ in range(P['n'])]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    n = P['n']
    deg = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    
    best_k = 0
    best_branches = None
    best_paths = None
    
    # Try multiple starting pairs and greedy extensions
    for i in range(len(order)):
        if time.time() > deadline:
            break
        for j in range(i+1, min(i+15, len(order))):
            if time.time() > deadline:
                break
            a, b = order[i], order[j]
            if b not in adj[a]:
                continue
            
            branches = [a, b]
            paths_dict = {(a, b): [a, b], (b, a): [a, b]}
            blocked = set()
            
            for cand in order:
                if time.time() > deadline:
                    break
                if cand in branches:
                    continue
                # Try to add cand: need paths from cand to all existing branches
                new_blocked = set(blocked)
                new_paths = {}
                feasible = True
                for br in branches:
                    p = _bfs_path(adj, br, cand, new_blocked)
                    if p is None:
                        feasible = False
                        break
                    for v in p[1:-1]:
                        if v in new_blocked or v in branches or v == cand:
                            feasible = False
                            break
                    if not feasible:
                        break
                    new_paths[br] = p
                    for v in p[1:-1]:
                        new_blocked.add(v)
                
                if feasible:
                    branches.append(cand)
                    blocked = new_blocked
                    for br in branches[:-1]:
                        paths_dict[(br, cand)] = new_paths[br]
                        paths_dict[(cand, br)] = new_paths[br]
                    if len(branches) > best_k:
                        best_k = len(branches)
                        best_branches = list(branches)
                        best_paths = dict(paths_dict)
    
    if best_branches is None:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    branch_list = [int(b) for b in best_branches]
    path_list = []
    for u, v in combinations(branch_list, 2):
        if (u, v) in best_paths:
            path_list.append([int(x) for x in best_paths[(u, v)]])
        elif (v, u) in best_paths:
            path_list.append([int(x) for x in best_paths[(v, u)]])
        else:
            path_list.append([u, v])
    
    return {'branches': branch_list, 'paths': path_list}
# EVOLVE-BLOCK-END
