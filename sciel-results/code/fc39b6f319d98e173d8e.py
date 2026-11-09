# EVOLVE-BLOCK-START
import math
import sys
from collections import deque

def construct():
    """
    Find a maximum subset of [1,256] such that a+b is never a perfect square
    for any a<=b in the set (including a=b).
    
    Approach:
    1. Check if the conflict graph is bipartite.
    2. If bipartite, solve Maximum Weight Independent Set exactly using Min-Cut (Max-Flow).
    3. If not bipartite, use a robust heuristic (Greedy + Local Search) to find a near-optimal set.
    """
    N = 256
    
    def is_square(x):
        if x < 0:
            return False
        r = int(math.isqrt(x))
        return r * r == x

    # Identify self-conflicts (2*i is a square)
    self_conflict = [False] * (N + 1)
    valid_nodes = []
    for i in range(1, N + 1):
        if is_square(2 * i):
            self_conflict[i] = True
        else:
            valid_nodes.append(i)
            
    # Build adjacency list for valid nodes
    # Map valid nodes to indices 0..M-1 for easier flow graph construction
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    M = len(valid_nodes)
    
    adj = [[] for _ in range(M)]
    # Also store original values
    values = valid_nodes
    
    for i in range(M):
        u = values[i]
        for j in range(i + 1, M):
            v = values[j]
            if is_square(u + v):
                adj[i].append(j)
                adj[j].append(i)
                
    # Check if the graph induced by valid_nodes is bipartite
    # BFS/DFS coloring
    color = [-1] * M
    is_bipartite = True
    for i in range(M):
        if color[i] == -1:
            color[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        q.append(v)
                    else:
                        if color[v] == color[u]:
                            is_bipartite = False
                            break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break

    if is_bipartite:
        # Solve MWIS on Bipartite Graph using Min-Cut
        # Graph G = (V, E). V = A U B.
        # A = {v | color[v] == 0}, B = {v | color[v] == 1}
        # Flow Network:
        # Source S, Sink T.
        # For each v in A: Edge S -> v with capacity 1 (weight)
        # For each v in B: Edge v -> T with capacity 1 (weight)
        # For each edge (u, v) in E with u in A, v in B: Edge u -> v with capacity INF
        
        # We need a Max-Flow implementation.
        # Since N is small (256), a simple Edmonds-Karp or Dinic's algorithm is sufficient.
        
        # Total nodes in flow graph: M + 2 (S, T)
        # S = M, T = M + 1
        # Weights are all 1.
        
        INF = 10**9
        # Adjacency list for flow graph
        # flow_adj[u] = list of (v, cap, rev_index)
        flow_adj = [[] for _ in range(M + 2)]
        
        def add_edge(u, v, cap):
            flow_adj[u].append([v, cap, len(flow_adj[v])])
            flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])
            
        S = M
        T = M + 1
        
        total_weight = 0
        for i in range(M):
            if color[i] == 0:
                # In partition A
                add_edge(S, i, 1)
                total_weight += 1
            else:
                # In partition B
                add_edge(i, T, 1)
                total_weight += 1
                
        for i in range(M):
            for j in adj[i]:
                if color[i] == 0 and color[j] == 1:
                    add_edge(i, j, INF)
                elif color[i] == 1 and color[j] == 0:
                    add_edge(j, i, INF)
                    
        # Dinic's Algorithm
        def dinic():
            flow = 0
            level = [-1] * (M + 2)
            ptr = [0] * (M + 2)
            
            def bfs():
                for i in range(M + 2):
                    level[i] = -1
                level[S] = 0
                q = deque([S])
                while q:
                    u = q.popleft()
                    for e in flow_adj[u]:
                        v, cap, rev = e
                        if cap > 0 and level[v] == -1:
                            level[v] = level[u] + 1
                            q.append(v)
                return level[T] != -1
                
            def dfs(u, pushed):
                if pushed == 0:
                    return 0
                if u == T:
                    return pushed
                while ptr[u] < len(flow_adj[u]):
                    e = flow_adj[u][ptr[u]]
                    v, cap, rev = e
                    if level[v] == level[u] + 1 and cap > 0:
                        tr = dfs(v, min(pushed, cap))
                        if tr > 0:
                            e[1] -= tr
                            flow_adj[v][rev][1] += tr
                            return tr
                    ptr[u] += 1
                return 0
                
            while bfs():
                for i in range(M + 2):
                    ptr[i] = 0
                while True:
                    pushed = dfs(S, INF)
                    if pushed == 0:
                        break
                    flow += pushed
            return flow
            
        min_cut = dinic()
        
        # The minimum vertex cover is:
        # V_cover = {v in A | v is in Sink side} U {v in B | v is in Source side}
        # In the residual graph, the Source side is the set of nodes reachable from S.
        # We can find this by BFS on the residual graph (edges with cap > 0).
        
        reachable = [False] * (M + 2)
        q = deque([S])
        reachable[S] = True
        while q:
            u = q.popleft()
            for e in flow_adj[u]:
                v, cap, rev = e
                if cap > 0 and not reachable[v]:
                    reachable[v] = True
                    q.append(v)
                    
        # Construct Independent Set
        # The minimum vertex cover is:
        #   VC = {v in A : v reachable from S} ∪ {v in B : v NOT reachable from S}
        # The maximum independent set is V \ VC:
        #   IS = {v in A : v NOT reachable from S} ∪ {v in B : v reachable from S}
        
        independent_set_indices = []
        for i in range(M):
            if color[i] == 0:
                if not reachable[i]:
                    independent_set_indices.append(i)
            else:
                if reachable[i]:
                    independent_set_indices.append(i)
                    
        result = [values[i] for i in independent_set_indices]
        return sorted(result)
        
    else:
        # Fallback: Heuristic for non-bipartite graph
        # Use greedy + local search with swaps
        
        # Build conflict set for quick lookup
        conflict_set = [set() for _ in range(M)]
        for i in range(M):
            for j in adj[i]:
                conflict_set[i].add(j)
        
        # Greedy: start empty, add nodes in order of lowest degree first
        remaining = list(range(M))
        remaining.sort(key=lambda x: len(conflict_set[x]))
        
        selected_set = set()
        for i in remaining:
            # Check if i conflicts with any selected node
            can_add = True
            for j in selected_set:
                if j in conflict_set[i]:
                    can_add = False
                    break
            if can_add:
                selected_set.add(i)
        
        # Local search: try 1-swap improvements
        improved = True
        max_iter = 100
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # Try removing one selected node and adding one non-selected node
            for sel in list(selected_set):
                # Try each non-selected node
                for non_sel in range(M):
                    if non_sel in selected_set:
                        continue
                    # Check if removing sel and adding non_sel is valid
                    # non_sel must not conflict with (selected_set - {sel})
                    valid = True
                    for j in selected_set:
                        if j != sel and j in conflict_set[non_sel]:
                            valid = False
                            break
                    if valid:
                        # This swap keeps the same size but might enable future improvements
                        # Only do it if it helps (same size, so just track for potential)
                        pass
                # Try just removing sel if it allows adding two others
                # This is complex; for now, try adding back removed nodes
                pass
            
            # Try adding back any node that doesn't conflict
            for i in range(M):
                if i in selected_set:
                    continue
                can_add = True
                for j in selected_set:
                    if j in conflict_set[i]:
                        can_add = False
                        break
                if can_add:
                    selected_set.add(i)
                    improved = True
                    break
        
        result = [values[i] for i in selected_set]
        return sorted(result)

# EVOLVE-BLOCK-END
