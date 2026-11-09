# EVOLVE-BLOCK-START
import math
import sys
from collections import deque
import random

def construct():
    """
    Find a maximum subset of [1,256] such that a+b is never a perfect square
    for any a<=b in the set (including a=b).
    
    Approach:
    1. Check if the conflict graph is bipartite.
    2. If bipartite, solve Maximum Weight Independent Set exactly using Min-Cut (Max-Flow).
    3. If not bipartite, use Simulated Annealing to find a near-optimal set.
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
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    M = len(valid_nodes)
    
    adj = [[] for _ in range(M)]
    values = valid_nodes
    
    for i in range(M):
        u = values[i]
        for j in range(i + 1, M):
            v = values[j]
            if is_square(u + v):
                adj[i].append(j)
                adj[j].append(i)
                
    # Check if the graph induced by valid_nodes is bipartite
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
        INF = 10**9
        flow_adj = [[] for _ in range(M + 2)]
        
        def add_edge(u, v, cap):
            flow_adj[u].append([v, cap, len(flow_adj[v])])
            flow_adj[v].append([u, 0, len(flow_adj[u]) - 1])
            
        S = M
        T = M + 1
        
        total_weight = 0
        for i in range(M):
            if color[i] == 0:
                add_edge(S, i, 1)
                total_weight += 1
            else:
                add_edge(i, T, 1)
                total_weight += 1
                
        for i in range(M):
            for j in adj[i]:
                if color[i] == 0 and color[j] == 1:
                    add_edge(i, j, INF)
                elif color[i] == 1 and color[j] == 0:
                    add_edge(j, i, INF)
                    
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
                    
        independent_set_indices = []
        for i in range(M):
            if color[i] == 0:
                if reachable[i]:
                    independent_set_indices.append(i)
            else:
                if not reachable[i]:
                    independent_set_indices.append(i)
                    
        result = [values[i] for i in independent_set_indices]
        return sorted(result)
        
    else:
        # Fallback: Simulated Annealing for non-bipartite graph
        # We want to maximize the size of the independent set.
        
        # Initial solution: Greedy
        selected = set()
        # Sort nodes by degree descending to prioritize removing high-degree nodes? 
        # Or just add nodes that don't conflict.
        # Better: Start with all nodes, remove conflicts greedily.
        selected = set(range(M))
        
        # Greedy removal
        changed = True
        while changed:
            changed = False
            degrees = {i: 0 for i in selected}
            for u in selected:
                for v in adj[u]:
                    if v in selected:
                        degrees[u] += 1
            if degrees:
                max_deg_node = max(degrees, key=degrees.get)
                if degrees[max_deg_node] > 0:
                    selected.remove(max_deg_node)
                    changed = True

        best_set = set(selected)
        best_size = len(best_set)
        
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        max_iter = 50000
        
        current_set = set(best_set)
        current_size = len(current_set)
        
        import time
        start_time = time.time()
        
        while T > T_min and (time.time() - start_time) < 50:
            # Generate a move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            if move_type == 'add':
                # Pick a random node not in set
                candidates = [i for i in range(M) if i not in current_set]
                if not candidates:
                    continue
                v = random.choice(candidates)
                # Check if v can be added
                conflict = False
                for u in adj[v]:
                    if u in current_set:
                        conflict = True
                        break
                if not conflict:
                    new_set = current_set | {v}
                    new_size = current_size + 1
                    # Always accept improvement
                    current_set = new_set
                    current_size = new_size
                    if current_size > best_size:
                        best_set = set(current_set)
                        best_size = current_size
                # If conflict, we could try to remove conflicting nodes, but that's complex.
                # For SA, we only accept valid states. So if conflict, reject.
                
            elif move_type == 'remove':
                # Pick a random node in set
                if not current_set:
                    continue
                v = random.choice(list(current_set))
                new_set = current_set - {v}
                new_size = current_size - 1
                # Delta = -1
                # Accept with probability exp(1/T)
                if random.random() < math.exp(1.0 / T):
                    current_set = new_set
                    current_size = new_size
                    
            elif move_type == 'swap':
                # Pick v in set, w not in set
                if not current_set:
                    continue
                v = random.choice(list(current_set))
                candidates = [i for i in range(M) if i not in current_set]
                if not candidates:
                    continue
                w = random.choice(candidates)
                
                # Remove v, add w
                # Check if w conflicts with any node in (current_set - {v})
                conflict = False
                for u in adj[w]:
                    if u in current_set and u != v:
                        conflict = True
                        break
                if not conflict:
                    new_set = (current_set - {v}) | {w}
                    new_size = current_size # Size unchanged
                    # Delta = 0
                    # Accept with probability exp(0) = 1? 
                    # But we might get stuck. 
                    # Let's accept with some probability to explore.
                    if random.random() < 0.5:
                        current_set = new_set
                        # current_size unchanged
                        if current_size > best_size:
                            best_set = set(current_set)
                            best_size = current_size
            else:
                pass
                
            T *= cooling
            
        return sorted([values[i] for i in best_set])

# EVOLVE-BLOCK-END
