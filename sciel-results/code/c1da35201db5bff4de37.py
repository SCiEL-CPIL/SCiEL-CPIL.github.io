import math
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Perfect squares in [2, 512]
    squares = set()
    i = 1
    while i * i <= max_sum:
        if i * i >= 2:
            squares.add(i * i)
        i += 1
    
    # Diagonal constraint: 2a not in squares
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)
    num_cand = len(candidates)
    
    # Partition into odd and even
    odds = [a for a in candidates if a % 2 == 1]
    evens = [a for a in candidates if a % 2 == 0]
    
    # Build bipartite conflict graph: odd_i -- even_j if odd_i + even_j is a perfect square
    # Map to indices
    odd_idx = {a: i for i, a in enumerate(odds)}
    even_idx = {a: i for i, a in enumerate(evens)}
    n_odd = len(odds)
    n_even = len(evens)
    
    # Adjacency: for each odd, list of conflicting even indices
    adj = [[] for _ in range(n_odd)]
    for i, o in enumerate(odds):
        for j, e in enumerate(evens):
            if o + e in squares:
                adj[i].append(j)
    
    # Solve Maximum Weight Independent Set on bipartite graph via min-cut
    # Node weights: all 1 (maximize cardinality)
    # 
    # Construction:
    #   Source S -> each odd node (capacity = weight = 1)
    #   Each even node -> Sink T (capacity = weight = 1)
    #   Odd_i -> Even_j (capacity = INF) for each conflict edge
    #
    # Min-cut: vertices on S-side of cut form the independent set
    # (odds reachable from S + evens NOT reachable from S)
    
    # Build flow network
    # Nodes: 0=Source, 1..n_odd=odds, n_odd+1..n_odd+n_even=evens, n_odd+n_even+1=Sink
    total_nodes = 1 + n_odd + n_even + 1
    source = 0
    sink = total_nodes - 1
    INF = 10**9
    
    # Adjacency list for residual graph
    # For each node, list of (neighbor, capacity, reverse_edge_index)
    # We'll use a simple adjacency structure
    graph = [[] for _ in range(total_nodes)]
    
    def add_edge(u, v, cap):
        graph[u].append([v, cap, len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])
    
    # Source -> odds (capacity 1)
    for i in range(n_odd):
        add_edge(source, 1 + i, 1)
    
    # Evens -> Sink (capacity 1)
    for j in range(n_even):
        add_edge(1 + n_odd + j, sink, 1)
    
    # Conflict edges: odd -> even (capacity INF)
    for i in range(n_odd):
        for j in adj[i]:
            add_edge(1 + i, 1 + n_odd + j, INF)
    
    # Dinic's algorithm for max flow
    def dinic():
        # BFS to build level graph
        def bfs():
            level = [-1] * total_nodes
            level[source] = 0
            queue = [source]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                for v, cap, _ in graph[u]:
                    if cap > 0 and level[v] < 0:
                        level[v] = level[u] + 1
                        queue.append(v)
            return level
        
        # DFS to find augmenting paths
        def dfs(u, flow, level, it):
            if u == sink:
                return flow
            while it[u] < len(graph[u]):
                v, cap, rev_idx = graph[u][it[u]]
                if cap > 0 and level[v] == level[u] + 1:
                    pushed = dfs(v, min(flow, cap), level, it)
                    if pushed > 0:
                        graph[u][it[u]][1] -= pushed
                        graph[v][rev_idx][1] += pushed
                        return pushed
                it[u] += 1
            return 0
        
        max_flow = 0
        while True:
            level = bfs()
            if level[sink] < 0:
                break
            it = [0] * total_nodes
            while True:
                pushed = dfs(source, INF, level, it)
                if pushed == 0:
                    break
                max_flow += pushed
        
        return level  # Return the final level array (reachable from source)
    
    final_level = dinic()
    
    # Extract independent set:
    # Odd nodes reachable from source (level >= 0) are in the independent set
    # Even nodes NOT reachable from source (level < 0) are in the independent set
    result_set = set()
    for i in range(n_odd):
        if final_level[1 + i] >= 0:
            result_set.add(odds[i])
    for j in range(n_even):
        if final_level[1 + n_odd + j] < 0:
            result_set.add(evens[j])
    
    # Verification
    for a in result_set:
        if 2 * a in squares:
            return []
    for a in result_set:
        for b in result_set:
            if a <= b and a + b in squares:
                return []
    
    if not result_set:
        # Fallback: return a single valid element
        for a in candidates:
            return [a]
    
    return list(result_set)
