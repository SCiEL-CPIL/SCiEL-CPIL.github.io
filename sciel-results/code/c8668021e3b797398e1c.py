"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from collections import deque
import sys
import random

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
        
    # Dinic's Max Flow Implementation
    class Dinic:
        def __init__(self, n):
            self.n = n
            self.graph = [[] for _ in range(n)]
            
        def add_edge(self, u, v, cap):
            self.graph[u].append([v, cap, len(self.graph[v])])
            self.graph[v].append([u, 0, len(self.graph[u]) - 1])
            
        def bfs(self, s, t):
            self.level = [-1] * self.n
            self.level[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v, cap, _ in self.graph[u]:
                    if cap > 0 and self.level[v] < 0:
                        self.level[v] = self.level[u] + 1
                        q.append(v)
            return self.level[t] >= 0
            
        def dfs(self, u, t, f):
            if u == t:
                return f
            for i in range(self.it[u], len(self.graph[u])):
                self.it[u] = i
                v, cap, rev = self.graph[u][i]
                if cap > 0 and self.level[u] < self.level[v]:
                    d = self.dfs(v, t, min(f, cap))
                    if d > 0:
                        self.graph[u][i][1] -= d
                        self.graph[v][rev][1] += d
                        return d
            return 0
            
        def max_flow(self, s, t):
            flow = 0
            INF = 10**18
            while self.bfs(s, t):
                self.it = [0] * self.n
                while True:
                    f = self.dfs(s, t, INF)
                    if f == 0:
                        break
                    flow += f
            return flow

    def is_feasible(branches):
        """
        Check if branches can support a clique subdivision using Max Flow.
        """
        k = len(branches)
        if k < 2:
            return False
        
        # Node mapping:
        # S = 0, T = 1
        # For each vertex v:
        #   if v is branch: v_src = 2 + 2*v, v_sink = 3 + 2*v
        #   if v is non-branch: v_in = 2 + 2*v, v_out = 3 + 2*v
        # Total nodes = 2 + 2*n
        
        num_nodes = 2 + 2 * n
        dinic = Dinic(num_nodes)
        
        S = 0
        T = 1
        INF = 10**9
        
        branch_set = set(branches)
        
        # Add internal edges for vertex capacities
        for v in range(n):
            v_node_in = 2 + 2 * v
            v_node_out = 3 + 2 * v
            if v in branch_set:
                # Branches:
                # S -> v_src (v_node_in)
                # v_sink (v_node_out) -> T
                # No direct edge v_in -> v_out with limited capacity for "passing through"
                # We handle connections via external edges.
                dinic.add_edge(S, v_node_in, k - 1)
                dinic.add_edge(v_node_out, T, k - 1)
            else:
                # Non-branches:
                # v_in -> v_out with capacity 1
                dinic.add_edge(v_node_in, v_node_out, 1)
        
        # Add graph edges
        for u, v in edges:
            # Determine source and sink nodes for direction u -> v
            # If u is branch, it acts as source (outgoing from u_src)
            # If u is non-branch, it acts as source (outgoing from u_out)
            if u in branch_set:
                u_out = 2 + 2 * u  # v_node_in is used as source port for branches? 
                # Wait, in my mapping above:
                # Branch: v_src = 2+2v, v_sink = 3+2v
                # So u_out (for flow leaving u) is 2+2u.
            else:
                u_out = 3 + 2 * u  # v_node_out
            
            if v in branch_set:
                v_in = 3 + 2 * v  # v_node_out is used as sink port for branches?
                # Branch: v_sink = 3+2v. Flow enters v_sink.
            else:
                v_in = 2 + 2 * v  # v_node_in
            
            dinic.add_edge(u_out, v_in, INF)
            
            # Reverse direction v -> u
            if v in branch_set:
                v_out = 2 + 2 * v
            else:
                v_out = 3 + 2 * v
                
            if u in branch_set:
                u_in = 3 + 2 * u
            else:
                u_in = 2 + 2 * u
            
            dinic.add_edge(v_out, u_in, INF)
            
        required_flow = k * (k - 1) // 2
        max_flow_val = dinic.max_flow(S, T)
        
        return max_flow_val == required_flow

    def find_paths(branches):
        """
        Try to find pairwise internally disjoint paths between all pairs of branches.
        Returns dict mapping (i,j) to path, or None if not possible.
        """
        k = len(branches)
        if k < 2:
            return None
        
        branch_to_idx = {v: i for i, v in enumerate(branches)}
        branch_set = set(branches)
        
        used_interior = set()
        paths = {}
        
        pairs = list(combinations(range(k), 2))
        
        def bfs_shortest_path(src, dst, forbidden):
            if src == dst:
                return [src]
            visited = {src}
            queue = deque([(src, [src])])
            while queue:
                curr, path = queue.popleft()
                for nb in adj[curr]:
                    if nb == dst:
                        return path + [dst]
                    if nb in forbidden or nb in visited:
                        continue
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
            return None
        
        # Try multiple orderings if the first one fails
        # Heuristic 1: Shortest path first
        # Heuristic 2: Longest path first
        # Heuristic 3: Random
        
        attempts = 0
        while attempts < 20:
            attempts += 1
            used_interior.clear()
            paths.clear()
            
            if attempts == 1:
                # Sort by shortest path length
                pair_lengths = []
                for i, j in pairs:
                    sp = bfs_shortest_path(branches[i], branches[j], set())
                    if sp is None:
                        return None
                    pair_lengths.append((len(sp), i, j))
                pair_lengths.sort()
                ordered_pairs = [(i, j) for _, i, j in pair_lengths]
            elif attempts == 2:
                # Sort by longest path length
                pair_lengths = []
                for i, j in pairs:
                    sp = bfs_shortest_path(branches[i], branches[j], set())
                    if sp is None:
                        return None
                    pair_lengths.append((len(sp), i, j))
                pair_lengths.sort(reverse=True)
                ordered_pairs = [(i, j) for _, i, j in pair_lengths]
            else:
                ordered_pairs = pairs[:]
                random.shuffle(ordered_pairs)
            
            success = True
            for i, j in ordered_pairs:
                sp = bfs_shortest_path(branches[i], branches[j], used_interior)
                if sp is None:
                    success = False
                    break
                # Check if path goes through another branch
                valid = True
                for v in sp[1:-1]:
                    if v in branch_set:
                        valid = False
                        break
                if not valid:
                    success = False
                    break
                    
                for v in sp[1:-1]:
                    used_interior.add(v)
                paths[(i, j)] = sp
                
            if success:
                return paths
        
        return None

    def try_subset(branches):
        """Try to find valid paths for the given subset of branches."""
        if not is_feasible(branches):
            return None
            
        paths = find_paths(branches)
        if paths is None:
            return None
        result_paths = []
        for (i, j), path in paths.items():
            result_paths.append(list(path))
        return {'branches': list(branches), 'paths': result_paths}
    
    # Strategy: Greedy addition with feasibility check
    best_result = None
    best_size = 0
    
    degrees = [len(adj[v]) for v in range(n)]
    sorted_vertices = sorted(range(n), key=lambda v: -degrees[v])
    
    # Try to build a set
    current_set = []
    
    for v in sorted_vertices:
        trial_set = current_set + [v]
        if len(trial_set) >= 2:
            # Only check feasibility if it's potentially better
            if len(trial_set) > best_size:
                result = try_subset(trial_set)
                if result is not None:
                    current_set = trial_set
                    best_result = result
                    best_size = len(trial_set)
    
    if best_result is not None:
        return best_result
        
    # Fallback: find any triangle
    for row in combinations(range(n), 3):
        if all((row[i], row[j]) in edge_set for i in range(3) for j in range(i+1, 3)):
            return {'branches': list(row), 'paths': [[row[i], row[j]] for i in range(3) for j in range(i+1, 3)]}
    
    # Fallback: find any edge
    if edges:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    # Should not reach here
    return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
