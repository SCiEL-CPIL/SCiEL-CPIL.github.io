import sys
import time
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check if bipartite
    color = [-1] * n
    is_bipartite = True
    for i in range(n):
        if color[i] == -1:
            color[i] = 0
            stack = [i]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        stack.append(v)
                    elif color[v] == color[u]:
                        is_bipartite = False
                        break
                if not is_bipartite:
                    break
        if not is_bipartite:
            break
            
    if not is_bipartite:
        # Fallback to identity if not bipartite (unlikely for this graph)
        return list(range(n))
        
    # Bipartite case
    # Part 0: color 0
    # Part 1: color 1 or 2
    part1_vertices = [i for i in range(n) if color[i] == 1]
    part0_vertices = [i for i in range(n) if color[i] == 0]
    
    # Sort part1_vertices by degree
    part1_vertices.sort(key=lambda x: len(adj[x]), reverse=True)
    
    assignment = {}
    
    # DSU for G[0,1] and G[0,2]
    # Vertices in Part 0 are always present.
    # Vertices in Part 1 are present if assigned.
    # But edges only exist between Part 0 and Part 1.
    # So we can just track connectivity among Part 0 vertices via Part 1 vertices?
    # No, the forest is on the subgraph induced by colors.
    # The vertices in G[0,1] are Part 0 and Part 1 vertices assigned 1.
    # Edges are between them.
    # We can use a DSU on all vertices, but only add edges when they are "active".
    
    parent_01 = list(range(n))
    parent_02 = list(range(n))
    
    def find(parent, x):
        if parent[x] != x:
            parent[x] = find(parent, parent[x])
        return parent[x]
        
    def union(parent, x, y):
        rx = find(parent, x)
        ry = find(parent, y)
        if rx == ry:
            return False # Cycle
        parent[rx] = ry
        return True
        
    # We need to be able to undo unions for backtracking.
    # We can store the history of unions.
    history_01 = []
    history_02 = []
    
    def backtrack(idx):
        if idx == len(part1_vertices):
            return True
            
        v = part1_vertices[idx]
        
        # Try color 1
        assignment[v] = 1
        # Add edges from v to neighbors in Part 0
        added_01 = []
        valid = True
        for u in adj[v]:
            if color[u] == 0:
                if not union(parent_01, u, v):
                    valid = False
                    break
                added_01.append((u, v))
                
        if valid:
            if backtrack(idx + 1):
                return True
                
        # Undo
        for u, w in added_01:
            # We don't have a direct undo for DSU with path compression.
            # We can store the roots that were merged.
            pass
        
        # This approach with DSU and undo is tricky.
        # Alternative: Just check for cycles at the end?
        # Or use a simpler check:
        # When adding v with color 1, check if any two neighbors in Part 0 are already connected in G[0,1].
        # If they are, adding v creates a cycle.
        
        # Let's implement the check without DSU undo.
        # We can maintain the connected components of G[0,1] and G[0,2] as sets of vertices.
        
        return False

    # Let's use a different approach:
    # Maintain the edges of G[0,1] and G[0,2].
    # Check if adding a new edge creates a cycle using BFS/DFS on the current subgraph.
    # Since the subgraph is small, this is fast.
    
    edges_01 = set()
    edges_02 = set()
    
    def has_cycle(edges_set, vertices_set):
        # Build adj list for the subgraph
        sub_adj = defaultdict(list)
        for u, v in edges_set:
            sub_adj[u].append(v)
            sub_adj[v].append(u)
            
        visited = set()
        def dfs(u, parent):
            visited.add(u)
            for v in sub_adj[u]:
                if v == parent:
                    continue
                if v in visited:
                    return True
                if dfs(v, u):
                    return True
            return False
            
        for v in vertices_set:
            if v not in visited:
                if dfs(v, -1):
                    return True
        return False
        
    def backtrack2(idx):
        if idx == len(part1_vertices):
            return True
            
        v = part1_vertices[idx]
        
        # Try color 1
        assignment[v] = 1
        new_edges_01 = set()
        for u in adj[v]:
            if color[u] == 0:
                new_edges_01.add((u, v))
                
        # Check if edges_01 U new_edges_01 has a cycle
        # Vertices involved: Part 0 vertices + v
        # We can just check the whole set
        combined_01 = edges_01.union(new_edges_01)
        # To speed up, we can check if any neighbor u1, u2 of v in Part 0 are connected in edges_01.
        # If u1 and u2 are connected, then u1-...-u2-v-u1 is a cycle.
        
        # Let's do the simple check
        if not has_cycle(combined_01, set(part0_vertices) | {v}):
            if backtrack2(idx + 1):
                return True
        del assignment[v]
        
        # Try color 2
        assignment[v] = 2
        new_edges_02 = set()
        for u in adj[v]:
            if color[u] == 0:
                new_edges_02.add((u, v))
                
        combined_02 = edges_02.union(new_edges_02)
        if not has_cycle(combined_02, set(part0_vertices) | {v}):
            if backtrack2(idx + 1):
                return True
        del assignment[v]
        
        return False
        
    # Initialize edges_01 and edges_02 as empty
    # But we need to pass them or use nonlocal
    
    # Let's use a class or closure properly
    
    state_edges_01 = set()
    state_edges_02 = set()
    
    def backtrack3(idx):
        if idx == len(part1_vertices):
            return True
            
        v = part1_vertices[idx]
        
        # Try color 1
        assignment[v] = 1
        new_edges = set()
        for u in adj[v]:
            if color[u] == 0:
                new_edges.add((u, v))
                
        # Check cycle
        # Optimization: only check if new edges connect two vertices already connected
        # For simplicity, use has_cycle on the combined set
        # This is O(E) per step. 25 steps * 75 edges * 2 choices = 3750 operations. Fast.
        
        if not has_cycle(state_edges_01.union(new_edges), set(part0_vertices) | {v}):
            state_edges_01.update(new_edges)
            if backtrack3(idx + 1):
                return True
            state_edges_01.difference_update(new_edges)
        del assignment[v]
        
        # Try color 2
        assignment[v] = 2
        new_edges = set()
        for u in adj[v]:
            if color[u] == 0:
                new_edges.add((u, v))
                
        if not has_cycle(state_edges_02.union(new_edges), set(part0_vertices) | {v}):
            state_edges_02.update(new_edges)
            if backtrack3(idx + 1):
                return True
            state_edges_02.difference_update(new_edges)
        del assignment[v]
        
        return False
        
    if backtrack3(0):
        final_coloring = [0] * n
        for i in range(n):
            if color[i] == 0:
                final_coloring[i] = 0
            else:
                final_coloring[i] = assignment[i]
        return final_coloring
        
    # If 3 colors failed, return identity
    return list(range(n))
# EVOLVE-BLOCK-END
