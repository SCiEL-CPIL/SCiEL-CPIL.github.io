import time
import sys
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Build adjacency set for fast lookup
    adj_set = [set() for _ in range(n)]
    for u, v in edges:
        adj_set[u].add(v)
        adj_set[v].add(u)

    def has_bicolored_cycle(coloring, c1, c2):
        """Check if the subgraph induced by colors c1 and c2 contains a cycle using BFS."""
        if c1 == c2:
            return False
        
        # Get all vertices with color c1 or c2
        vertices = set()
        for i in range(n):
            if coloring[i] == c1 or coloring[i] == c2:
                vertices.add(i)
        
        if len(vertices) < 3:
            return False
        
        # BFS to detect cycle: a cycle exists if we find a back edge
        # In a forest, each connected component with k vertices has exactly k-1 edges
        visited = set()
        for start in vertices:
            if start in visited:
                continue
            # BFS from start
            queue = deque([start])
            visited.add(start)
            edge_count = 0
            vertex_count = 0
            while queue:
                curr = queue.popleft()
                vertex_count += 1
                for nxt in adj[curr]:
                    if nxt in vertices:
                        edge_count += 1  # Count each edge twice, so divide by 2 later
                        if nxt not in visited:
                            visited.add(nxt)
                            queue.append(nxt)
            edge_count //= 2
            # A tree with vertex_count vertices has vertex_count - 1 edges
            if edge_count >= vertex_count:
                return True
        return False

    def is_valid_coloring(coloring):
        """Check if coloring is proper and has no bicolored cycles."""
        if coloring is None:
            return False
        # Check properness
        for u, v in edges:
            if coloring[u] == coloring[v]:
                return False
        # Check bicolored cycles
        colors_used = set(coloring)
        for i, c1 in enumerate(colors_used):
            for c2 in list(colors_used)[i+1:]:
                if has_bicolored_cycle(coloring, c1, c2):
                    return False
        return True

    def greedy_coloring(k, order=None):
        """Greedy coloring with forward checking."""
        if order is None:
            order = list(range(n))
        
        coloring = [-1] * n
        
        for v in order:
            # Find forbidden colors
            forbidden = set()
            for u in adj[v]:
                if coloring[u] != -1:
                    forbidden.add(coloring[u])
            
            # Try colors in order, check bicolored constraint
            for c in range(k):
                if c in forbidden:
                    continue
                
                # Tentatively assign
                coloring[v] = c
                
                # Check if this creates a bicolored cycle
                valid = True
                for u in adj[v]:
                    if coloring[u] != -1 and coloring[u] != c:
                        if has_bicolored_cycle(coloring, c, coloring[u]):
                            valid = False
                            break
                
                if valid:
                    continue
                else:
                    coloring[v] = -1
            
            if coloring[v] == -1:
                return None
        
        return coloring[:]

    def backtracking_coloring(k, order, deadline):
        """Backtracking with union-find for efficient bicolored cycle detection."""
        coloring = [-1] * n
        
        # Union-Find structures per color pair
        # uf_data[key] = [parent_dict, rank_dict]
        uf_data = {}
        # Undo stack for each UF: list of operations
        uf_undo = {}
        
        def get_uf(c1, c2):
            key = (c1, c2) if c1 < c2 else (c2, c1)
            if key not in uf_data:
                uf_data[key] = [{}, {}]
                uf_undo[key] = []
            return key, uf_data[key]
        
        def uf_find(parent, x):
            if x not in parent:
                return x
            while parent[x] != x:
                x = parent[x]
            return x
        
        def uf_union(parent, rank, u, v, undo_stack):
            ru = uf_find(parent, u)
            rv = uf_find(parent, v)
            if ru == rv:
                return False
            
            if rank.get(ru, 0) < rank.get(rv, 0):
                ru, rv = rv, ru
            
            undo_stack.append((u, v, ru, rv, parent.get(ru, ru), parent.get(rv, rv), rank.get(ru, 0), rank.get(rv, 0)))
            
            parent[rv] = ru
            if rank.get(ru, 0) == rank.get(rv, 0):
                rank[ru] = rank.get(ru, 0) + 1
            return True
        
        def uf_undo_last(parent, rank, undo_stack):
            if not undo_stack:
                return False
            u, v, ru, rv, parent_ru, parent_rv, rank_ru, rank_rv = undo_stack.pop()
            
            if parent_ru != ru:
                parent[ru] = parent_ru
            if parent_rv != rv:
                parent[rv] = parent_rv
            
            if rank_ru != 0:
                rank[ru] = rank_ru
            else:
                rank.pop(ru, None)
            if rank_rv != 0:
                rank[rv] = rank_rv
            else:
                rank.pop(rv, None)
            return True
        
        def can_assign(v, c):
            # Check properness
            for u in adj[v]:
                if coloring[u] == c:
                    return False
            
            # Check bicolored cycles using union-find
            neighbor_colors = defaultdict(list)
            for u in adj[v]:
                if coloring[u] != -1 and coloring[u] != c:
                    neighbor_colors[coloring[u]].append(u)
            
            for d, neighbors in neighbor_colors.items():
                if len(neighbors) < 2:
                    continue
                
                key, (parent, rank) = get_uf(c, d)
                
                # Check if any two neighbors are connected
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        if uf_find(parent, neighbors[i]) == uf_find(parent, neighbors[j]):
                            return False
            return True
        
        def assign(v, c):
            coloring[v] = c
            for u in adj[v]:
                if coloring[u] != -1 and coloring[u] != c:
                    d = coloring[u]
                    key, (parent, rank) = get_uf(c, d)
                    uf_union(parent, rank, v, u, uf_undo[key])
        
        def unassign(v, c):
            coloring[v] = -1
            for u in adj[v]:
                if coloring[u] != -1 and coloring[u] != c:
                    d = coloring[u]
                    key, (parent, rank) = get_uf(c, d)
                    uf_undo_last(parent, rank, uf_undo[key])
        
        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if idx == n:
                return coloring[:]
            
            v = order[idx]
            
            for c in range(k):
                if can_assign(v, c):
                    assign(v, c)
                    res = backtrack(idx + 1)
                    if res is not None:
                        return res
                    unassign(v, c)
            return None
        
        return backtrack(0)

    # Strategy: Try multiple vertex orderings and approaches
    # 1. Natural order
    # 2. Degree descending
    # 3. BFS order from vertex 0
    # 4. A structural ordering based on the graph's pattern
    
    orders = []
    
    # Order 1: Natural
    orders.append(list(range(n)))
    
    # Order 2: Degree descending
    degrees = [len(adj[i]) for i in range(n)]
    orders.append(sorted(range(n), key=lambda x: degrees[x], reverse=True))
    
    # Order 3: BFS from vertex 0
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    bfs_order = [0]
    while queue:
        curr = queue.popleft()
        for nxt in adj[curr]:
            if not visited[nxt]:
                visited[nxt] = True
                queue.append(nxt)
                bfs_order.append(nxt)
    orders.append(bfs_order)
    
    # Order 4: Structural - group by "layer" based on the graph pattern
    # The graph seems to have a pattern where vertices can be grouped
    # Let's try ordering by: first the "spine" then the "branches"
    # Spine: 0-1-2-3-4-5-6-7-8-...-49 (the main path)
    # But also identify "ring" structures
    
    # Try a "wave" ordering: process vertices in waves from multiple sources
    wave_order = []
    visited = [False] * n
    # Start from vertices with high degree or at "corners"
    sources = [0, 49, 6, 43]
    for s in sources:
        if not visited[s]:
            visited[s] = True
            wave_order.append(s)
    
    frontier = list(sources)
    while frontier:
        next_frontier = []
        for v in frontier:
            for u in adj[v]:
                if not visited[u]:
                    visited[u] = True
                    next_frontier.append(u)
        wave_order.extend(next_frontier)
        frontier = next_frontier
    
    orders.append(wave_order)

    best_result = None
    best_k = n
    
    # Try increasing k with multiple orderings
    for k in range(3, 11):
        if time.time() > deadline - 5:
            break
        
        for order in orders:
            if time.time() > deadline - 2:
                break
            
            # Try backtracking first (more thorough)
            result = backtracking_coloring(k, order, deadline)
            if result is not None:
                if is_valid_coloring(result):
                    best_result = result
                    best_k = k
                    break
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback: greedy with more colors
        for k in range(5, 15):
            if time.time() > deadline - 2:
                break
            for order in orders:
                result = greedy_coloring(k, order)
                if result is not None and is_valid_coloring(result):
                    best_result = result
                    best_k = k
                    break
            if best_result is not None:
                break
    
    if best_result is None:
        # Final fallback: trivial coloring
        return [int(i) for i in range(n)]
    
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
