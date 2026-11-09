"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import deque

P = {'n': 12, 'minimum_degree': 3, 'girth': 4}

def _count_distinct_cycle_lengths(adj):
    """Count distinct simple cycle lengths in the graph."""
    n = len(adj)
    # Find all simple cycles using Johnson's algorithm (simplified)
    # For small graphs, we can use a simpler approach
    cycle_lengths = set()
    
    # Use DFS to find all simple cycles
    # For efficiency, we'll find cycles by fixing a start vertex and doing DFS
    # with the constraint that the start vertex is the smallest in the cycle
    
    for start in range(n):
        # Find all simple cycles where 'start' is the smallest vertex
        # and the cycle returns to start
        # Use backtracking
        visited = [False] * n
        path = [start]
        visited[start] = True
        
        def dfs(current, length):
            # Only consider cycles where start is the minimum vertex
            # and we've returned to start
            for neighbor in adj[current]:
                if neighbor == start and length >= 4:
                    # Found a cycle
                    cycle_lengths.add(length)
                    return
                if (not visited[neighbor]) and neighbor > start:
                    # Only allow vertices > start to avoid duplicate cycles
                    # and to ensure start is the minimum
                    visited[neighbor] = True
                    path.append(neighbor)
                    dfs(neighbor, length + 1)
                    path.pop()
                    visited[neighbor] = False
        
        dfs(start, 1)
    
    return len(cycle_lengths)

def _validate_graph(edges):
    """Validate the graph meets all constraints."""
    n = P['n']
    # Check no loops or duplicates
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u >= n or v >= n or u < 0 or v < 0:
            return False
        e = (min(u,v), max(u,v))
        if e in edge_set:
            return False
        edge_set.add(e)
    
    # Check degree
    degree = [0] * n
    adj = [[] for _ in range(n)]
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
        adj[u].append(v)
        adj[v].append(u)
    
    if any(d < P['minimum_degree'] for d in degree):
        return False
    
    # Check triangle-free
    for i in range(n):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and k in adj[i]:
                    return False
    
    # Check has a cycle
    # BFS/DFS to check connectivity and cycles
    # A graph with n vertices and m edges has a cycle if m >= n (for connected components)
    # More precisely, if any component has edges >= vertices
    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            # BFS
            queue = deque([i])
            visited[i] = True
            comp_vertices = 0
            comp_edges = 0
            while queue:
                u = queue.popleft()
                comp_vertices += 1
                comp_edges += len(adj[u])
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            if comp_edges // 2 >= comp_vertices:
                return True  # Has a cycle
    
    return False  # No cycle found

def _make_edges_from_adj(adj):
    """Convert adjacency list to edge list."""
    edges = []
    n = len(adj)
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    n = P['n']
    min_deg = P['minimum_degree']
    
    best_score = None
    best_edges = None
    
    # First, try the known prism graph as a baseline
    es = set()
    for i in range(6):
        for u,v in [(i,(i+1)%6),(i+6,(i+1)%6+6),(i,i+6)]:
            es.add(tuple(sorted((u,v))))
    baseline_edges = [list(e) for e in sorted(es)]
    if _validate_graph(baseline_edges):
        adj = [[] for _ in range(n)]
        for u,v in baseline_edges:
            adj[u].append(v)
            adj[v].append(u)
        best_score = _count_distinct_cycle_lengths(adj)
        best_edges = baseline_edges
    
    # Now do randomized search
    random.seed(42)
    
    for iteration in range(100000):
        if time.time() > deadline:
            break
        
        # Generate a random cubic graph
        # Use configuration model or random edge addition
        adj = [[] for _ in range(n)]
        edges = set()
        
        # Try to build a random cubic graph
        success = False
        for attempt in range(50):
            adj = [[] for _ in range(n)]
            edges = set()
            # Randomly add edges until all degrees are 3
            remaining = [3] * n
            edge_list = []
            valid = True
            for _ in range(n * 3 // 2):  # 18 edges
                # Pick two vertices with remaining degree
                candidates = [i for i in range(n) if remaining[i] > 0]
                if len(candidates) < 2:
                    valid = False
                    break
                u, v = random.sample(candidates, 2)
                if (min(u,v), max(u,v)) not in edges:
                    edges.add((min(u,v), max(u,v)))
                    remaining[u] -= 1
                    remaining[v] -= 1
                    edge_list.append((u,v))
                else:
                    valid = False
                    break
            
            if valid and all(r == 0 for r in remaining):
                success = True
                break
        
        if not success:
            continue
        
        # Check triangle-free
        adj = [[] for _ in range(n)]
        for u,v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        triangle_free = True
        for i in range(n):
            for j in adj[i]:
                for k in adj[j]:
                    if k != i and k in adj[i]:
                        triangle_free = False
                        break
                if not triangle_free:
                    break
            if not triangle_free:
                break
        
        if not triangle_free:
            continue
        
        # Check has cycle (should be true for cubic graph with 12 vertices)
        # Count distinct cycle lengths
        score = _count_distinct_cycle_lengths(adj)
        
        if best_score is None or score < best_score:
            best_score = score
            best_edges = [list(e) for e in sorted(edges)]
            if best_score == 1:
                break  # Can't do better than 1
    
    if best_edges is None:
        # Fallback to baseline
        best_edges = baseline_edges
    
    # Ensure proper types
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
