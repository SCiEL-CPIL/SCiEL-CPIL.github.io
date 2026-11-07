import random
import time
import sys
from collections import deque

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    n = 12
    min_deg = 3
    
    best_edges = None
    best_value = float('inf')
    
    # Helper to get distinct cycle lengths using Johnson's algorithm
    def get_cycle_lengths(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        lengths = set()
        
        # Johnson's algorithm for finding all simple cycles
        # For each start node, find cycles where start is the smallest node
        for start in range(n):
            # Block/unblock for Johnson's algorithm
            blocked = set()
            B = [set() for _ in range(n)]
            
            def unblock(u):
                if u in blocked:
                    blocked.discard(u)
                    for w in B[u]:
                        unblock(w)
            
            def circuit(v, start):
                nonlocal lengths
                cycle = []
                u = v
                while u != start:
                    cycle.append(u)
                    u = adj[u]  # This won't work directly, need to track path
                # This approach is flawed; let's use a different method
                pass
            
            # Simpler approach: DFS with start as minimum vertex
            def dfs(u, path, visited):
                nonlocal lengths
                for v in adj[u]:
                    if v == start and len(path) >= 4:
                        # Found a cycle of length len(path)
                        lengths.add(len(path))
                    elif v > start and v not in visited:
                        dfs(v, path + [v], visited | {v})
            
            dfs(start, [start], {start})
        
        return lengths
    
    # Better cycle enumeration using proper DFS
    def get_cycle_lengths_v2(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        lengths = set()
        
        for start in range(n):
            # Find all cycles where start is the smallest vertex
            # Use iterative DFS to avoid recursion depth issues
            stack = [(start, [start], {start})]
            while stack:
                u, path, visited = stack.pop()
                for v in adj[u]:
                    if v == start and len(path) >= 4:
                        lengths.add(len(path))
                    elif v > start and v not in visited:
                        stack.append((v, path + [v], visited | {v}))
        
        return lengths
    
    # Check if graph is valid: min degree >= 3, triangle-free, has a cycle
    def is_valid(edges):
        if not edges:
            return False
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check min degree
        for i in range(n):
            if len(adj[i]) < min_deg:
                return False
        
        # Check triangle-free
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    for w in adj[v]:
                        if w > v and w in adj[u]:
                            return False
        
        # Check has a cycle (n >= 4 and connected component with cycle)
        # For n=12 with min degree 3, there will be cycles
        lengths = get_cycle_lengths_v2(edges)
        if not lengths:
            return False
        
        return True
    
    # Try to find a 3-regular graph with girth >= 5 (to potentially get value 1)
    # Backtracking search for 3-regular triangle-free graphs with girth >= 5
    
    def try_girth5_search():
        nonlocal best_edges, best_value
        start_time_search = time.time()
        
        # Use backtracking to build a 3-regular graph with girth >= 5
        # This is complex; instead, try known constructions
        
        # Petersen graph has 10 vertices, girth 5, 3-regular
        # We need 12 vertices. Can we extend it?
        
        # Alternative: Try random 3-regular graphs and check girth
        for attempt in range(10000):
            if time.time() - start_time_search > 20:
                break
            
            # Generate random 3-regular graph using configuration model
            stubs = [i for i in range(n) for _ in range(3)]
            random.shuffle(stubs)
            
            edges_set = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                e = (min(u, v), max(u, v))
                if e in edges_set:
                    valid = False
                    break
                edges_set.add(e)
            
            if not valid:
                continue
            
            edges = [list(e) for e in edges_set]
            
            if not is_valid(edges):
                continue
            
            # Check girth >= 5 (no 4-cycles)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            has_4cycle = False
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        for w in adj[v]:
                            if w > v:
                                for x in adj[w]:
                                    if x > w and x in adj[u]:
                                        has_4cycle = True
                                        break
                                    if x == u:
                                        has_4cycle = True
                                        break
                            if has_4cycle:
                                break
                    if has_4cycle:
                        break
                if has_4cycle:
                    break
            
            if has_4cycle:
                continue
            
            # Girth >= 5, compute distinct cycle lengths
            lengths = get_cycle_lengths_v2(edges)
            val = len(lengths)
            
            if val < best_value:
                best_value = val
                best_edges = edges
                
                if best_value == 1:
                    return True
        
        return False
    
    # Try girth-5 search first
    if not try_girth5_search():
        # Fall back to bipartite approach
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            
            left = list(range(6))
            right = list(range(6, 12))
            
            left_stubs = [i for i in range(6) for _ in range(3)]
            right_stubs = [i for i in range(6, 12) for _ in range(3)]
            
            random.shuffle(right_stubs)
            
            current_edges = set()
            valid = True
            for u in left_stubs:
                v = right_stubs.pop()
                e = (min(u, v), max(u, v))
                if e in current_edges:
                    valid = False
                    break
                current_edges.add(e)
            
            if not valid:
                continue
            
            edges = [list(e) for e in current_edges]
            
            # Check min degree
            deg = [0] * n
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            if min(deg) < min_deg:
                continue
            
            lengths = get_cycle_lengths_v2(edges)
            
            if not lengths:
                continue
            
            val = len(lengths)
            if val < best_value:
                best_value = val
                best_edges = edges
                
                if best_value == 1:
                    break
    
    if best_edges is None:
        # Fallback: Two disjoint K_{3,3} components
        # K_{3,3} has vertices 0,1,2 and 3,4,5 (first component)
        # and 6,7,8 and 9,10,11 (second component)
        edges = []
        for i in range(3):
            for j in range(3, 6):
                edges.append([i, j])
        for i in range(6, 9):
            for j in range(9, 12):
                edges.append([i, j])
        best_edges = edges
    
    # Ensure return types are correct
    result = []
    for e in best_edges:
        u = int(e[0])
        v = int(e[1])
        result.append([u, v])
    
    return result

# EVOLVE-BLOCK-END
