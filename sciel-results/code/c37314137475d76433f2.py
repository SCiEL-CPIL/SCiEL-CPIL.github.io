import time
from collections import defaultdict
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 12
    
    def get_cycle_lengths(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cycle_lengths = set()
        
        def find_cycles_from(start):
            found = []
            def dfs(node, path, visited):
                for neighbor in sorted(adj[node]):
                    if neighbor == start and len(path) >= 4:  # Girth >= 4, so no triangles
                        found.append(len(path))
                    elif neighbor not in visited and neighbor > start:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, path, visited)
                        path.pop()
                        visited.discard(neighbor)
            dfs(start, [start], {start})
            return found
        
        for v in range(n):
            for c in find_cycles_from(v):
                cycle_lengths.add(c)
        
        return cycle_lengths
    
    def is_valid(edges):
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u,v), max(u,v))
            if key in edge_set:
                return False
            edge_set.add(key)
        
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        for d in deg:
            if d < 3:
                return False
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check triangle-free
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False
        
        # Check has cycle (using Union-Find)
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return True
            parent[px] = py
            return False
        
        has_cycle = False
        for u, v in edges:
            if union(u, v):
                has_cycle = True
                break
        
        return has_cycle
    
    def make_k33(offset):
        edges = []
        for i in range(3):
            for j in range(3):
                edges.append([offset + i, offset + 3 + j])
        return edges
    
    # Start with known good: two K3,3 components
    best_edges = make_k33(0) + make_k33(6)
    best_score = 2  # K3,3 has cycle lengths {4, 6}
    
    # Verify the initial solution is valid
    if not is_valid(best_edges):
        # Fallback: construct manually
        best_edges = []
        for i in range(3):
            for j in range(3):
                best_edges.append([i, 3+j])
                best_edges.append([6+i, 9+j])
        best_score = 2
    
    # Try to find a graph with 1 distinct cycle length
    # Strategy: search over 3-regular bipartite graphs on 12 vertices
    # Parts: A={0..5}, B={6..11}
    # A 3-regular bipartite graph can be represented by a 6x6 0-1 matrix with row/col sums 3
    
    random.seed(42)
    
    # Generate random 3-regular bipartite graphs
    # Method: start with a base permutation structure and modify
    
    for attempt in range(50000):
        if time.time() > deadline - 5:
            break
        
        # Generate random 3-regular bipartite graph
        # Use random permutation approach: each vertex in A connects to 3 vertices in B
        # We can represent this as a 3-regular bipartite graph
        
        # Method 1: Random 3-regular bipartite via configuration model
        # A vertices: 0-5, B vertices: 6-11
        # Each A vertex has degree 3, each B vertex has degree 3
        
        # Use a simple approach: generate random permutation and take 3 "shifts"
        perm = list(range(6))
        random.shuffle(perm)
        
        edges = []
        for i in range(6):
            for j in [0, 1, 2]:
                a = i
                b = 6 + perm[(i + j) % 6]
                edges.append([a, b])
        
        edge_set = set()
        valid = True
        for e in edges:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
        
        if not valid:
            continue
        
        edges = [list(e) for e in edge_set]
        
        if not is_valid(edges):
            continue
        
        cl = get_cycle_lengths(edges)
        if not cl:
            continue
        
        score = len(cl)
        if score < best_score:
            best_score = score
            best_edges = edges
            if score == 1:
                break
    
    # Also try: connected 3-regular bipartite graphs with specific structure
    # The prism graph Y6 (cubical prism) has cycle lengths {4, 6}
    # But what about other structures?
    
    # Try: single 3-regular bipartite graph on 12 vertices that is connected
    # and see if we can find one with only 4-cycles or only 6-cycles
    
    # A graph with only 4-cycles would be a "4-cycle graph" - but 3-regular bipartite
    # with only 4-cycles... Let's think.
    
    # Actually, let's try a different approach: enumerate small structures
    # The K3,3 has girth 4 and cycles of length 4 and 6.
    
    # What about the utility graph variants?
    
    # Let's also try: two disjoint 6-cycles with cross edges to make it 3-regular
    # Two 6-cycles: (0,1,2,3,4,5) and (6,7,8,9,10,11)
    # Add cross edges to make degree 3
    
    # This is getting complex. Let's stick with the random search.
    
    # Final: ensure types are correct
    result = [list(e) for e in sorted(set(tuple(sorted(e)) for e in best_edges))]
    
    return result
