import time
import itertools

def construct():
    """
    Constructs a graph on vertices 0..11 with minimum degree 3,
    triangle-free, minimizing distinct simple cycle lengths.
    
    Strategy: Search for a connected 3-regular bipartite graph on 12 vertices
    that has only one distinct cycle length (ideally 4). If found, raw value = 1.
    Otherwise, fall back to two disjoint K_{3,3} components (raw value = 2).
    
    A 3-regular bipartite graph on 12 vertices has 18 edges.
    We search over all 3-regular bipartite graphs (partitions of 6+6) 
    and check cycle lengths.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    # Fallback: two disjoint K_{3,3}
    def make_fallback():
        edges = []
        # Component 1: K_{3,3} on vertices 0,1,2 (left) and 3,4,5 (right)
        for i in range(3):
            for j in range(3):
                edges.append([int(i), int(3 + j)])
        # Component 2: K_{3,3} on vertices 6,7,8 (left) and 9,10,11 (right)
        for i in range(3):
            for j in range(3):
                edges.append([int(6 + i), int(9 + j)])
        return edges
    
    # Try to find a connected 3-regular bipartite graph with minimal distinct cycle lengths
    # Left partition: {0,1,2,3,4,5}, Right partition: {6,7,8,9,10,11}
    # We need to assign to each left vertex 3 right vertices, and vice versa, such that
    # the resulting graph is 3-regular bipartite.
    
    # Represent the bipartite graph by the adjacency matrix A (6x6) where A[i][j]=1 means edge.
    # Each row and column must have exactly 3 ones.
    
    # Generate all 6x6 binary matrices with row sum 3 and col sum 3.
    # This is the set of 3-regular bipartite graphs on K_{6,6}.
    # We can generate these by enumerating all subsets of size 3 for each row,
    # and checking column sums. But that's C(6,3)^6 = 20^6 = 64 million, too many.
    
    # Better approach: use networkx or a smarter enumeration.
    # Actually, let's use a different strategy: enumerate all 3-regular bipartite graphs
    # using a backtracking approach with pruning.
    
    # Alternatively, we can use the fact that there are only a few isomorphism classes.
    # But let's just do a randomized search with a time limit.
    
    import random
    
    best_raw = 2
    best_edges = None
    
    # Random search: generate random 3-regular bipartite graphs
    for _ in range(50000):
        if time.time() > deadline:
            break
        
        # Generate random 3-regular bipartite graph
        # Start with random permutation-based construction
        # Use the configuration model: 3 stubs on each left vertex, 3 on each right
        left_stubs = [(i, k) for i in range(6) for k in range(3)]
        right_stubs = [(6 + j, k) for j in range(6) for k in range(3)]
        
        random.shuffle(right_stubs)
        
        # Pair left stubs with right stubs
        edges_set = set()
        valid = True
        for (lu, _), (rv, _) in zip(left_stubs, right_stubs):
            if lu == rv:
                valid = False
                break
            if (lu, rv) in edges_set or (rv, lu) in edges_set:
                valid = False
                break
            edges_set.add((lu, rv))
        
        if not valid:
            continue
        
        # Check if it's a simple graph (no multiple edges) - already checked
        # Check connectivity
        # Build adjacency list
        adj = {i: [] for i in range(12)}
        for u, v in edges_set:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check connectivity
        visited = set()
        stack = [0]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        if len(visited) != 12:
            continue
        
        # Find all simple cycles and their lengths
        # For a small graph, we can use a DFS-based cycle enumeration
        # But this might be slow. Let's use a simpler approach:
        # Count cycles of each length using adjacency matrix powers? No, that counts walks.
        # Let's just enumerate cycles up to length 12.
        
        cycle_lengths = set()
        
        # Enumerate cycles by starting from each vertex and doing DFS
        # To avoid duplicates, only start from the smallest vertex in the cycle
        # and only go in one direction
        
        def find_cycles(adj, n):
            lengths = set()
            for start in range(n):
                # DFS to find cycles starting and ending at start
                # Only visit vertices > start to avoid duplicates
                stack = [(start, [start], {start})]
                while stack:
                    node, path, visited = stack.pop()
                    for neighbor in adj[node]:
                        if neighbor == start and len(path) >= 3:
                            lengths.add(len(path))
                        elif neighbor > start and neighbor not in visited:
                            stack.append((neighbor, path + [neighbor], visited | {neighbor}))
            return lengths
        
        cycle_lengths = find_cycles(adj, 12)
        
        if len(cycle_lengths) < best_raw:
            best_raw = len(cycle_lengths)
            best_edges = sorted([list(e) for e in edges_set])
            if best_raw == 1:
                break
    
    if best_edges is not None:
        # Ensure all elements are Python int
        result = []
        for e in best_edges:
            result.append([int(e[0]), int(e[1])])
        return result
    else:
        return make_fallback()
