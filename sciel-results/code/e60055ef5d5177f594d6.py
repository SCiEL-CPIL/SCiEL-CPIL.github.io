import random
import time
import sys
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Combines cyclic (Z_n-regular) construction with proper non-backtracking
    girth verification. Falls back to LW graph if needed.
    """
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill

    def check_girth_nb_fast(adj, n, g, start_vertex=0):
        """
        Check girth using non-backtracking walks from a single start vertex.
        Optimized for cyclic graphs where girth is vertex-transitive.
        Returns True if no cycle shorter than g exists starting from start_vertex.
        """
        # We only need to check from start_vertex for vertex-transitive graphs (like cyclic ones).
        # If a short cycle exists, it will be found from any vertex due to symmetry.
        
        for first_neighbor in adj[start_vertex]:
            # State: (node, parent)
            # We use a list of booleans or an integer array for visited.
            # State index: node * n + parent
            visited = [-1] * (n * n)
            queue = deque()
            queue.append((first_neighbor, start_vertex, 1))
            visited[first_neighbor * n + start_vertex] = 1
            
            found_short_cycle = False
            while queue:
                curr, prev, depth = queue.popleft()
                
                # Check if we can close the cycle to 'start_vertex'
                # Edge (curr, start_vertex) is valid if start_vertex != prev
                if start_vertex != prev:
                    # Check if edge (curr, start_vertex) exists
                    # In a regular graph, we can just check if start_vertex is in adj[curr]
                    # But for speed, we assume the graph is simple and regular.
                    # We need to verify the edge exists.
                    # Since it's a cyclic graph, we know the edges.
                    # But adj is a list of lists. Checking `start_vertex in adj[curr]` is O(k).
                    # k=4, so it's fast.
                    if start_vertex in adj[curr]:
                        cycle_len = depth + 1
                        if cycle_len < g:
                            return False
                
                # Expand
                for nxt in adj[curr]:
                    if nxt == prev:
                        continue
                    state_idx = nxt * n + curr
                    if visited[state_idx] == 0:
                        visited[state_idx] = 1
                        queue.append((nxt, curr, depth + 1))
                        
                # Pruning: if depth + 1 >= g, we don't need to go deeper
                # because any cycle closed from a deeper node would be longer
                if depth + 1 >= g:
                    # Actually, if depth+1 >= g, the next step would be depth+2.
                    # If we are at depth d, the shortest cycle closing from here is d+1.
                    # If d+1 >= g, then any cycle closing from a child (depth d+1) would be d+2 >= g+1.
                    # So we can stop expanding this branch.
                    # But we already checked the current node.
                    # So we don't add children.
                    continue

        return True

    def make_cyclic_graph(n, offsets):
        """
        Create a cyclic Z_n-regular graph with given offsets.
        """
        edges = []
        edge_set = set()
        
        if 0 in offsets:
            return None
        if len(set(o % n for o in offsets)) < len(offsets):
            return None
        
        for i in range(len(offsets)):
            for j in range(i+1, len(offsets)):
                if (offsets[i] + offsets[j]) % n == 0:
                    return None
        
        for i in range(n):
            for off in offsets:
                j = (i + off) % n
                if i == j:
                    return None
                key = (min(i, j), max(i, j))
                if key in edge_set:
                    return None
                edge_set.add(key)
                edges.append((i, j))
        
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if any(d != k for d in degree):
            return None
            
        return edges

    def check_girth_from_edges_fast(edges, n, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # For cyclic graphs, checking from vertex 0 is sufficient
        return check_girth_nb_fast(adj, n, g, 0)

    best_n = float('inf')
    best_edges = None

    # Focus on n=320 as it is the known minimum.
    # Try n=320 with high attempts first.
    # Then try nearby values.
    
    target_n = 320
    # Range to search
    # If we find a graph at 320, it's optimal (given lower bound 243 and known 320).
    # We should verify if 320 is the absolute minimum for this specific construction.
    # The problem states "smallest known such graph has 320 vertices".
    # So we aim for 320.
    
    # Strategy:
    # 1. Try n=320 with many random attempts.
    # 2. If not found, try n=322, 324, etc.
    # 3. If not found, fallback.
    
    # To be safe, we can also try a few values below 320 with low attempts.
    search_ns = [320] + list(range(322, 350, 2)) + list(range(243, 320, 10))
    # Prioritize 320
    
    for n in search_ns:
        if time.time() > deadline - 5:
            break
            
        if n == 320:
            attempts = 20000
        elif n < 320:
            attempts = 100
        else:
            attempts = 1000

        for _ in range(attempts):
            if time.time() > deadline - 2:
                break
                
            offsets = []
            while len(offsets) < 4:
                o = random.randint(1, n-1)
                if o in offsets:
                    continue
                valid = True
                for existing in offsets:
                    if (o + existing) % n == 0:
                        valid = False
                        break
                if valid:
                    offsets.append(o)
            
            edges = make_cyclic_graph(n, offsets)
            if edges is not None:
                if check_girth_from_edges_fast(edges, n, g):
                    return n, edges

    # Fallback to LW graph
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
