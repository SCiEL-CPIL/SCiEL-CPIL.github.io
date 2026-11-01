import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    if k != 3:
        # Fallback for other k
        return _luw_construct(k, g)

    # Target girth g (typically 16)
    # Moore bound for k=3, g=16 is 512
    # We use Petersen graph (10 vertices) as base
    # Lift by m, N = 10m
    
    # Petersen graph edges
    # Vertices 0..4 outer, 5..9 inner
    # Outer cycle: 0-1-2-3-4-0
    # Inner star: 5-7-9-6-8-5
    # Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Map edge to index
    edge_index = {}
    for i, (u, v) in enumerate(petersen_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    # Adjacency list for Petersen
    adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Precompute all non-backtracking closed walks of length < g
    # We need to check lengths 3 to g-1
    # Petersen has girth 5, so no walks of length 3, 4
    # We check 5 to g-1
    
    walks = [] # List of edge index sequences
    max_len = g - 1
    
    # BFS for non-backtracking walks
    # State: (current_vertex, previous_vertex, length, path_edges)
    # We only care about closed walks (current_vertex == start_vertex)
    # But we can just generate all walks and check if they close
    
    # To avoid duplicates and explosion, we can use the fact that 
    # number of walks is manageable for L=15.
    
    # Start from each vertex
    for start in range(10):
        # Queue: (v, prev, length, path)
        # We don't need to store full path if we just want to check sums later?
        # No, we need the edge sequence to build the matrix A.
        
        # Actually, we can just store the edge sequence for each walk.
        
        dq = deque()
        # Initial moves from start
        for w in adj[start]:
            dq.append((start, w, 1, [edge_index[(start, w)]]))
            
        while dq:
            v, prev, length, path = dq.popleft()
            
            if length >= max_len:
                continue
                
            # Check if this is a closed walk
            # Note: We started at 'start', moved to 'prev'.
            # If we are back at 'start', it's closed.
            # But we must ensure it's non-backtracking.
            # The path is a sequence of edges.
            # If v == start and length >= 3, it's a candidate.
            # However, the walk must not backtrack.
            # Our generation ensures non-backtracking by not returning to prev.
            
            if v == start and length >= 3:
                # Add to walks
                # We only need to store unique walks?
                # Different paths might have same edge multiset but different order.
                # The sum depends on the order if voltages are not commutative?
                # Voltages are in Z_m (commutative group), so order doesn't matter for sum!
                # Wait, voltage assignment is on edges.
                # Sum is sum of voltages of edges in the walk.
                # Since addition is commutative, the sum is just the sum of voltages
                # of the edges with multiplicity.
                # So we just need the count of each edge in the walk.
                
                # So we can represent a walk by a vector of edge counts.
                # This reduces the number of unique "walk types".
                
                pass # We'll handle this below
            
            for w in adj[v]:
                if w == prev:
                    continue
                dq.append((v, w, length + 1, path + [edge_index[(v, w)]]))

    # This approach is slow and memory heavy.
    # Better: Precompute the count vectors for all possible non-backtracking closed walks.
    
    # Let's use a different strategy:
    # Since we just need to check if sum(v_e * count_e) % m == 0,
    # and the number of edges is small (15), we can just iterate through 
    # all possible "short" walks in the base graph during the random search?
    # No, precomputation is better.
    
    # Let's optimize the walk generation.
    # We only need the edge count vector.
    
    walk_vectors = set()
    
    for start in range(10):
        # DFS/BFS to find all non-backtracking walks starting and ending at start
        # Length up to max_len
        def dfs(v, prev, length, counts):
            if length > max_len:
                return
            if v == start and length >= 3:
                # Add a copy of counts
                walk_vectors.add(tuple(counts))
            
            for w in adj[v]:
                if w == prev:
                    continue
                counts[edge_index[(v, w)]] += 1
                dfs(v, w, length + 1, counts)
                counts[edge_index[(v, w)]] -= 1
        
        init_counts = [0] * 15
        for w in adj[start]:
            init_counts[edge_index[(start, w)]] += 1
            dfs(start, w, 1, init_counts)
            init_counts[edge_index[(start, w)]] -= 1
            
    # Convert to NumPy array
    if not walk_vectors:
        # Should not happen for Petersen
        pass
        
    A = np.array(list(walk_vectors), dtype=np.int32)
    # A is (N_walks, 15)
    
    # Now search for m and voltages
    best_n = float('inf')
    best_edges = None
    
    # Moore bound is 512. N = 10m.
    # m must be at least 52.
    # Try m from 52 upwards.
    
    for m in range(52, 100):
        if time.time() > deadline - 10:
            break
        n = 10 * m
        if n >= best_n:
            continue
            
        # Try random voltage assignments
        # We want to avoid any walk sum being 0 mod m.
        
        # Number of walks is len(A)
        # For each assignment, we compute A @ voltages % m
        # and check if any is 0.
        
        # To speed up, we can check a subset of walks first?
        # No, all walks must be non-zero.
        
        max_attempts = 1000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            rng = np.random.RandomState(attempt * m + start)
            # Voltages should be non-zero?
            # If a voltage is 0, it doesn't contribute.
            # But if all are 0, sum is 0.
            # We should avoid 0 voltages to increase "randomness" of sums?
            # Actually, random in 1..m-1 is better.
            
            voltages = rng.randint(1, m, size=15)
            
            # Compute sums
            sums = (A @ voltages) % m
            
            # Check if any sum is 0
            if np.any(sums == 0):
                continue
            
            # If no sum is 0, this is a valid lift!
            # Construct the graph
            edges = []
            for i, (u, v) in enumerate(petersen_edges):
                vol = voltages[i]
                for k in range(m):
                    a = u * m + k
                    b = v * m + (k + vol) % m
                    edges.append((a, b))
                    
            best_n = n
            best_edges = edges
            return n, edges
            
    # Fallback: LUW
    if best_edges is not None:
        return best_n, best_edges
        
    n_luw, edges_luw = _luw_construct(k, g)
    return n_luw, edges_luw

def _luw_construct(k, g, d_override=None):
    """LUW construction with optional d override."""
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
    else:
        add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
