import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Fallback: LUW construction
    def luw_construct(k, g):
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

    def check_girth(n, edges, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS for girth
        min_girth = float('inf')
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif parent[u] != v and parent[v] != u:
                        # Cycle found
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                        if min_girth < g:
                            return False
        return min_girth >= g

    def generate_random_regular_graph(n, k):
        # Returns adjacency matrix or None if failed
        # Use configuration model
        if n * k % 2 != 0:
            return None
        
        # Create stubs
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        # Pair stubs
        edges = []
        adj = np.zeros((n, n), dtype=int)
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                continue # Loop, invalid for simple graph, retry or skip? 
                         # In configuration model, loops/multi-edges are possible.
                         # We need simple graph.
                         # If u==v or adj[u][v] > 0, it's invalid.
                         # This simple method might fail often for small n/high k.
                         # For k=4, n>10, it's usually okay with retries.
            if adj[u][v] > 0:
                continue
            adj[u][v] = 1
            adj[v][u] = 1
            edges.append((u, v))
        
        # Check if degree is k for all
        for i in range(n):
            if adj[i].sum() != k:
                return None
        
        return adj, edges

    def count_short_cycles(adj, max_len=5):
        # Counts cycles of length 3, 4, 5
        # N3 = trace(A^3) / 6
        # N4 = (trace(A^4) - 2*trace(A^2) + 2*n) / 8 ? 
        # Actually, trace(A^2) = 2m = nk.
        # trace(A^4) counts closed walks of length 4.
        # Each 4-cycle is counted 8 times (4 starts * 2 directions).
        # Walks that go back and forth on an edge: u-v-u-v. 
        # Number of such walks: for each edge (u,v), 2 walks (u-v-u-v, v-u-v-u).
        # Total edges m. So 2m walks of type u-v-u-v.
        # Are there other non-cycle closed walks of length 4?
        # u-v-w-v-u? No, that's backtracking.
        # The formula for number of 4-cycles C4 is:
        # C4 = (trace(A^4) - 2*n*k - 4*m) / 8 ? 
        # Let's use a robust method or just trace for N3 and approximate for N4/N5.
        # For SA, relative change is what matters.
        
        A2 = adj @ adj
        A3 = A2 @ adj
        A4 = A2 @ A2
        A5 = A4 @ adj
        
        n = adj.shape[0]
        
        # N3
        n3 = np.trace(A3) / 6.0
        
        # N4
        # trace(A^4) = 8*C4 + 2*n*k (backtracking walks u-v-u-v)
        # Wait, backtracking walks of length 4: u-v-u-v. 
        # For each directed edge (u,v), there is 1 such walk starting at u? 
        # u->v->u->v. 
        # Number of directed edges is n*k.
        # So 2 * (n*k) walks? No.
        # Walk u-v-u-v. Starts at u, ends at v? No, closed walk.
        # u-v-u-v is not closed unless u=v.
        # Closed walk u-v-u-v-u? Length 4.
        # u-v-u-v. Not closed.
        # Closed walks of length 4:
        # 1. 4-cycles: 8 per cycle.
        # 2. Backtracking: u-v-u-v? No.
        #    u-v-u-x? No.
        #    u-v-u-v is not closed.
        #    u-v-u-v... 
        #    Walks like u-v-u-v are not closed.
        #    Walks like u-v-w-v-u? 
        #    If w=u, u-v-u-v (not closed).
        #    If w!=u, u-v-w-v-u. This is a "triangle with a tail"? 
        #    Actually, any closed walk of length 4 that is not a 4-cycle must backtrack.
        #    Forms: u-v-u-v (not closed), u-v-u-x (not closed unless x=u -> u-v-u-u loop).
        #    The only non-simple closed walks of length 4 in a simple graph are:
        #    - u-v-u-v (not closed)
        #    - u-v-w-v-u where w != u. This is a 3-cycle u-v-w-u with an extra step? No.
        #      u->v->w->v->u. This is a closed walk.
        #      It corresponds to a triangle u-v-w-u.
        #      For each triangle, how many such walks?
        #      u-v-w-v-u. 
        #      Starts at u. 
        #      For a fixed triangle {u,v,w}, there are 3 such walks? 
        #      u-v-w-v-u, v-w-u-w-v, w-u-v-u-w.
        #      Also reverse: u-w-v-w-u.
        #      So 6 walks per triangle?
        #      
        #    Let's check trace(A^4) for a triangle (K3).
        #    A = [[0,1,1],[1,0,1],[1,1,0]].
        #    A^2 = [[2,1,1],[1,2,1],[1,1,2]].
        #    A^4 = A^2 @ A^2.
        #    trace(A^4) = 2^2+1+1 + 1+2^2+1 + 1+1+2^2 = 6+6+6 = 18.
        #    N3 = 1.
        #    If formula C4 = (trace - 2*n*k - 6*N3) / 8?
        #    n=3, k=2. 2*n*k = 12.
        #    18 - 12 - 6*1 = 0. C4=0. Correct.
        #    
        #    So C4 = (trace(A^4) - 2*n*k - 6*N3) / 8.
        
        n4 = (np.trace(A4) - 2*n*k - 6*n3) / 8.0
        
        # N5
        # trace(A^5) = 10 * C5 + ...
        # Walks of length 5 that are not 5-cycles:
        # Must backtrack.
        # This gets complicated.
        # For SA, we can just use N3 and N4.
        # Minimizing N3 and N4 is a good proxy.
        
        return n3, n4

    def try_find_smaller(k, g, max_n):
        # Try n values from max_n-2 down to Moore bound
        r = g // 2
        moore = 2 * ((k-1)**r - 1) // (k-2)
        min_n = max(moore, 2*k)
        
        for n in range(max_n - 2, min_n - 1, -2):
            if time.time() > deadline - 10:
                break
                
            # Try to find a graph of size n
            best_adj = None
            best_cost = float('inf')
            
            # Initialize with random graph
            adj, edges = generate_random_regular_graph(n, k)
            if adj is None:
                continue
            
            n3, n4 = count_short_cycles(adj, k)
            cost = n3 + 2*n4
            
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            alpha = 0.99
            steps = 1000 # Limit steps
            
            for step in range(steps):
                if time.time() > deadline - 10:
                    break
                    
                # Pick random 2-switch
                # Find two edges (u,v) and (x,y)
                # We can pick random u, x and random neighbors v, y
                u = random.randint(0, n-1)
                x = random.randint(0, n-1)
                if u == x:
                    continue
                    
                neighbors_u = np.where(adj[u] > 0)[0]
                neighbors_x = np.where(adj[x] > 0)[0]
                
                if len(neighbors_u) < k or len(neighbors_x) < k:
                    continue
                
                v = neighbors_u[random.randint(0, len(neighbors_u)-1)]
                y = neighbors_x[random.randint(0, len(neighbors_x)-1)]
                
                # Check validity of swap (u,v) and (x,y) -> (u,x) and (v,y)
                # 1. u != v, x != y (guaranteed by simple graph)
                # 2. u != x, v != y (guaranteed by u != x and random pick? No, v could be x? 
                #    If v == x, then edge (u,x) exists. We are trying to add (u,x). 
                #    So we must ensure (u,x) does NOT exist.
                #    Similarly (v,y) must NOT exist.
                #    Also (u,y) and (v,x) should not be removed? We are removing (u,v) and (x,y).
                #    So we need:
                #    adj[u][v] == 1
                #    adj[x][y] == 1
                #    adj[u][x] == 0
                #    adj[v][y] == 0
                #    Also, the new edges must not create loops (u!=x, v!=y) - already checked.
                
                if adj[u][v] != 1 or adj[x][y] != 1:
                    continue
                if adj[u][x] != 0 or adj[v][y] != 0:
                    continue
                
                # Apply swap
                adj[u][v] = 0; adj[v][u] = 0
                adj[x][y] = 0; adj[y][x] = 0
                adj[u][x] = 1; adj[x][u] = 1
                adj[v][y] = 1; adj[y][v] = 1
                
                # Calculate new cost
                new_n3, new_n4 = count_short_cycles(adj, k)
                new_cost = new_n3 + 2*new_n4
                
                delta_cost = new_cost - cost
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    cost = new_cost
                    # Keep swap
                else:
                    # Revert
                    adj[u][v] = 1; adj[v][u] = 1
                    adj[x][y] = 1; adj[y][x] = 1
                    adj[u][x] = 0; adj[x][u] = 0
                    adj[v][y] = 0; adj[y][v] = 0
                    
                T *= alpha
                
                if cost == 0:
                    # Potential success
                    # Verify girth
                    edges_list = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i][j]:
                                edges_list.append((i, j))
                    if check_girth(n, edges_list, g):
                        return n, edges_list
            
        return None

    # Get LUW result
    n_luw, edges_luw = luw_construct(k, g)
    
    # Try to find smaller
    if k == 4 and g == 10:
        result = try_find_smaller(k, g, n_luw)
        if result:
            return result
            
    return n_luw, edges_luw
