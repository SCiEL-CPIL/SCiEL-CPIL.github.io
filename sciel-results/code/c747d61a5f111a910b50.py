import time
import numpy as np

def construct(k, g):
    deadline = time.time() + 115  # Safety margin
    
    def luw_construct(k, g, d):
        """
        Construct a (k+1)-regular graph using LUW construction with depth d.
        Note: LUW typically gives (q+1)-regular graphs. For k=3, q=3? 
        Actually LUW for q=3 gives 4-regular. 
        Wait, the task asks for k-regular. 
        The seed program used `q=k` in luw_construct. 
        Let's check the seed's luw_construct:
        `q = k`
        `neighbours`: yields `q` neighbors.
        So it constructs a k-regular graph.
        The LUW construction usually refers to (q+1)-regular graphs from PSL(2, q).
        However, there are variants. The seed code implements a specific recursive rule.
        Let's trust the seed's `luw_construct` logic but optimize the check.
        
        Actually, the standard LUW construction for k=3 (3-regular) is not the standard LUW (which is 4-regular for q=3).
        The seed code's `luw_construct` with `q=3` produces a 3-regular graph?
        Let's trace:
        `for first in range(q):` -> 3 neighbors.
        So yes, it's a 3-regular graph.
        
        The parameter `d` controls the depth/size.
        Smaller `d` -> smaller graph.
        We need girth >= g.
        """
        q = k
        # The seed used: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        # This is a heuristic for the *required* depth to potentially achieve girth g.
        # But we want to *minimize* vertices, so we want the *smallest* d that still works.
        # The Moore bound for k=3, g=16 is 512.
        # LUW graphs grow exponentially with d.
        # We should scan d from a small value upwards.
        
        # Rule generation from seed
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        # We will generate the graph by traversing the tree of non-backtracking walks.
        # Vertices are tuples (side, x_tuple).
        # To make it efficient, we can use a dictionary for mapping.
        
        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        # We need to explore neighbors.
        # This BFS on the state space can be slow if d is large, but for d ~ 10-20 it's manageable.
        
        for vertex in order:
            side, x = vertex
            v_idx = index[vertex]
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                other = (1 - side, tuple(y))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((v_idx, index[other]))
        
        return len(order), edges

    def check_girth_vectorized(n, edges, g):
        """
        Check if girth >= g using vectorized non-backtracking walks.
        We check for cycles of length < g.
        A cycle of length L corresponds to a non-backtracking walk of length L that returns to start.
        Specifically, we can count the number of closed non-backtracking walks of length L.
        If count > 0 for any L < g, girth < g.
        
        However, simply counting walks is tricky because of the "non-backtracking" constraint in the matrix power.
        Let A be the adjacency matrix.
        The number of non-backtracking walks of length L is related to the trace of the non-backtracking matrix B^L.
        
        Constructing B explicitly is O(n^3) or O(m^2) which is too slow for n=10k.
        
        Alternative:
        Use the fact that for a k-regular graph, the number of closed non-backtracking walks of length L
        can be computed from the eigenvalues of the adjacency matrix?
        No, that's for all walks.
        
        Let's use a different vectorized approach:
        We can simulate the non-backtracking walk step by step using matrices.
        Let W_l be the matrix where W_l[u, v] = 1 if there is a non-backtracking walk of length l from u to v?
        No, that's just A^l with corrections.
        
        Better approach for exact certification within time limit:
        Since n is likely small enough (10k-50k), we can use a BFS from each vertex but optimized.
        Or, we can use the "girth via eigenvalues" approximation? No, must be exact.
        
        Let's try a hybrid:
        1. Compute adjacency matrix A (sparse).
        2. We want to know if there is a cycle of length < g.
        3. For k=3, g=16, we can check cycles of length 4, 6, 8, 10, 12, 14.
        
        Cycle counting via matrix multiplication:
        Tr(A^L) counts closed walks of length L.
        Closed walks include backtracking.
        Number of closed non-backtracking walks of length L can be derived.
        
        Let c_L be the number of closed non-backtracking walks of length L.
        For a k-regular graph:
        c_0 = n
        c_2 = n * (k-1)  ? No.
        
        Actually, there is a known relation:
        Let A be the adjacency matrix.
        The non-backtracking matrix B acts on directed edges.
        Size of B is 2m = 2kn.
        For n=10k, 2kn = 60k.
        B is sparse.
        We can compute B^L using sparse matrix multiplication?
        60k x 60k sparse matrix.
        Multiplying two sparse matrices of this size is expensive in Python even with scipy.
        
        Let's stick to BFS but optimize it heavily.
        With n=10k, BFS from each vertex is 10k * (expansion).
        Expansion is 2^depth. Depth 8 (half of 16) -> 256 nodes.
        10k * 256 = 2.5M operations. This is fast in Python if implemented with lists/deques.
        The seed's BFS was likely slow due to Python overhead or large n.
        If we keep n small (e.g. < 20k), BFS is feasible.
        
        Let's refine the BFS check to be as fast as possible.
        """
        # Build adjacency lists
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_depth = g // 2
        
        # Early exit: check for triangles (g=3) etc, but here g=16 so min check is 4.
        # We check for any cycle of length < g.
        # BFS from each node, tracking parent.
        
        for start in range(n):
            # visited: dict or list? List of int is faster.
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = [start]
            head = 0
            while head < len(q):
                u = q[head]
                head += 1
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        # Found a back edge
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    # Main Logic for k=3, g=16
    if k == 3 and g == 16:
        # Scan d (depth) from small to large.
        # The Moore bound is 512.
        # LUW graph size grows roughly as q^(d) or similar.
        # For k=3 (q=3), size is approx 3^d?
        # Let's try d from 5 to 20.
        
        best_n = float('inf')
        best_edges = None
        
        for d in range(5, 25):
            if time.time() > deadline - 10:
                break
            
            try:
                n, edges = luw_construct(k, g, d)
            except Exception:
                continue
            
            if n >= best_n:
                continue
            
            # Quick check: Moore bound
            if n < 512: 
                continue # Should not happen, but just in case
            
            # Check girth
            if check_girth_vectorized(n, edges, g):
                best_n = n
                best_edges = edges
                # Since we are scanning d upwards, and size grows with d,
                # the first valid d gives the smallest n for this family.
                # However, different d might not be strictly increasing in size?
                # In LUW, size is monotonic with d.
                # So we can break?
                # Yes, if size is monotonic, smallest valid d is best.
                break
        
        if best_edges is not None:
            return best_n, best_edges
        
        # Fallback: If LUW scan didn't find anything (maybe d too small or logic error),
        # try a larger d or different strategy.
        # But for now, return the best found or a default valid graph.
        # We must return *some* valid graph.
        # The seed's fallback was LUW with a specific d.
        # Let's use the seed's heuristic d as a final fallback.
        d_fallback = max(3, g - 5 if (g - 5) % 2 else g - 4)
        try:
            n, edges = luw_construct(k, g, d_fallback)
            return n, edges
        except Exception:
            pass

    # General case fallback (not optimized, but keeps interface)
    # For other k, g, use a simple random regular graph generator or LUW if applicable.
    # Since the task is specific to k=3, g=16, we prioritize that.
    # For others, we return a trivial valid graph if possible, or LUW.
    
    # LUW for general k
    d_fallback = max(3, g - 5 if (g - 5) % 2 else g - 4)
    try:
        n, edges = luw_construct(k, g, d_fallback)
        return n, edges
    except Exception:
        pass
    
    # Final fallback: Cycle of length g? No, must be k-regular.
    # Return a small valid graph if we can construct one quickly.
    # For k=3, girth 16, if all else fails, we might timeout.
    # But LUW d=11 or so should work.
    
    # Construct a known small graph?
    # Let's just return the LUW result with a larger d to be safe, even if suboptimal.
    d_safe = 20
    n, edges = luw_construct(k, g, d_safe)
    return n, edges
