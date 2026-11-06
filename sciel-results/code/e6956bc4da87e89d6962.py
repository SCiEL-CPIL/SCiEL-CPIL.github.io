import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    # Helper to generate Polarity Graph of PG(2, q)
    def get_polarity_graph(q):
        # Field F_q
        # For q=4, use GF(4)
        # We need a representation of GF(q)
        
        if q == 4:
            # GF(4) = {0, 1, w, w2} where w^2 = w + 1
            # Represent elements as integers 0,1,2,3
            # 0:0, 1:1, 2:w, 3:w2
            def add(a, b):
                return a ^ b # XOR for GF(2^k)
            
            def mul(a, b):
                if a == 0 or b == 0: return 0
                # w=2, w2=3
                # 1*1=1
                # 1*2=2
                # 1*3=3
                # 2*2=3 (w^2)
                # 2*3=2 (w^3 = w*w^2 = w^3 = (w+1)w = w^2+w = (w+1)+w = 1? No. w^3 = w*(w+1) = w^2+w = (w+1)+w = 1. Wait.
                # In GF(4), w^2+w+1=0 => w^2=w+1.
                # w^3 = w(w+1) = w^2+w = (w+1)+w = 1.
                # So 2*3 = 2*2*2 = 3*2 = 2*3.
                # 2*2=3.
                # 2*3 = 2*(2*2) = (2*2)*2 = 3*2.
                # 3*2 = (2*2)*2 = 2*(2*2) = 2*3.
                # Let's make a table.
                # 0:0, 1:1, 2:w, 3:w^2
                # 1*w = w (2)
                # 1*w^2 = w^2 (3)
                # w*w = w^2 (3)
                # w*w^2 = w^3 = 1
                # w^2*w^2 = w^4 = w
                mult_table = [
                    [0, 1, 2, 3],
                    [1, 1, 2, 3],
                    [2, 2, 3, 1],
                    [3, 3, 1, 2]
                ]
                return mult_table[a][b]
            
            def neg(a):
                return a # char 2
            
            def inv(a):
                if a == 1: return 1
                if a == 2: return 3 # w * w^2 = 1
                if a == 3: return 2
                return 0
            
            # Points of PG(2,4): non-zero vectors in F4^3, modulo scaling
            points = []
            seen = set()
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        if x==0 and y==0 and z==0: continue
                        # Normalize: make first non-zero component 1
                        # Since char 2, 1 is the only non-zero scalar that is 1? No, scalars are 1, w, w^2.
                        # We can just iterate and add if not seen
                        vec = (x, y, z)
                        # Check if proportional to existing
                        is_new = True
                        for p in points:
                            # p ~ vec?
                            # There exists k != 0 such that p = k * vec
                            # Since field is small, just check
                            for k in [1, 2, 3]:
                                if (mul(k, p[0]) == vec[0] and mul(k, p[1]) == vec[1] and mul(k, p[2]) == vec[2]):
                                    is_new = False
                                    break
                            if not is_new: break
                        if is_new:
                            points.append(vec)
            
            n = len(points)
            # Polarity: x . y = x[0]y[2] + x[1]y[1] + x[2]y[0]
            # In GF(4), this is symmetric.
            # Is it alternating? x . x = 2 x[0]x[2] + x[1]^2.
            # If x[0]=1, x[1]=0, x[2]=0, x.x = 0.
            # If x[0]=0, x[1]=1, x[2]=0, x.x = 1.
            # So not all points are absolute.
            # Adjacency: u ~ v if dot(u, v) == 0.
            
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    u = points[i]
                    v = points[j]
                    dot = add(add(mul(u[0], v[2]), mul(u[1], v[1])), mul(u[2], v[0]))
                    if dot == 0:
                        edges.append([i, j])
            
            return n, edges

        else:
            return 0, []

    # Try q=4
    n4, edges4 = get_polarity_graph(4)
    
    # Check independence number of G4
    # Use a simple greedy or branch and bound if n is small
    def max_independent_set(n, edges):
        # Adjacency matrix
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
            
        # Branch and bound
        best = 0
        # Order vertices by degree
        degrees = [sum(adj[i][j] for j in range(n)) for i in range(n)]
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # Simple recursive search with pruning
        # Given small n (21), this is fast
        def dfs(idx, current_set, size):
            nonlocal best
            if size > best:
                best = size
            
            if idx == n:
                return
                
            # Pruning: even if we add all remaining, can we beat best?
            if size + (n - idx) <= best:
                return
            
            v = order[idx]
            
            # Option 1: Include v
            if True: # v is always a candidate if not conflicting
                # Check if v conflicts with any in current_set
                conflict = False
                for u in current_set:
                    if adj[v][u]:
                        conflict = True
                        break
                if not conflict:
                    dfs(idx + 1, current_set + [v], size + 1)
            
            # Option 2: Exclude v
            dfs(idx + 1, current_set, size)

        dfs(0, [], 0)
        return best

    alpha4 = max_independent_set(n4, edges4)
    
    # If alpha4 <= 5, we can use two copies
    if alpha4 <= 5:
        # Construct two disjoint copies
        n_total = 2 * n4
        edges_total = []
        for u, v in edges4:
            edges_total.append([u, v])
            edges_total.append([u + n4, v + n4])
        
        return {'n': n_total, 'edges': edges_total}
    else:
        # Fall back to seed or try q=5
        # For now, return seed if q=4 fails alpha check
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
