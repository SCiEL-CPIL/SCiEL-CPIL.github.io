import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    def verify_graph(n, edges, k, g):
        if n == 0:
            return False
        adj = [[] for _ in range(n)]
        seen_edges = set()
        for u, v in edges:
            if u == v or u < 0 or u >= n or v < 0 or v >= n:
                return False
            key = (min(u, v), max(u, v))
            if key in seen_edges:
                return False
            seen_edges.add(key)
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        max_depth = (g - 1) // 2
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([(u, -1)])
            while q:
                curr, par = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                for v in adj[curr]:
                    if v == par:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append((v, curr))
                    else:
                        if v == parent[curr]:
                            continue
                        cycle_len = dist[curr] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def build_luw_graph(k, d):
        """
        Constructs the Lazebnik-Ustimenko-Woldar graph LUW(k, d).
        Returns (n, edges).
        n = 2 * k**d
        """
        # LUW construction details:
        # Vertices are pairs (side, vector) where side in {0,1} and vector in (Z_k)^d.
        # Actually, for k=4, it's often defined over Z_k or similar.
        # The standard LUW graph is defined for any integer k >= 2.
        
        # The recurrence for LUW(k,d):
        # Let V = {0,1} x Z_k^d.
        # Neighbors of (0, x) are (1, y) where y is computed from x.
        # Neighbors of (1, x) are (0, y) where y is computed from x.
        
        # The specific rule for LUW(k,d) involves a sequence of operations.
        # For k=4, the operations are addition, subtraction, and multiplication by 1, 2, 3 in Z_4?
        # Actually, the general LUW construction uses a "rule" sequence.
        # The girth is at least 2d+2.
        
        # Implementation of LUW for general k:
        # We use the definition from Lazebnik, Ustimenko, Woldar (1998).
        # The graph is defined by a sequence of d operations.
        # Each operation is of the form: y_i = x_i + x_j * x_l (or similar).
        # For k=4, we can use the standard Z_4 arithmetic.
        
        # A simpler specific construction for k=4 is the "4-ary" LUW.
        # Let's implement the general LUW(k,d) using the standard recurrence.
        
        # The LUW graph G(k,d) has vertices (s, x) with s in {0,1}, x in Z_k^d.
        # The neighbors of (0, x) are (1, f(x)) and the neighbors of (1, x) are (0, g(x)).
        # Actually, it's a bipartite graph between V0 and V1.
        # Each vertex has degree k.
        
        # The mapping is defined by a sequence of d steps.
        # At step i, we compute a new component.
        
        # Let's use the specific construction for k=4 which is well-documented.
        # For k=4, the operations are:
        # 1. y_1 = x_1
        # 2. y_2 = x_2 + x_1 * x_1 (or similar)
        
        # Actually, a robust way is to use the "Cayley graph" interpretation or the explicit formula.
        # Formula for LUW(k,d):
        # Vertices: (epsilon, x_1, ..., x_d) with epsilon in {0,1}, x_i in Z_k.
        # Edges:
        # (0, x) is connected to (1, y) where:
        # y_1 = x_1
        # y_2 = x_2 + x_1 * x_1
        # y_3 = x_3 + x_2 * x_2
        # ...
        # y_i = x_i + x_{i-1} * x_{i-1}
        
        # Wait, this specific recurrence might not yield girth 2d+2 for all k.
        # The standard LUW uses a more complex recurrence involving a sequence of coefficients.
        
        # Let's stick to the implementation in the seed but generalize the parameter d.
        # The seed implementation:
        # if q_val == 4:
        #   table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]] # Multiplication table for Z4? No, this is XOR?
        #   0^0=0, 0^1=1... Yes, it's XOR.
        #   add = lambda x,y: x^y
        #   sub = lambda x,y: x^y
        #   mul = lambda x,y: table[x][y] # This is not standard Z4 mul.
        #   Z4 mul: 1*1=1, 1*2=2, 2*2=0.
        #   Table:
        #   0: 0,0,0,0
        #   1: 0,1,2,3
        #   2: 0,2,3,1  <- 2*2=3? No.
        #   This table looks like:
        #   0*0=0
        #   1*1=1, 1*2=2, 1*3=3
        #   2*2=3, 2*3=1
        #   3*3=2
        #   This is multiplication in Z4?
        #   2*2=4=0 in Z4. Here it is 3.
        #   So it's not Z4.
        
        # Let's look at the rule in the seed:
        # rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        # This rule defines the recurrence for the LUW graph.
        
        # I will reuse the seed's `build_luw_graph` logic but allow `d_val` to be 4.
        
        q_val = k
        d_val = d
        
        if q_val == 4:
            # Reuse the specific table from the seed which worked for d=5
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x,y: x^y
            sub = lambda x,y: x^y
            mul = lambda x,y: table[x][y]
        else:
            add = lambda x,y: (x+y)%q_val
            sub = lambda x,y: (x-y)%q_val
            mul = lambda x,y: x*y%q_val
            
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        def get_rule(dv):
            r = dict(rule)
            for c in range(5, dv+1):
                rc = (c-5)%4
                c0 = c-(c-5)%4
                if rc==0: r[c]=(1,c0-2)
                elif rc==1: r[c]=(c0-1,1)
                elif rc==2: r[c]=(c0,1)
                else: r[c]=(1,c0+1)
            return r
            
        r = get_rule(d_val)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0]*(d_val-1)
                for c in range(2, d_val+1):
                    a,b = r[c]
                    if side==0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
                
        start = (0, (None,)+(0,)*d_val)
        index = {start: 0}
        order = [start]
        edges = []
        for i, vertex in enumerate(order):
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    # Determine minimal d such that girth >= g
    # Girth of LUW(k,d) is >= 2d+2
    # So 2d+2 >= g => d >= (g-2)/2
    min_d = (g - 2) // 2 + 1
    if min_d < 1:
        min_d = 1
        
    # Try from min_d upwards
    best_n = float('inf')
    best_edges = None
    
    for d in range(min_d, min_d + 5): # Try a few
        if time.time() > deadline - 10:
            break
        try:
            n, edges = build_luw_graph(k, d)
            if verify_graph(n, edges, k, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
        except Exception:
            continue
            
    if best_edges is None:
        # Fallback to random search if LUW fails (unlikely)
        # But LUW should work.
        return 0, []
        
    return best_n, best_edges
