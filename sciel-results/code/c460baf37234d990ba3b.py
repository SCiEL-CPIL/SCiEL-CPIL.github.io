import time
import random
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety

    # Target: k=4, g=10
    if k != 4 or g != 10:
        raise ValueError("Only k=4, g=10 is supported")

    # 1. Generate LUW graph as fallback
    # LUW CD(5, 4)
    q = 4
    d = 5
    
    # GF(4) operations
    mul_table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: mul_table[x][y]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
    def neighbours_luw(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start_luw = (0, (None,) + (0,) * d)
    index_luw = {start_luw: 0}
    order_luw = [start_luw]
    edges_luw = []
    
    i = 0
    while i < len(order_luw):
        if time.time() > deadline - 20: # Leave 20s for verification/output
            break
        vertex = order_luw[i]
        i += 1
        for other in neighbours_luw(vertex):
            if other not in index_luw:
                index_luw[other] = len(order_luw)
                order_luw.append(other)
            if vertex[0] == 0:
                edges_luw.append((index_luw[vertex], index_luw[other]))
    
    n_luw = len(order_luw)
    # If LUW generation was cut off, it's invalid. But it's small enough.
    # Check if complete
    if i < len(order_luw):
        # Fallback to a trivial valid graph if LUW fails? 
        # No, LUW for d=5 is 2048 vertices. It should finish in < 1s.
        # If it doesn't, something is wrong.
        pass

    # 2. Try Voltage Lift Search
    # Base: K_4,4 (8 vertices, 4-regular)
    # Try to find a lift with girth >= 10 and N < n_luw
    best_n = n_luw
    best_edges = edges_luw
    
    # Precompute short cycles in K_4,4
    # Vertices 0-3 (L), 4-7 (R)
    # Edges (l, r)
    base_edges = [(l, 4+r) for l in range(4) for r in range(4)]
    
    # Find all cycles of length 4, 6, 8 in K_4,4
    # 4-cycles: (l1, r1, l2, r2, l1)
    cycles_4 = []
    for l1 in range(4):
        for l2 in range(l1+1, 4):
            for r1 in range(4):
                for r2 in range(r1+1, 4):
                    # Cycle: l1 - r1 - l2 - r2 - l1
                    # Edges: (l1, 4+r1), (l2, 4+r1), (l2, 4+r2), (l1, 4+r2)
                    cycles_4.append([(l1, 4+r1), (l2, 4+r1), (l2, 4+r2), (l1, 4+r2)])
    
    # 6-cycles and 8-cycles are more numerous.
    # For simplicity and speed, we will only enforce 4-cycle constraints in the search,
    # and then verify the full girth at the end.
    # If 4-cycles are eliminated, girth is at least 6.
    # We hope that random voltage assignments also eliminate 6 and 8 cycles.
    
    def try_voltage_lift(m, time_limit=30.0):
        deadline_lift = time.time() + time_limit
        rng = random.Random(42)
        
        # Edge indices for K_4,4
        # e[l][r] corresponds to edge (l, 4+r)
        # We assign voltages v[l][r] in Z_m
        
        # Constraints: For each 4-cycle, sum of voltages != 0
        # Cycle: (l1, r1), (l2, r1), (l2, r2), (l1, r2)
        # Note: Directions matter.
        # Walk: l1 -> 4+r1 -> l2 -> 4+r2 -> l1
        # Edge 1: l1 to 4+r1. Voltage v[l1][r1].
        # Edge 2: 4+r1 to l2. This is edge (l2, 4+r1) traversed backwards.
        # In voltage lift, if we go u->v with voltage v_e, we go v->u with -v_e.
        # So sum = v[l1][r1] - v[l2][r1] + v[l2][r2] - v[l1][r2]
        # We need this sum != 0 mod m.
        
        def check_constraints(voltages):
            # voltages is a 4x4 list
            for cyc in cycles_4:
                # cyc is list of (l, r) tuples in order?
                # My cycles_4 list is not in walk order.
                # Let's reconstruct walk order.
                # l1, r1, l2, r2
                # Walk: l1 -> (4+r1) -> l2 -> (4+r2) -> l1
                # e1: (l1, 4+r1) forward. v[l1][r1]
                # e2: (4+r1, l2) which is (l2, 4+r1) backward. -v[l2][r1]
                # e3: (l2, 4+r2) forward. v[l2][r2]
                # e4: (4+r2, l1) which is (l1, 4+r2) backward. -v[l1][r2]
                s = (voltages[l1][r1] - voltages[l2][r1] + voltages[l2][r2] - voltages[l1][r2]) % m
                if s == 0:
                    return False
            return True

        # Backtracking search
        # Order edges: row by row
        edges_to_assign = [(l, r) for l in range(4) for r in range(4)]
        
        def backtrack(idx, current_voltages):
            if time.time() > deadline_lift:
                return None
            if idx == 16:
                return current_voltages
            l, r = edges_to_assign[idx]
            # Try random values
            vals = list(range(m))
            rng.shuffle(vals)
            for val in vals:
                current_voltages[l][r] = val
                # Prune: check constraints involving this edge if possible
                # Simple: just check all 4-cycles that include this edge?
                # For speed, we'll just check at the end or every few steps.
                # Given m is small (e.g. 10-50), checking all 36 cycles is fast.
                if check_constraints(current_voltages):
                    res = backtrack(idx + 1, current_voltages)
                    if res:
                        return res
            return None

        # Try different m
        for m in [10, 20, 40, 80]:
            if time.time() > deadline_lift:
                break
            # Initialize with zeros
            v = [[0]*4 for _ in range(4)]
            res = backtrack(0, v)
            if res:
                # Construct lifted graph
                # Vertices: (v_base, i) where i in 0..m-1
                # Edge (u, v) in base with voltage w becomes (u, i) -- (v, i+w)
                n_lifted = 8 * m
                edges_lifted = set()
                for l in range(4):
                    for r in range(4):
                        u_base = l
                        v_base = 4 + r
                        w = res[l][r]
                        for i in range(m):
                            u_lifted = u_base * m + i
                            v_lifted = v_base * m + (i + w) % m
                            if u_lifted > v_lifted:
                                u_lifted, v_lifted = v_lifted, u_lifted
                            edges_lifted.add((u_lifted, v_lifted))
                
                # Verify girth >= 10
                if _verify_girth(n_lifted, list(edges_lifted), k, g):
                    return n_lifted, list(edges_lifted)
        return None

    # Try voltage lift
    lift_result = try_voltage_lift(10, time_limit=30.0)
    if lift_result:
        n, edges = lift_result
        if n < best_n:
            best_n = n
            best_edges = edges

    return best_n, best_edges

def _verify_girth(n, edges, k, g):
    """
    Verify simple, k-regular, connected, girth >= g.
    """
    if n < 1:
        return False
    if len(edges) != len(set(edges)):
        return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
        if a < 0 or a >= n or b < 0 or b >= n or a == b:
            return False
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        if len(adj[i]) != k:
            return False

    # Connectivity
    seen = [False] * n
    stack = [0]
    seen[0] = True
    cnt = 1
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if not seen[w]:
                seen[w] = True
                cnt += 1
                stack.append(w)
    if cnt != n:
        return False

    # Girth check
    INF = 10**9
    for (u, v) in edges:
        dist = {}
        q = []
        for w in adj[u]:
            dist[(u, w)] = 1
            q.append((u, w, 1))
        head = 0
        best = INF
        while head < len(q):
            prev, cur, L = q[head]
            head += 1
            if cur == v and L >= 1:
                if L + 1 < best:
                    best = L + 1
                continue
            if L >= g:
                break
            for w in adj[cur]:
                if w == prev:
                    continue
                if (cur, w) not in dist:
                    dist[(cur, w)] = L + 1
                    q.append((cur, w, L + 1))
        if best < g:
            return False
    return True
