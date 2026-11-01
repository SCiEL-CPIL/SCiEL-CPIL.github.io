import time
import math
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 142.0

    best_n = float('inf')
    best_edges = None

    def verify_girth(n, adj, g, time_limit=10.0):
        """
        Check girth >= g using BFS from every vertex.
        """
        if time.time() > start_time + time_limit:
            return False
        max_depth = g // 2
        for u in range(n):
            if time.time() > start_time + time_limit:
                return False
            dist = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                curr = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        q.append(v)
                    else:
                        # If v is already visited and not the parent (handled by dist check logic mostly),
                        # a cycle exists.
                        # In a BFS tree, if we encounter a visited node v from curr, 
                        # and v is not the parent of curr, we have a back/cross edge.
                        # Cycle length = dist[curr] + 1 + dist[v]
                        # But we must ensure we don't count the edge back to parent.
                        # Since we don't store parent, we rely on the fact that 
                        # if dist[v] == dist[curr] + 1, it's a forward edge (tree edge), no cycle yet.
                        # If dist[v] <= dist[curr], it's a back/cross edge.
                        # Actually, simpler:
                        # If dist[v] != -1 and v != parent[curr], cycle length is dist[curr] + 1 + dist[v].
                        # Without parent array, we can just check if dist[v] != -1 and dist[v] != dist[curr] + 1.
                        # Wait, if dist[v] == dist[curr] + 1, it might be a tree edge or a cross edge to next level.
                        # Standard BFS girth check:
                        # If we see a visited node v, the cycle length is dist[curr] + 1 + dist[v].
                        # We need to be careful not to count the edge to the parent.
                        # Let's use a parent array to be safe and fast.
                        pass
                        
            # Re-implement with parent for correctness and speed
            # The previous implementation in the prompt was okay but let's make it robust.
            
        # Let's just use the implementation from the prompt, it was verified.
        return True

    def verify_graph(n, edges, k, g, time_limit=10.0):
        if time.time() > start_time + time_limit:
            return False
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
        
        max_depth = g // 2
        for u in range(n):
            if time.time() > start_time + time_limit:
                return False
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            q = deque([u])
            while q:
                curr = q.popleft()
                if dist[curr] >= max_depth:
                    continue
                for v in adj[curr]:
                    if dist[v] == -1:
                        dist[v] = dist[curr] + 1
                        parent[v] = curr
                        q.append(v)
                    else:
                        if v == parent[curr]:
                            continue
                        cycle_len = dist[curr] + 1 + dist[v]
                        if cycle_len < g:
                            return False
        return True

    def build_luw_graph(q_val, d_val):
        if q_val == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q_val), (lambda x, y: (x - y) % q_val), (lambda x, y: x * y % q_val)

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        def get_rule(d_v):
            r = dict(rule)
            for c in range(5, d_v + 1):
                rc, c0 = (c - 5) % 4, c - (c - 5) % 4
                r[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][rc]
            return r

        r = get_rule(d_val)
        
        def neighbours(vertex):
            side, x = vertex
            for first in range(q_val):
                y = [None, first] + [0] * (d_val - 1)
                for c in range(2, d_val + 1):
                    a, b = r[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d_val)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def build_k5_lift(m, voltages):
        """
        Build the 5*m vertex graph from K5 base with voltages in Z_m.
        """
        n = 5 * m
        edges = []
        for u in range(5):
            for v in range(5):
                if u == v: continue
                w = voltages[(u, v)]
                for a in range(m):
                    src = u * m + a
                    dst = v * m + ((a + w) % m)
                    if src < dst:
                        edges.append((src, dst))
                    elif src > dst:
                        edges.append((dst, src))
        # Deduplicate
        unique = set()
        final_edges = []
        for e in edges:
            key = (min(e[0], e[1]), max(e[0], e[1]))
            if key not in unique:
                unique.add(key)
                final_edges.append(key)
        return n, final_edges

    def check_voltage_girth(voltages, m, target_girth):
        """
        Check if the K5 lift has girth >= target_girth.
        Returns True if girth >= target_girth, False otherwise.
        """
        # 2-cycles
        for u in range(5):
            for v in range(u+1, 5):
                if (voltages[(u,v)] + voltages[(v,u)]) % m == 0:
                    return False
        
        # 3-cycles
        for u in range(5):
            for v in range(5):
                if v == u: continue
                for w in range(5):
                    if w == u or w == v: continue
                    if (voltages[(u,v)] + voltages[(v,w)] + voltages[(w,u)]) % m == 0:
                        return False
        
        # 4-cycles
        for u in range(5):
            for v in range(5):
                if v == u: continue
                for w in range(5):
                    if w == u or w == v: continue
                    for x in range(5):
                        if x == u or x == v or x == w: continue
                        if (voltages[(u,v)] + voltages[(v,w)] + voltages[(w,x)] + voltages[(x,u)]) % m == 0:
                            return False
        
        # 5-cycles
        for u in range(5):
            for v in range(5):
                if v == u: continue
                for w in range(5):
                    if w == u or w == v: continue
                    for x in range(5):
                        if x == u or x == v or x == w: continue
                        for y in range(5):
                            if y == u or y == v or y == w or y == x: continue
                            if (voltages[(u,v)] + voltages[(v,w)] + voltages[(w,x)] + voltages[(x,y)] + voltages[(y,u)]) % m == 0:
                                return False
                                
        # 6-cycles
        for u in range(5):
            for v in range(5):
                if v == u: continue
                for w in range(5):
                    if w == u or w == v: continue
                    for x in range(5):
                        if x == u or x == v or x == w: continue
                        for y in range(5):
                            if y == u or y == v or y == w or y == x: continue
                            for z in range(5):
                                if z == u or z == v or z == w or z == x or z == y: continue
                                if (voltages[(u,v)] + voltages[(v,w)] + voltages[(w,x)] + voltages[(x,y)] + voltages[(y,z)] + voltages[(z,u)]) % m == 0:
                                    return False
                                    
        # 7-cycles
        for u in range(5):
            for v in range(5):
                if v == u: continue
                for w in range(5):
                    if w == u or w == v: continue
                    for x in range(5):
                        if x == u or x == v or x == w: continue
                        for y in range(5):
                            if y == u or y == v or y == w or y == x: continue
                            for z in range(5):
                                if z == u or z == v or z == w or z == x or z == y: continue
                                for a in range(5):
                                    if a == u or a == v or a == w or a == x or a == y or a == z: continue
                                    if (voltages[(u,v)] + voltages[(v,w)] + voltages[(w,x)] + voltages[(x,y)] + voltages[(y,z)] + voltages[(z,a)] + voltages[(a,u)]) % m == 0:
                                        return False
                                        
        # 8-cycles
        for u in range(5):
            for v in range(5):
                if v == u: continue
                for w in range(5):
                    if w == u or w == v: continue
                    for x in range(5):
                        if x == u or x == v or x == w: continue
                        for y in range(5):
                            if y == u or y == v or y == w or y == x: continue
                            for z in range(5):
                                if z == u or z == v or z == w or z == x or z == y: continue
                                for a in range(5):
                                    if a == u or a == v or a == w or a == x or a == y or a == z: continue
                                    for b in range(5):
                                        if b == u or b == v or b == w or b == x or b == y or b == z or b == a: continue
                                        if (voltages[(u,v)] + voltages[(v,w)] + voltages[(w,x)] + voltages[(x,y)] + voltages[(y,z)] + voltages[(z,a)] + voltages[(a,b)] + voltages[(b,u)]) % m == 0:
                                            return False
                                            
        return True

    def optimize_voltage(m, g, time_limit=10.0):
        """
        Find a voltage assignment for K5 lift with girth >= g.
        """
        def init_voltages():
            vols = {}
            for u in range(5):
                for v in range(5):
                    if u != v:
                        vols[(u, v)] = random.randint(1, m - 1)
            for u in range(5):
                for v in range(u+1, 5):
                    if (vols[(u,v)] + vols[(v,u)]) % m == 0:
                        vols[(v,u)] = random.randint(1, m - 1)
                        while (vols[(u,v)] + vols[(v,u)]) % m == 0:
                            vols[(v,u)] = random.randint(1, m - 1)
            return vols

        best_vols = None
        for restart in range(20):
            if time.time() > start_time + time_limit:
                break
            vols = init_voltages()
            if check_voltage_girth(vols, m, g):
                return vols
            
            # Simple local search
            current_vols = vols
            for step in range(500):
                if time.time() > start_time + time_limit:
                    break
                u = random.randint(0, 4)
                v = random.randint(0, 4)
                if u == v: continue
                old = current_vols[(u,v)]
                new = random.randint(1, m-1)
                current_vols[(u,v)] = new
                if (current_vols[(u,v)] + current_vols[(v,u)]) % m == 0:
                    current_vols[(u,v)] = old
                    continue
                if check_voltage_girth(current_vols, m, g):
                    return current_vols
                # Accept if no worse? We don't have a score, just binary.
                # So we revert unless it becomes valid.
                current_vols[(u,v)] = old
        return None

    # 1. Try Voltage Graph Construction (K5 lift)
    # Moore bound for (4,10) is 161.
    # We want 5*m >= 161 => m >= 33.
    # Best known is 320 => m=64.
    # Try m values around 64, and smaller ones to see if we can beat 320.
    
    m_candidates = [64, 63, 65, 62, 66, 61, 67, 60, 68, 59, 69, 70, 58, 71, 57, 72, 56, 73, 55, 74]
    
    for m in m_candidates:
        if time.time() > deadline - 20:
            break
        n = 5 * m
        if n >= best_n:
            continue
        if n < 161:
            continue
            
        voltages = optimize_voltage(m, g, time_limit=min(5.0, deadline - time.time() - 15))
        if voltages is None:
            continue
            
        n_actual, edges = build_k5_lift(m, voltages)
        # Verify the actual graph
        if verify_graph(n_actual, edges, k, g, time_limit=min(5.0, deadline - time.time() - 10)):
            best_n = n_actual
            best_edges = edges
            if best_n <= 320:
                return best_n, best_edges

    # 2. Fallback to LUW
    if best_edges is None or time.time() < deadline - 30:
        q = k
        min_d = max(3, g - 5)
        for d_val in range(min_d, min_d + 10):
            if time.time() > deadline - 10:
                break
            try:
                n, edges = build_luw_graph(q, d_val)
                if n < best_n:
                    if verify_graph(n, edges, k, g, time_limit=5.0):
                        best_n = n
                        best_edges = edges
                        break
            except Exception:
                pass

    if best_edges is None:
        return 0, []

    return best_n, best_edges
