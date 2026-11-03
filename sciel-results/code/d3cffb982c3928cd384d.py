"""Deterministic algebraic construction for regular graphs of large girth."""
import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # seconds; evaluator kills at 150s, keep margin

    q = k
    d = max(3, (g + 1) // 2 - 1)
    
    # Precompute arithmetic tables
    add_table = [[(x+y)%q for y in range(q)] for x in range(q)]
    sub_table = [[(x-y)%q for y in range(q)] for x in range(q)]
    mul_table = [[x*y%q for y in range(q)] for x in range(q)]
    
    # Precompute powers of q for encoding
    q_pows = [1] * (d + 2)
    for i in range(1, d + 2):
        q_pows[i] = q_pows[i-1] * q
    q_d = q_pows[d]  # q^d
    
    def build_rule(d_max):
        """Build the LUW rule table for columns 2..d_max."""
        rule = [None] * (d_max + 1)
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d_max + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0:
                rule[c] = (1, c0 - 2)
            elif r == 1:
                rule[c] = (c0 - 1, 1)
            elif r == 2:
                rule[c] = (c0, 1)
            else:
                rule[c] = (1, c0 + 1)
        return rule
    
    def encode(side, x_list):
        """Encode a vertex as an integer."""
        code = side * q_d
        for i in range(1, d + 1):
            code += x_list[i] * q_pows[d - i]
        return code
    
    def build_graph(d_val, rule):
        """Build the graph using integer-encoded vertices."""
        add_t = add_table
        sub_t = sub_table
        mul_t = mul_table
        q_pows_local = q_pows
        q_d_local = q_d
        
        # Precompute neighbor contributions for each column
        # For side 0: y[c] = x[c] + y[a] * x[b]
        # For side 1: y[c] = x[c] - x[a] * y[b]
        
        index = {}
        order = []
        edges = []
        
        # Start vertex: side=0, all zeros
        start_code = 0
        index[start_code] = 0
        order.append(start_code)
        
        for vidx in range(len(order)):
            code = order[vidx]
            side = 1 if code >= q_d_local else 0
            # Decode x
            x = [0] * (d_val + 1)
            rem = code % q_d_local
            for i in range(1, d_val + 1):
                x[i] = rem % q
                rem //= q
            
            if side == 0:
                for first in range(q):
                    y = [0] * (d_val + 1)
                    y[1] = first
                    for c in range(2, d_val + 1):
                        a, b = rule[c]
                        y[c] = add_t[x[c]][mul_t[y[a]][x[b]]]
                    other_code = q_d_local + y[1] * q_pows_local[d_val - 1]
                    for i in range(2, d_val + 1):
                        other_code += y[i] * q_pows_local[d_val - i]
                    if other_code not in index:
                        index[other_code] = len(order)
                        order.append(other_code)
                    edges.append((vidx, index[other_code]))
            else:
                for first in range(q):
                    y = [0] * (d_val + 1)
                    y[1] = first
                    for c in range(2, d_val + 1):
                        a, b = rule[c]
                        y[c] = sub_t[x[c]][mul_t[x[a]][y[b]]]
                    other_code = y[1] * q_pows_local[d_val - 1]
                    for i in range(2, d_val + 1):
                        other_code += y[i] * q_pows_local[d_val - i]
                    if other_code not in index:
                        index[other_code] = len(order)
                        order.append(other_code)
                    edges.append((vidx, index[other_code]))
        
        n = len(order)
        return n, edges
    
    def verify_girth(n, edges, min_girth):
        """Check girth >= min_girth using non-backtracking BFS."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_dist = (min_girth - 1) // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            found_short = False
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                du = dist[u]
                if du >= max_dist:
                    continue
                pu = parent[u]
                for v in adj[u]:
                    if v == pu:
                        continue
                    dv = dist[v]
                    if dv == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif dv >= du:
                        cycle_len = du + dv + 1
                        if cycle_len < min_girth:
                            return False
        return True
    
    # Build with initial depth
    rule = build_rule(d)
    n, edges = build_graph(d, rule)
    
    if verify_girth(n, edges, g):
        return n, edges
    
    # Try larger depths if girth is insufficient
    for d_try in range(d + 1, d + 6):
        if time.time() - t0 > deadline - 5:
            break
        rule_try = build_rule(d_try)
        n_try, edges_try = build_graph(d_try, rule_try)
        if verify_girth(n_try, edges_try, g):
            return n_try, edges_try
    
    # If all fail, return the largest graph we built (best effort)
    return n, edges
# EVOLVE-BLOCK-END
