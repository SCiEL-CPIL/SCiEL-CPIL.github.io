import time
import random
from collections import deque
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Set seed for reproducibility in debugging, but we want randomness for search
    # random.seed(42) 

    def build_luw(k, d, rule):
        q = k
        # Precompute operations for speed? 
        # For k=3, we can just use % 3.
        
        def neighbours(vertex):
            side, x = vertex
            # x is a tuple of length d+1, x[0] is None, x[1..d] are values
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    if side == 0:
                        # y[c] = x[c] + y[a] * x[b]
                        # Note: y[a] is known if a < c? 
                        # In the original code, y is built sequentially.
                        # y[1] is set. y[2] uses y[a]. If a=1, ok. If a>1, y[a] must be computed.
                        # The rule must ensure a < c and b < c? 
                        # In the original patterns, a and b are usually < c.
                        y_val = (x[c] + y[a] * x[b]) % q
                    else:
                        # y[c] = x[c] - x[a] * y[b]
                        y_val = (x[c] - x[a] * y[b]) % q
                    y[c] = y_val
                yield (1-side, tuple(y))

        start = (0, (None,) + (0,)*d)
        index = {start: 0}
        order = [start]
        edges = []
        
        # Use a set for fast lookup
        # The original code used dict lookup.
        
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                # Add edge only once (from side 0)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                    
        return len(order), edges

    def check_girth_fast(n, edges, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False, n

        # BFS for girth
        max_level = (g - 1) // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = deque([s])
            while queue:
                u = queue.popleft()
                d = dist[u]
                if d >= max_level:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = d + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        # Found a cycle
                        if parent[v] != u and parent[u] != v:
                            cycle_len = d + dist[v] + 1
                            if cycle_len < g:
                                return False, n
            if time.time() > deadline - 10:
                return None, n # Timeout
        return True, n

    def valid_rule(rule, d):
        for c in range(2, d+1):
            a, b = rule[c]
            if a < 1 or a > d or b < 1 or b > d:
                return False
            # Must be causal: a < c and b < c?
            # In the LUW construction, y[c] depends on y[a] and x[b].
            # y is built left to right. So y[a] must be defined.
            # Thus a < c is required.
            # x[b] is from the input, so b can be anything 1..d?
            # But x is the "other" side.
            # Let's enforce a < c.
            if a >= c:
                return False
        return True

    best_n = float('inf')
    best_edges = None

    if k == 3:
        d_candidates = [8, 9, 10, 11]
        
        # 1. Try known good patterns from current program
        def rule_pattern_A(d):
            rule = {2: (1,1), 3: (2,1), 4: (1,2)}
            for c in range(5, d+1):
                r = (c-5) % 4
                c0 = c - r
                rule[c] = [(1, c0-2), (c0-1, 1), (c0, 1), (1, c0+1)][r]
            return rule

        def rule_pattern_B(d):
            rule = {2: (1,1), 3: (2,1), 4: (1,2)}
            for c in range(5, d+1):
                rule[c] = (1, c-2)
            return rule

        def rule_pattern_C(d):
            rule = {2: (1,1), 3: (2,1), 4: (1,2)}
            for c in range(5, d+1):
                if c % 2 == 0:
                    rule[c] = (1, c-2)
                else:
                    rule[c] = (c-2, 1)
            return rule

        def rule_pattern_D(d):
            rule = {2: (1,1), 3: (2,1), 4: (1,2)}
            for c in range(5, d+1):
                if c % 2 == 0:
                    rule[c] = (c-2, 1)
                else:
                    rule[c] = (1, c-1)
                return rule # BUG in original? No, return inside loop? 
            # Correcting the original code's potential bug
            rule = {2: (1,1), 3: (2,1), 4: (1,2)}
            for c in range(5, d+1):
                if c % 2 == 0:
                    rule[c] = (c-2, 1)
                else:
                    rule[c] = (1, c-1)
            return rule

        def rule_pattern_E(d):
            rule = {2: (1,1), 3: (2,1), 4: (1,2)}
            for c in range(5, d+1):
                rule[c] = (c-2, 1)
            return rule

        def rule_pattern_F(d):
            rule = {2: (1,1), 3: (2,1), 4: (1,2)}
            for c in range(5, d+1):
                rule[c] = (1, c-1)
            return rule

        def rule_pattern_G(d):
            rule = {2: (1,1), 3: (2,1), 4: (1,2)}
            for c in range(5, d+1):
                r = (c-5) % 4
                rule[c] = [(1, c-2), (2, c-1), (3, c-2), (1, c-1)][r]
            return rule

        def rule_pattern_H(d):
            rule = {2: (1,1), 3: (2,1), 4: (1,2)}
            for c in range(5, d+1):
                if c - 3 >= 1:
                    rule[c] = (1, c-3)
                else:
                    rule[c] = (1, c-2)
            return rule

        patterns = [rule_pattern_A, rule_pattern_B, rule_pattern_C, rule_pattern_D, 
                    rule_pattern_E, rule_pattern_F, rule_pattern_G, rule_pattern_H]

        for d_try in d_candidates:
            if time.time() > deadline - 20:
                break
            for p in patterns:
                if time.time() > deadline - 15:
                    break
                rule = p(d_try)
                if not valid_rule(rule, d_try):
                    continue
                try:
                    n, edges = build_luw(k, d_try, rule)
                    if n > 5000: # Prune large graphs
                        continue
                    is_valid, n_val = check_girth_fast(n, edges, g)
                    if is_valid is True:
                        if n_val < best_n:
                            best_n = n_val
                            best_edges = edges
                        # If we found a very good one, we can stop early?
                        if best_n <= 936:
                            return best_n, best_edges
                    elif is_valid is None:
                        # Timeout in check, assume it might be good? No, risky.
                        pass
                except Exception:
                    continue

        # 2. Randomized Search
        # We will try random rules for d=8, 9, 10
        # Number of trials depends on time
        trials = 0
        while time.time() < deadline - 5:
            trials += 1
            d_try = random.choice([8, 9, 10])
            rule = {}
            # Base rules
            rule[2] = (1, 1)
            rule[3] = (2, 1)
            rule[4] = (1, 2)
            
            valid = True
            for c in range(5, d_try+1):
                # a must be < c
                a = random.randint(1, c-1)
                # b can be 1..d, but usually small
                b = random.randint(1, d_try)
                # Avoid trivial self-loops in the logic?
                rule[c] = (a, b)
            
            if not valid_rule(rule, d_try):
                continue
                
            try:
                n, edges = build_luw(k, d_try, rule)
                if n > 5000:
                    continue
                # Quick check: is it simple?
                # build_luw ensures simple? 
                # It adds edges (u,v). If u==v, loop.
                # If edge exists twice, not simple.
                # The construction from LUW usually yields simple graphs if rules are well-behaved.
                
                is_valid, n_val = check_girth_fast(n, edges, g)
                if is_valid is True:
                    if n_val < best_n:
                        best_n = n_val
                        best_edges = edges
                        if best_n <= 936:
                            return best_n, best_edges
            except Exception:
                continue

    if best_n != float('inf') and best_edges is not None:
        return best_n, best_edges

    # Fallback: Return a valid graph if possible, or the smallest found
    # If nothing found, try d=8 with pattern A
    d = 8
    rule = rule_pattern_A(d)
    n, edges = build_luw(k, d, rule)
    return n, edges

