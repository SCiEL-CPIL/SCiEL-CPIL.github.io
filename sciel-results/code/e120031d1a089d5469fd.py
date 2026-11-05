import time
import random
import sys
import numpy as np

def construct(k, g):
    deadline = time.time() + 140.0
    
    # LUW Construction
    def get_luw_graph():
        # Determine d
        min_d_odd = max(3, g - 5)
        if min_d_odd % 2 == 0:
            min_d_odd += 1
        min_d_even = max(4, g - 4)
        if min_d_even % 2 == 1:
            min_d_even += 1
        d = min(min_d_odd, min_d_even)
        q = k
        
        # Define field operations
        if q == 4:
            mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return mul_table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return (x * y) % q

        # Rules
        rule_a = [0] * (d + 1)
        rule_b = [0] * (d + 1)
        rule_a[2], rule_b[2] = 1, 1
        rule_a[3], rule_b[3] = 2, 1
        rule_a[4], rule_b[4] = 1, 2
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
            if r == 0: a, b = 1, c0 - 2
            elif r == 1: a, b = c0 - 1, 1
            elif r == 2: a, b = c0, 1
            else: a, b = 1, c0 + 1
            rule_a[c] = a
            rule_b[c] = b

        n_vertices = 2 * (q ** (d + 1))
        adj = [[] for _ in range(n_vertices)]
        
        # Iterate over all vertices
        for side in range(2):
            for x_tuple in range(q ** (d + 1)):
                # Decode x
                x = [0] * (d + 1)
                temp = x_tuple
                for i in range(d + 1):
                    x[i] = temp % q
                    temp //= q
                
                vid = side * (q ** (d + 1)) + x_tuple
                
                for first in range(q):
                    y = [0] * (d + 1)
                    y[1] = first
                    if side == 0:
                        for c in range(2, d + 1):
                            a, b = rule_a[c], rule_b[c]
                            y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        for c in range(2, d + 1):
                            a, b = rule_a[c], rule_b[c]
                            y[c] = sub(x[c], mul(x[a], y[b]))
                    
                    # Encode y
                    y_tuple = 0
                    for i in range(d + 1):
                        y_tuple += y[i] * (q ** i)
                    
                    other_vid = (1 - side) * (q ** (d + 1)) + y_tuple
                    adj[vid].append(other_vid)
        
        edges = []
        for u in range(n_vertices):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        return n_vertices, edges

    return get_luw_graph()
