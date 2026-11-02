import time
import sys
import math
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140
    
    # For k=4, g=10, target n=320
    if k == 4 and g == 10:
        M = 40
        # Base graph K_{4,4}
        # Vertices 0-3 (U), 4-7 (V)
        # Voltages w[u][v]
        
        def count_short_cycles(voltages, M):
            # Returns count of cycles of length 4, 6, 8
            # We check from a subset of vertices to speed up?
            # No, we need to ensure NO short cycles.
            # But for optimization, we can count them.
            # A 4-cycle in the lift corresponds to a 4-cycle in the base graph
            # with voltages summing to 0 mod M.
            # Base graph K_{4,4} has many 4-cycles.
            # A 4-cycle in base: u1-v1-u2-v2-u1.
            # Voltage sum: w[u1][v1] - w[u2][v1] + w[u2][v2] - w[u1][v2] == 0 mod M?
            # Direction: u1->v1 (w[u1][v1]), v1->u2 (-w[u2][v1]), u2->v2 (w[u2][v2]), v2->u1 (-w[u1][v2]).
            # Sum = w[u1][v1] - w[u2][v1] + w[u2][v2] - w[u1][v2].
            
            count_4 = 0
            count_6 = 0
            count_8 = 0
            
            # Check 4-cycles
            for u1 in range(4):
                for u2 in range(4):
                    if u1 == u2: continue
                    for v1 in range(4):
                        for v2 in range(4):
                            if v1 == v2: continue
                            # Cycle u1-v1-u2-v2-u1
                            s = voltages[u1][v1] - voltages[u2][v1] + voltages[u2][v2] - voltages[u1][v2]
                            if s % M == 0:
                                count_4 += 1
            
            # 6-cycles and 8-cycles are harder to enumerate analytically.
            # We can just run a BFS check on the full graph for the final candidate.
            # For the local search, we only minimize 4-cycles?
            # If 4-cycles are 0, we still need to check 6 and 8.
            # But 4-cycles are the most numerous and easiest to break.
            
            return count_4, count_6, count_8

        def is_valid(voltages, M, g):
            # Full check
            n_vertices = 8 * M
            for start_v in range(n_vertices):
                visited = set()
                queue = []
                u = start_v // M
                i = start_v % M
                
                if u < 4:
                    for v in range(4):
                        nv = (v + 4) * M + (i + voltages[u][v]) % M
                        visited.add((nv, start_v))
                        queue.append((nv, start_v, 1))
                else:
                    v = u - 4
                    for u_prime in range(4):
                        nv = u_prime * M + (i - voltages[u_prime][v]) % M
                        visited.add((nv, start_v))
                        queue.append((nv, start_v, 1))
                        
                while queue:
                    curr, parent, depth = queue.pop(0)
                    if curr == start_v and depth > 1:
                        if depth < g:
                            return False
                    if depth + 1 >= g:
                        continue
                    u_c = curr // M
                    i_c = curr % M
                    if u_c < 4:
                        for v in range(4):
                            nv = (v + 4) * M + (i_c + voltages[u_c][v]) % M
                            if nv == parent: continue
                            if (nv, curr) in visited: continue
                            visited.add((nv, curr))
                            queue.append((nv, curr, depth + 1))
                    else:
                        v_c = u_c - 4
                        for u_prime in range(4):
                            nv = u_prime * M + (i_c - voltages[u_prime][v_c]) % M
                            if nv == parent: continue
                            if (nv, curr) in visited: continue
                            visited.add((nv, curr))
                            queue.append((nv, curr, depth + 1))
            return True

        # Local Search
        best_volts = None
        best_score = float('inf')
        
        for _ in range(10): # 10 restarts
            if time.time() > deadline - 20: break
            volts = [[random.randint(0, M-1) for _ in range(4)] for _ in range(4)]
            curr_score = count_short_cycles(volts, M)[0]
            
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for u in range(4):
                    for v in range(4):
                        old_val = volts[u][v]
                        for new_val in range(M):
                            if new_val == old_val: continue
                            volts[u][v] = new_val
                            new_score = count_short_cycles(volts, M)[0]
                            if new_score < curr_score:
                                curr_score = new_score
                                improved = True
                                break
                            elif new_score == curr_score and random.random() < 0.1:
                                # Random walk to escape local minima?
                                curr_score = new_score
                                improved = True
                                break
                            else:
                                volts[u][v] = old_val
            
            if curr_score < best_score:
                best_score = curr_score
                best_volts = [row[:] for row in volts]
                
            if best_score == 0:
                # Check for 6 and 8 cycles
                if is_valid(best_volts, M, g):
                    edges = set()
                    for u in range(4):
                        for i in range(M):
                            v_start = u * M + i
                            for v in range(4):
                                v_end = (v + 4) * M + (i + best_volts[u][v]) % M
                                if v_start < v_end:
                                    edges.add((v_start, v_end))
                                else:
                                    edges.add((v_end, v_start))
                    return 320, list(edges)

    # Fallback LUW
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def gf_add(x, y): return x ^ y
        def gf_sub(x, y): return x ^ y
        def gf_mul(x, y): return table[x][y]
    else:
        def gf_add(x, y): return (x + y) % q
        def gf_sub(x, y): return (x - y) % q
        def gf_mul(x, y): return x * y % q

    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    queue = [0]
    head = 0
    while head < len(order):
        if time.time() > deadline: break
        v_idx = queue[head]
        head += 1
        v = order[v_idx]
        for other in neighbours(v):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(len(order) - 1)
            if v[0] == 0:
                edges.append((v_idx, index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
