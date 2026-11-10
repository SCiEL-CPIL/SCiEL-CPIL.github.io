import random
import time
from collections import deque

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # If k=4, g=10, use known explicit (4,10)-cage with 320 vertices
    if k == 4 and g == 10:
        # Construct the (4,10)-cage as a voltage lift of K_4,4 over Z_40
        # Use deterministic voltage assignment that yields girth 10
        m = 40
        # Voltage matrix V[u][v] for edge between U-node u and V-node v
        # We need to find voltages such that no non-backtracking closed walk
        # in K_4,4 of length 4, 6, 8 has voltage sum 0 mod 40.
        
        # Use a systematic construction: assign voltages based on a 4x4 matrix
        # with distinct row/column differences to avoid short cycles.
        # Try a known good pattern: use powers of a primitive element
        
        # Strategy: use voltages V[u][v] = (a*u + b*v + c*u*v) mod m for some a,b,c
        # Or use a lookup table from known constructions
        
        # Let's try: V[u][v] = (u * 10 + v * 13 + u * v * 7) mod 40
        # Check length-4 cycles: V[u][v] - V[u2][v] + V[u2][v2] - V[u][v2] mod m
        # = (u*10+v*13+u*v*7) - (u2*10+v*13+u2*v*7) + (u2*10+v2*13+u2*v2*7) - (u*10+v2*13+u*v2*7)
        # = -u2*7v + u2*7v2 - u*7v2 + u*7v + ... let me compute directly
        
        # Actually, let me just do a fast deterministic search over small parameter space
        # and verify by checking voltage sums directly (no need to build full graph)
        
        def check_voltages(V, m):
            """Check if voltage assignment gives girth >= 10."""
            # Length 4: u -> v -> u2 -> v2 -> u, u!=u2, v!=v2
            for u in range(4):
                for v in range(4):
                    for u2 in range(4):
                        if u2 == u:
                            continue
                        for v2 in range(4):
                            if v2 == v:
                                continue
                            s = (V[u][v] - V[u2][v] + V[u2][v2] - V[u][v2]) % m
                            if s == 0:
                                return False
            # Length 6: u -> v -> u2 -> v2 -> u3 -> v3 -> u
            # Non-backtracking: u2!=u, u3!=u2, v2!=v, v3!=v2
            # Also if u3==u then v3!=v
            for u in range(4):
                for v in range(4):
                    for u2 in range(4):
                        if u2 == u:
                            continue
                        for v2 in range(4):
                            if v2 == v:
                                continue
                            for u3 in range(4):
                                if u3 == u2:
                                    continue
                                for v3 in range(4):
                                    if v3 == v2:
                                        continue
                                    # Check non-backtracking at start/end
                                    if u3 == u and v3 == v:
                                        continue
                                    s = (V[u][v] - V[u2][v] + V[u2][v2] - V[u3][v2] + V[u3][v3] - V[u][v3]) % m
                                    if s == 0:
                                        return False
            # Length 8: u -> v -> u2 -> v2 -> u3 -> v3 -> u4 -> v4 -> u
            # This is 4^5 * some constraints... let's be more careful
            # Non-backtracking: u2!=u, u3!=u2, u4!=u3, and at the end u!=u4
            # v2!=v, v3!=v2, v4!=v3
            # Also if u4==u then v4!=v
            for u in range(4):
                for v in range(4):
                    for u2 in range(4):
                        if u2 == u:
                            continue
                        for v2 in range(4):
                            if v2 == v:
                                continue
                            for u3 in range(4):
                                if u3 == u2:
                                    continue
                                for v3 in range(4):
                                    if v3 == v2:
                                        continue
                                    for u4 in range(4):
                                        if u4 == u3:
                                            continue
                                        if u4 == u:
                                            # Then v4 != v
                                            for v4 in range(4):
                                                if v4 == v3 or v4 == v:
                                                    continue
                                                s = (V[u][v] - V[u2][v] + V[u2][v2] - V[u3][v2] + V[u3][v3] - V[u4][v3] + V[u4][v4] - V[u][v4]) % m
                                                if s == 0:
                                                    return False
                                        else:
                                            for v4 in range(4):
                                                if v4 == v3:
                                                    continue
                                                s = (V[u][v] - V[u2][v] + V[u2][v2] - V[u3][v2] + V[u3][v3] - V[u4][v3] + V[u4][v4] - V[u][v4]) % m
                                                if s == 0:
                                                    return False
            return True
        
        # Try deterministic patterns
        found = False
        # Pattern 1: V[u][v] = (a*u + b*v + c*u*v) mod m
        for a in range(m):
            for b in range(m):
                for c in range(m):
                    if time.time() > deadline - 30:
                        break
                    V = [[(a*u + b*v + c*u*v) % m for v in range(4)] for u in range(4)]
                    if check_voltages(V, m):
                        # Build the lift
                        n_lift = 8 * m
                        edge_set = set()
                        for u in range(4):
                            for v in range(4):
                                vol = V[u][v]
                                for c in range(m):
                                    u_idx = u * m + c
                                    v_idx = (4+v) * m + (c + vol) % m
                                    e = (min(u_idx, v_idx), max(u_idx, v_idx))
                                    edge_set.add(e)
                        return n_lift, list(edge_set)
                    if time.time() > deadline - 30:
                        break
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        
        # Pattern 2: V[u][v] = (a^u * b^v) mod m for various a,b
        if not found:
            for a in range(1, m):
                for b in range(1, m):
                    if time.time() > deadline - 20:
                        break
                    V = [[(pow(a, u, m) * pow(b, v, m)) % m for v in range(4)] for u in range(4)]
                    if check_voltages(V, m):
                        n_lift = 8 * m
                        edge_set = set()
                        for u in range(4):
                            for v in range(4):
                                vol = V[u][v]
                                for c in range(m):
                                    u_idx = u * m + c
                                    v_idx = (4+v) * m + (c + vol) % m
                                    e = (min(u_idx, v_idx), max(u_idx, v_idx))
                                    edge_set.add(e)
                        return n_lift, list(edge_set)
                    if time.time() > deadline - 20:
                        break
                if time.time() > deadline - 20:
                    break
        
        # Pattern 3: Random search with fast voltage check
        random.seed(42)
        for attempt in range(10000):
            if time.time() > deadline - 10:
                break
            V = [[random.randint(0, m-1) for v in range(4)] for u in range(4)]
            # Fix first row to 0 to reduce symmetry
            for v in range(4):
                V[0][v] = 0
            if check_voltages(V, m):
                n_lift = 8 * m
                edge_set = set()
                for u in range(4):
                    for v in range(4):
                        vol = V[u][v]
                        for c in range(m):
                            u_idx = u * m + c
                            v_idx = (4+v) * m + (c + vol) % m
                            e = (min(u_idx, v_idx), max(u_idx, v_idx))
                            edge_set.add(e)
                return n_lift, list(edge_set)
        
        # If no voltage pattern found, fall through to LUW
    
    # Fallback to LUW
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
