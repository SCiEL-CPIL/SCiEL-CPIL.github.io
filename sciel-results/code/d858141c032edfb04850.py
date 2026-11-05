import time
import random
import math
import itertools

def construct(k, g):
    start_time = time.time()
    deadline = 120.0
    
    if k == 4 and g == 10:
        # Voltage graph construction for (4,10)-cage
        # Base: K4,4
        # Group: Z_40
        
        cycles = []
        
        # 4-cycles
        for l1 in range(4):
            for l2 in range(l1+1, 4):
                for r1 in range(4):
                    for r2 in range(r1+1, 4):
                        e1 = l1*4 + r1
                        e2 = l2*4 + r1
                        e3 = l2*4 + r2
                        e4 = l1*4 + r2
                        cycles.append((e1, -e2, e3, -e4))
                        
        # 6-cycles
        for l1 in range(4):
            for l2 in range(l1+1, 4):
                for l3 in range(l2+1, 4):
                    for r1 in range(4):
                        for r2 in range(r1+1, 4):
                            for r3 in range(r2+1, 4):
                                e1 = l1*4 + r1
                                e2 = l2*4 + r1
                                e3 = l2*4 + r2
                                e4 = l3*4 + r2
                                e5 = l3*4 + r3
                                e6 = l1*4 + r3
                                cycles.append((e1, -e2, e3, -e4, e5, -e6))
                                
        # 8-cycles
        for p in itertools.permutations([0,1,2,3]):
            r1, r2, r3, r4 = p
            e1 = 0*4 + r1
            e2 = 1*4 + r1
            e3 = 1*4 + r2
            e4 = 2*4 + r2
            e5 = 2*4 + r3
            e6 = 3*4 + r3
            e7 = 3*4 + r4
            e8 = 0*4 + r4
            cycles.append((e1, -e2, e3, -e4, e5, -e6, e7, -e8))
            
        for attempt in range(500):
            if time.time() > deadline - 5:
                break
            voltages = [random.randint(0, 39) for _ in range(16)]
            
            g_val = 40
            for v in voltages:
                g_val = math.gcd(g_val, v)
            if g_val != 1:
                continue
                
            valid = True
            for cyc in cycles:
                prod = 0
                for idx, sign in cyc:
                    prod += sign * voltages[idx]
                if prod % 40 == 0:
                    valid = False
                    break
            
            if valid:
                n = 320
                edges = []
                for l in range(4):
                    for r in range(4):
                        w = voltages[l*4+r]
                        for g_val2 in range(40):
                            u_lift = l * 40 + g_val2
                            v_lift = (r+4) * 40 + (g_val2 + w) % 40
                            edges.append((u_lift, v_lift))
                            
                return n, edges
                
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
