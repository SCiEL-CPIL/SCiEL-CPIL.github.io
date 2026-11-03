import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if k == 4 and g == 10:
        return _construct_4_10(start_time, deadline)

    return _luw_construct(k, g)


def _construct_4_10(start_time, deadline):
    # Base graph: K_{5,5} minus perfect matching
    # Vertices 0..4 (top), 5..9 (bottom)
    # Edges (i, 5+j) for i != j
    base_edges = []
    for i in range(5):
        for j in range(5):
            if i != j:
                base_edges.append((i, 5 + j))

    n_lift = 320
    group_size = 32

    # Precompute all closed walks of length 4, 6, 8 in the base graph
    # A closed walk of length 2m alternates top-bottom-top-...-top
    # Represented as sequence of (top_idx, bottom_idx) pairs
    
    # For length 4: top_i -> bot_j -> top_k -> bot_l -> top_i where i!=j, k!=j, k!=l, i!=l
    # Voltage sum: v[i,j] + v[k,j] + v[k,l] + v[i,l] = 0 (mod 32)
    # Wait, need to be careful with signs. Going top->bottom adds voltage, bottom->top subtracts.
    # Actually in a voltage graph, the voltage of edge (u,v) is w(u,v) = -w(v,u)
    # For a cycle u0->u1->...->um->u0, the total voltage is sum of w(ui, ui+1)
    
    # Let's define voltage w[i][j] for edge from top i to bottom j (i != j)
    # Then w[j][i] (from bottom j to top i) = -w[i][j] mod 32
    
    # Cycle of length 4: top_a -> bot_b -> top_c -> bot_d -> top_a
    # Conditions: a!=b, c!=b, c!=d, a!=d
    # Voltage: w[a][b] + (-w[c][b]) + w[c][d] + (-w[a][d]) = 0
    # i.e., w[a][b] - w[c][b] + w[c][d] - w[a][d] = 0 mod 32
    
    # Let's enumerate all such cycles
    cycles_4 = []
    for a in range(5):
        for b in range(5):
            if a == b:
                continue
            for c in range(5):
                if c == b:
                    continue
                for d in range(5):
                    if c == d or a == d:
                        continue
                    # This is a valid 4-cycle
                    cycles_4.append((a, b, c, d))
    
    # Cycle of length 6: top_a -> bot_b -> top_c -> bot_d -> top_e -> bot_f -> top_a
    # Conditions: a!=b, c!=b, c!=d, e!=d, e!=f, a!=f
    # Voltage: w[a][b] - w[c][b] + w[c][d] - w[e][d] + w[e][f] - w[a][f] = 0
    
    cycles_6 = []
    for a in range(5):
        for b in range(5):
            if a == b:
                continue
            for c in range(5):
                if c == b:
                    continue
                for d in range(5):
                    if c == d:
                        continue
                    for e in range(5):
                        if e == d:
                            continue
                        for f in range(5):
                            if e == f or a == f:
                                continue
                            cycles_6.append((a, b, c, d, e, f))
    
    # Cycle of length 8: top_a -> bot_b -> top_c -> bot_d -> top_e -> bot_f -> top_g -> bot_h -> top_a
    # Conditions: a!=b, c!=b, c!=d, e!=d, e!=f, g!=f, g!=h, a!=h
    # Voltage: w[a][b] - w[c][b] + w[c][d] - w[e][d] + w[e][f] - w[g][f] + w[g][h] - w[a][h] = 0
    
    cycles_8 = []
    for a in range(5):
        for b in range(5):
            if a == b:
                continue
            for c in range(5):
                if c == b:
                    continue
                for d in range(5):
                    if c == d:
                        continue
                    for e in range(5):
                        if e == d:
                            continue
                        for f in range(5):
                            if e == f:
                                continue
                            for g in range(5):
                                if g == f:
                                    continue
                                for h in range(5):
                                    if g == h or a == h:
                                        continue
                                    cycles_8.append((a, b, c, d, e, f, g, h))
    
    # Now search for voltage assignments that avoid all these cycles having zero sum
    # Use a more efficient approach: try structured voltage assignments first
    
    # Strategy 1: Linear voltage assignment v[i][j] = (alpha*i + beta*j + gamma) mod 32
    # for i != j
    
    def check_voltage_assignment(v):
        """Check if voltage assignment v (5x5, v[i][j] for i!=j) gives girth >= 10"""
        # Check 4-cycles
        for (a, b, c, d) in cycles_4:
            total = (v[a][b] - v[c][b] + v[c][d] - v[a][d]) % 32
            if total == 0:
                return False
        # Check 6-cycles
        for (a, b, c, d, e, f) in cycles_6:
            total = (v[a][b] - v[c][b] + v[c][d] - v[e][d] + v[e][f] - v[a][f]) % 32
            if total == 0:
                return False
        # Check 8-cycles
        for (a, b, c, d, e, f, g, h) in cycles_8:
            total = (v[a][b] - v[c][b] + v[c][d] - v[e][d] + w_e_f(v, e, f) - w_g_f(v, g, f) + w_g_h(v, g, h) - w_a_h(v, a, h)) % 32
            if total == 0:
                return False
        return True
    
    def w_e_f(v, e, f):
        return v[e][f]
    def w_g_f(v, g, f):
        return -v[g][f] % 32
    def w_g_h(v, g, h):
        return v[g][h]
    def w_a_h(v, a, h):
        return -v[a][h] % 32
    
    # Let me fix the cycle checking functions
    def check_voltage(v):
        """v[i][j] is voltage from top i to bottom j, defined for i != j"""
        # 4-cycles: a->b->c->d->a, voltage = v[a][b] - v[c][b] + v[c][d] - v[a][d]
        for (a, b, c, d) in cycles_4:
            total = (v[a][b] - v[c][b] + v[c][d] - v[a][d]) % 32
            if total == 0:
                return False
        # 6-cycles: a->b->c->d->e->f->a, voltage = v[a][b] - v[c][b] + v[c][d] - v[e][d] + v[e][f] - v[a][f]
        for (a, b, c, d, e, f) in cycles_6:
            total = (v[a][b] - v[c][b] + v[c][d] - v[e][d] + v[e][f] - v[a][f]) % 32
            if total == 0:
                return False
        # 8-cycles: a->b->c->d->e->f->g->h->a
        # voltage = v[a][b] - v[c][b] + v[c][d] - v[e][d] + v[e][f] - v[g][f] + v[g][h] - v[a][h]
        for (a, b, c, d, e, f, g, h) in cycles_8:
            total = (v[a][b] - v[c][b] + v[c][d] - v[e][d] + v[e][f] - v[g][f] + v[g][h] - v[a][h]) % 32
            if total == 0:
                return False
        return True
    
    def build_edges(v):
        """Build the 320-vertex graph from voltage assignment v"""
        edges = []
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                w = v[i][j]
                for x in range(32):
                    u_id = i * 32 + x
                    v_id = (5 + j) * 32 + (x + w) % 32
                    edges.append((u_id, v_id))
        return edges
    
    # Try structured linear assignments
    found = False
    for alpha in range(32):
        for beta in range(32):
            for gamma in range(32):
                if time.time() > deadline - 10:
                    break
                v = [[0]*5 for _ in range(5)]
                for i in range(5):
                    for j in range(5):
                        if i != j:
                            v[i][j] = (alpha * i + beta * j + gamma) % 32
                if check_voltage(v):
                    edges = build_edges(v)
                    return 320, edges
        if time.time() > deadline - 10:
            break
    
    # Try more general structured assignments: v[i][j] = (alpha*i*j + beta*i + gamma*j + delta) mod 32
    for alpha in range(32):
        for beta in range(32):
            for gamma in range(32):
                for delta in range(32):
                    if time.time() > deadline - 10:
                        break
                    v = [[0]*5 for _ in range(5)]
                    for i in range(5):
                        for j in range(5):
                            if i != j:
                                v[i][j] = (alpha * i * j + beta * i + gamma * j + delta) % 32
                    if check_voltage(v):
                        edges = build_edges(v)
                        return 320, edges
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    # Random search
    random.seed(42)
    max_attempts = 100000
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        v = [[0]*5 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                if i != j:
                    v[i][j] = random.randint(0, 31)
        
        if check_voltage(v):
            edges = build_edges(v)
            return 320, edges
    
    # Fallback to LUW
    return _luw_construct(4, 10)


def _luw_construct(k, g):
    """LUW construction for fallback"""
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
