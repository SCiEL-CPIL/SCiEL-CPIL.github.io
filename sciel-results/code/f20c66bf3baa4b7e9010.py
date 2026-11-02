import time
import random
from collections import defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140

    # Fallback: LUW Graph
    def luw_construct(k, g):
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

    if k != 4 or g != 10:
        return luw_construct(k, g)

    # Specific (4,10) Cage Construction
    # 1. Base Graph
    p_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    p_set = set(p_edges)
    p_set.update([(b, a) for a, b in p_edges])
    
    # Perfect Matching
    candidate_match = [(0, 2), (1, 4), (3, 5), (6, 8), (7, 9)]
    if not all((u, v) not in p_set for u, v in candidate_match):
        # Find another
        used = [False] * 10
        match = []
        def find_match(idx):
            if idx == 10: return True
            if used[idx]: return find_match(idx + 1)
            for j in range(idx + 1, 10):
                if not used[j] and (idx, j) not in p_set:
                    used[idx] = used[j] = True
                    match.append((idx, j))
                    if find_match(idx + 1): return True
                    match.pop()
                    used[idx] = used[j] = False
            return False
        find_match(0)
        candidate_match = match

    base_edges = p_edges + candidate_match
    
    # 2. Walks
    adj = defaultdict(list)
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
        
    all_walks = []
    for start_v in range(10):
        stack = [(start_v, -1, [])]
        while stack:
            curr, prev, path = stack.pop()
            if len(path) >= 3 and curr == start_v:
                is_nb = True
                for i in range(len(path) - 1):
                    if path[i][1] == path[i+1][0] and path[i][0] == path[i+1][1]:
                        is_nb = False
                        break
                if is_nb and len(path) > 1:
                    if path[-1][1] == path[0][0] and path[-1][0] == path[0][1]:
                        is_nb = False
                if is_nb:
                    all_walks.append(path)
                    continue
            if len(path) < 9:
                for next_v in adj[curr]:
                    if next_v != prev:
                        stack.append((next_v, curr, path + [(curr, next_v)]))
                        
    # 3. Voltage Search
    oriented_edges = {}
    for i, (u, v) in enumerate(base_edges):
        if u > v: u, v = v, u
        oriented_edges[(u, v)] = i
        
    def get_vol(u, v, vols):
        if (u, v) in oriented_edges:
            return vols[oriented_edges[(u, v)]]
        return -vols[oriented_edges[(v, u)]]
        
    def check_walk(walk, vols):
        s = 0
        for u, v in walk:
            s += get_vol(u, v, vols)
        return s % 32 == 0
        
    random.seed(42)
    vols = [random.randint(1, 31) for _ in range(len(base_edges))]
    
    # Greedy Search
    for _ in range(10000):
        if time.time() > deadline:
            break
        violated = [w for w in all_walks if check_walk(w, vols)]
        if not violated:
            break
        w = random.choice(violated)
        u, v = random.choice(w)
        idx = oriented_edges.get((u, v))
        if idx is None: idx = oriented_edges.get((v, u))
        if idx is not None:
            old = vols[idx]
            vols[idx] = random.randint(1, 31)
            # Optional: check if it improved, else revert
            # For speed, just keep it
            pass

    # Verify
    if any(check_walk(w, vols) for w in all_walks):
        return luw_construct(k, g)

    # 4. Lift
    n = 320
    edges = []
    for u, v in base_edges:
        if (u, v) in oriented_edges:
            vol = vols[oriented_edges[(u, v)]]
        else:
            vol = -vols[oriented_edges[(v, u)]]
        for i in range(32):
            edges.append((u * 32 + i, v * 32 + (i + vol) % 32))
            
    return n, edges
