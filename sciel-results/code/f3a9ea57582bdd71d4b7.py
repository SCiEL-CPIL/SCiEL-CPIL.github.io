import time

def _luw_fallback(k, g):
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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    def is_valid_circulant(n, S, g):
        if len(set(S)) != 5 or 0 in S:
            return False
        
        # Check girth
        # We use a set of visited states (last_vertex_offset, current_sum, length) ?
        # No, just enumerate walks.
        
        # Optimization: Precompute valid next steps for each s in S
        # nexts[s] = [t for t in S if t != -s]
        
        # We check lengths 3 to g-1
        
        for L in range(3, g):
            # Use a set to track sums? No, we just need to know if 0 is reachable.
            # DFS
            def dfs(current_sum, last_s, depth):
                if depth == L:
                    return current_sum % n == 0
                for next_s in S:
                    if next_s == (-last_s) % n:
                        continue
                    if dfs((current_sum + next_s) % n, next_s, depth + 1):
                        return True
                return False
            
            for s1 in S:
                if dfs(s1 % n, s1, 1):
                    return False
        return True

    best_n = float('inf')
    best_S = None
    
    # Search n from 428
    # We try structured S sets
    for n in range(428, 1500):
        if time.time() > deadline:
            break
            
        candidates = []
        # Pattern 1: Symmetric + 1
        for p in range(2, 20):
            for q in range(p+1, 30):
                S = [1, n-1, p, n-p, q]
                if len(set(S)) == 5:
                    candidates.append(S)
                    
        # Pattern 2: Consecutive
        for start in range(1, 10):
            S = [start, start+1, start+2, start+3, start+4]
            S = [x % n for x in S]
            if len(set(S)) == 5 and 0 not in S:
                candidates.append(S)
                
        for S in candidates:
            if is_valid_circulant(n, S, g):
                best_n = n
                best_S = S
                break
        if best_S:
            break
            
    if best_S:
        n = best_n
        S = best_S
        edges = []
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if i < j:
                    edges.append((i, j))
        return n, edges
    else:
        return _luw_fallback(k, g)
