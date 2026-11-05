<<<<<<< SEARCH
    # Fallback: Lazebnik-Ustimenko-Woldar
    def luw_fallback():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_val, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_val]

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
=======
    # Fallback: Lazebnik-Ustimenko-Woldar
    def luw_fallback():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_op = lambda x, y: x ^ y
            sub_op = lambda x, y: x ^ y
            mul_op = lambda x, y: table[x][y]
        else:
            add_op = lambda x, y: (x + y) % q
            sub_op = lambda x, y: (x - y) % q
            mul_op = lambda x, y: x * y % q
        
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_val, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_val]

        # Precompute neighbors for all possible x tuples
        # x is a tuple of length d+1 where x[0] is None, x[1..d] are in range(q)
        # side is 0 or 1
        # For each (side, x), compute all q neighbors
        
        from collections import defaultdict
        neighbor_cache = {}
        
        def compute_neighbors(side, x):
            key = (side, x)
            if key in neighbor_cache:
                return neighbor_cache[key]
            
            result = []
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_op(x[c], mul_op(y[a], x[b]))
                    else:
                        y[c] = sub_op(x[c], mul_op(x[a], y[b]))
                result.append((1 - side, tuple(y)))
            
            neighbor_cache[key] = result
            return result

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        
        # Use BFS with explicit queue for better performance
        from collections import deque
        queue = deque([start])
        
        while queue:
            vertex = queue.popleft()
            idx_v = index[vertex]
            side, x = vertex
            
            for other in compute_neighbors(side, x):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                    queue.append(other)
                if side == 0:
                    edges.append((idx_v, index[other]))
        
        return len(order), edges
>>>>>>> REPLACE

