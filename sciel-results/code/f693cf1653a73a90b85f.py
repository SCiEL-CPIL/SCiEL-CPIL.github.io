import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # For k=4, g=10, use LUW CD(5, 4)
    if k == 4 and g == 10:
        q = 4
        d = 5
        
        # GF(4) operations
        # 0:0, 1:1, 2:w, 3:w^2
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y # In char 2, -x = x
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        mul = lambda x, y: mul_table[x][y]
        
        # Rule for CD(d, q)
        # The rule definition in current program:
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3)}
        
        def neighbours(vertex):
            side, x = vertex
            # x is a tuple of length d+1, x[0] is None (or unused), x[1]..x[d] are values
            # Pre-allocate y list
            y = [None, 0] * (d + 1)
            
            for first in range(q):
                y[1] = first
                # y[0] is unused
                # y[2..d] need to be computed
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        
        i = 0
        while i < len(order):
            vertex = order[i]
            i += 1
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                # Add edge only if we are on the point side (0) to avoid duplicates
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        
        return len(order), edges

    else:
        # Fallback for other k, g
        raise ValueError("Only k=4, g=10 is supported")
