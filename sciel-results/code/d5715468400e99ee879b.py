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
        # y[c] = y[c-1] + y[c-2] for c >= 3? 
        # The LUW rule is:
        # Point p=(p_1..p_d), Line l=(l_1..l_d)
        # p ~ l if l_i = p_i + p_{i-1} p_{i-2} ? 
        # Let's stick to the provided rule in the current program which is known to work.
        # rule[c] = (a, b) such that y[c] = y[c-1] + y[a] * x[b] ??
        
        # Re-implementing the neighbor generation from the current program
        # which is:
        # y[c] = add(x[c], mul(y[a], x[b])) if side == 0 
        #       = sub(x[c], mul(x[a], y[b])) if side == 1
        
        # The rule definition in current program:
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            
        # For d=5, we only need rule[5].
        # c=5: r = 0, c0 = 5.
        # rule[5] = [(1, 3), (4, 1), (5, 1), (1, 6)][0] = (1, 3).
        # So for c=5: a=1, b=3.
        # y[5] = x[5] + y[1] * x[3]
        
        def neighbours(vertex):
            side, x = vertex
            # x is a tuple of length d+1, x[0] is None (or unused), x[1]..x[d] are values
            # Actually, in the current program:
            # start = (0, (None,) + (0,) * d)
            # So x[0] is None, x[1] is 0, ..., x[d] is 0.
            # Wait, the loop in current program:
            # for first in range(q):
            #   y = [None, first] + [0] * (d - 1)
            #   for c in range(2, d + 1):
            #      ...
            #      y[c] = ...
            #   yield 1 - side, tuple(y)
            
            # This generates neighbors.
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
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
            if time.time() > deadline:
                # Return whatever we have, but it might be incomplete.
                # However, LUW is deterministic and small.
                pass
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
        # Fallback for other k, g (though task specifies 4, 10)
        # Return a trivial graph or error?
        # The task says "the evaluator passes this task's degree k and girth g".
        # It implies k=4, g=10.
        # If other values are passed, we should handle them or return invalid.
        # For safety, return a small graph if possible, else raise.
        # But the contract says "Small 4-regular graph of girth 10".
        # So I assume k=4, g=10.
        raise ValueError("Only k=4, g=10 is supported")
