"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    q = k
    # Determine d such that girth >= g
    # Girth of CD(d, q) is at least d + 5 for odd d
    # We need d + 5 >= g => d >= g - 5
    # If d is even, girth is d + 4? 
    # The seed code uses: d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    # This ensures d is odd if g-5 is odd?
    # Let's stick to the seed logic for d.
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Encode state as integer for speed
    # State: side (0/1) and x[1..d]
    # x[i] in 0..q-1
    # Log2(q) bits per digit
    # For q=3, 2 bits per digit.
    # Total bits = 1 + d * 2
    # We can use a base-q representation for x[1..d]
    
    def encode(side, x_tuple):
        # x_tuple is (x1, x2, ..., xd)
        val = side
        for i in range(d):
            val = val * q + x_tuple[i]
        return val

    def decode(val):
        side = val % 2
        val //= 2
        x = [0] * d
        for i in range(d - 1, -1, -1):
            x[i] = val % q
            val //= q
        return side, tuple(x)

    # Precompute neighbors for each state?
    # Number of states is 2 * q^d.
    # For q=3, d=11, 2 * 177147 = 354294 states.
    # This is manageable.
    
    # We will perform BFS on the state space.
    # start state: side 0, x = (0, 0, ..., 0)
    start_state = encode(0, tuple([0] * d))
    
    visited = {start_state: 0}
    order = [start_state]
    edges = []
    
    # Precompute transitions for each side and each state?
    # Too much memory. Compute on the fly.
    
    for idx, state in enumerate(order):
        if time.time() > deadline:
            break
        side, x = decode(state)
        # Generate neighbors
        for first in range(q):
            y = [0] * d
            y[0] = first
            # Compute y[1..d-1] (indices 1..d-1 in 0-based, which are x[2]..x[d] in 1-based)
            # Wait, the seed code uses y[c] for c=2..d.
            # In 0-based tuple x, x[0] is x1, x[1] is x2...
            # Seed: y[c] = ... for c=2..d
            # So y[1]..y[d-1] are computed.
            for c in range(2, d + 1):
                a, b = rule[c]
                # a, b are 1-based indices in the vector
                # In 0-based: a-1, b-1
                if side == 0:
                    # y[c] = x[c] + y[a] * x[b]
                    # c is 1-based, so c-1 is 0-based
                    # a, b are 1-based, so a-1, b-1 are 0-based
                    y[c-1] = add(x[c-1], mul(y[a-1], x[b-1]))
                else:
                    # y[c] = x[c] - x[a] * y[b]
                    y[c-1] = sub(x[c-1], mul(x[a-1], y[b-1]))
            
            neighbor_state = encode(1 - side, tuple(y))
            
            if neighbor_state not in visited:
                visited[neighbor_state] = len(order)
                order.append(neighbor_state)
            
            if side == 0:
                edges.append((idx, visited[neighbor_state]))
                
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
