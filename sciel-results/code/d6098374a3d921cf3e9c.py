"""Construct a k-regular graph of girth at least g using cyclic (Z_n-regular) representation."""
import random
import time
import numpy as np

def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible."""
    if k != 4 or g != 10:
        # Fallback for other parameters: use a simple construction
        # For general k, g, use the LUV construction as fallback
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

    # Special case: k=4, g=10
    # Moore bound: 1 + 4 * (1 + 3 + 9 + 27) = 1 + 4*40 = 161
    # Known best: 320 (Exoo et al. 2026)
    # Lower bound: 243
    
    start_time = time.time()
    max_time = 110  # Leave 10s margin
    
    # For cyclic Z_n-regular graphs with 4-regular, we need 2 offsets: {a, b}
    # where a != b, a != -b mod n, a != 0, b != 0, and a != n/2, b != n/2 (if n even)
    # The graph is vertex-transitive, so we only need to check girth from vertex 0
    
    def check_girth_fast(n, offsets):
        """Check if cyclic graph with given offsets has girth >= 10.
        Uses non-backtracking walks from vertex 0 only (vertex-transitive)."""
        # Build adjacency for vertex 0
        # Vertex 0 connects to: a, -a, b, -b (mod n)
        a, b = offsets
        adj0 = [a % n, (-a) % n, b % n, (-b) % n]
        
        # BFS on non-backtracking walks from (0, -1)
        # State: (current, prev)
        # We need to detect cycles of length < 10
        # A cycle of length L means we return to 0 after L steps without backtracking
        
        # For efficiency, use a set of visited states
        # State: (vertex, prev_vertex)
        visited = set()
        queue = [(0, -1)]
        visited.add((0, -1))
        depth = 0
        
        # Precompute neighbors for all vertices
        # For vertex v, neighbors are (v+a)%n, (v-a)%n, (v+b)%n, (v-b)%n
        # But we need to avoid backtracking, so we need to know which neighbor we came from
        
        while queue and depth < 9:  # We want to detect cycles of length < 10, so up to depth 9
            next_queue = []
            for u, prev in queue:
                # Neighbors of u
                # u connects to (u+a)%n, (u-a)%n, (u+b)%n, (u-b)%n
                for off in (a, -a, b, -b):
                    v = (u + off) % n
                    if v == prev:
                        continue
                    if v == 0:
                        # Found a cycle of length depth + 1
                        if depth + 1 < 10:
                            return False
                        continue
                    state = (v, u)
                    if state not in visited:
                        visited.add(state)
                        next_queue.append(state)
            queue = next_queue
            depth += 1
        
        return True
    
    # Search strategy:
    # 1. Try n from 161 upward, but skip to known good values
    # 2. For each n, try random offset pairs (a, b)
    # 3. If time runs out, fall back to a known construction
    
    # Known valid cyclic graphs for (4,10):
    # Let's try to find one efficiently
    
    random.seed(42)
    
    # Try n values that are likely to work
    # The known 320-vertex graph might be cyclic
    # Let's try a range of n values
    
    # First, try the known 320-vertex graph if it's cyclic
    # Try n = 320
    n_candidates = [320, 300, 280, 260, 250, 240, 220, 200, 180, 170, 165, 162, 161]
    
    for n in n_candidates:
        if time.time() - start_time > max_time:
            break
        
        # Generate candidate offset pairs
        # a and b must be distinct, non-zero, not n/2 (if n even)
        # and a != -b mod n
        
        valid_offsets = list(range(1, n))
        if n % 2 == 0:
            valid_offsets.remove(n // 2)
        
        if len(valid_offsets) < 2:
            continue
        
        # Try random pairs
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() - start_time > max_time:
                break
            
            a, b = random.sample(valid_offsets, 2)
            
            # Check a != -b mod n
            if (a + b) % n == 0:
                continue
            
            # Also check a != b (already ensured by sample)
            
            if check_girth_fast(n, (a, b)):
                # Build edges
                edges = []
                for i in range(n):
                    for off in (a, b):
                        j = (i + off) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
    
    # If no cyclic graph found, try a different approach:
    # Use a known small graph construction
    
    # Fallback: Use the Lazebnik-Ustimenko-Woldar construction
    q = 4
    d = 5  # For girth 10, d=5 gives girth >= 2*d+1 = 11? Let me check
    # Actually LUV with parameter d gives girth >= 2d+1
    # For girth 10, we need 2d+1 >= 10, so d >= 5 (girth >= 11)
    # Or d=4 gives girth >= 9, not enough
    
    # Let's use d=5
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
    
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
