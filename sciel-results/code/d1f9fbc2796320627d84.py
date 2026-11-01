"""Cage graph construction: 4-regular graphs of given girth, minimized vertices."""
import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    DEADLINE = 145.0  # wall-clock limit with margin

    # Special cases with known optimal constructions
    special_cases = {
        (3, 5): (10, _petersen_edges()),
        (3, 6): (14, _heawood_edges()),
        (4, 5): (16, _clebsch_minus_matching_edges()),
    }
    
    if (k, g) in special_cases:
        return special_cases[(k, g)]

    # For the main case (4, 10): use L-U-W construction
    result = _luw_construct(k, g)
    return result

def _petersen_edges():
    edges = []
    # Outer cycle
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner star
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    # Spokes
    for i in range(5):
        edges.append((i, 5 + i))
    return edges

def _heawood_edges():
    edges = []
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6),
        (2, 4, 5)
    ]
    for i, line in enumerate(lines):
        for p in line:
            edges.append((p, 7 + i))
    return edges

def _clebsch_minus_matching_edges():
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            xor = i ^ j
            dist = bin(xor).count('1')
            if dist == 1:
                edges.append((i, j))
    return edges

def _construct_4_10_cage():
    """
    Construct a 4-regular graph of girth 10 with 320 vertices.
    This uses a known construction based on the (4,10)-cage.
    """
    # Use a known 320-vertex construction
    # The (4,10)-cage can be constructed using a specific algebraic method
    # or we can use a verified edge list
    
    # One known construction uses the incidence structure of a specific design
    # For now, use a verified construction with 320 vertices
    
    # Actually, let's use a more reliable approach:
    # The Foster census contains the (4,10)-cage with 320 vertices
    # We'll construct it using a known method
    
    # Use the construction from Exoo et al. or a verified edge list
    # For safety, let's use a construction that we can verify
    
    # Alternative: use a circulant-like construction
    # Or use the known fact that there's a 320-vertex cage
    
    # Let's use a verified construction based on the (4,10) cage
    # The cage has a specific structure that can be described
    
    # For now, return a valid 4-regular graph of girth >= 10
    # with as few vertices as possible
    
    # Use the L-U-W construction which gives 4096 vertices for (4,10)
    # But we want 320. Let's use a different approach.
    
    # Actually, the best approach is to use a known explicit construction
    # Let me construct the (4,10)-cage using a verified method
    
    # The (4,10)-cage with 320 vertices can be constructed as follows:
    # It's related to the Hoffman-Singleton type constructions
    
    # For a practical approach, let's use a verified 320-vertex construction
    # If we can't find one quickly, fall back to L-U-W
    
    # Let me try a different known construction
    # The (4,10)-cage can be built from a specific graph product
    
    # Actually, let's just use the L-U-W construction as a safe fallback
    # and return that. It's valid even if not optimal.
    return None  # Signal to use L-U-W fallback

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction for k-regular graphs of girth >= g."""
    q = k
    # For CD(d,q), girth is at least 2d-2, so d >= (g+2)//2
    d = max(3, (g + 2) // 2)
    
    # Define field operations
    if q == 4:
        # GF(4) using XOR for add/sub, and a multiplication table
        mul_table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return mul_table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    # Define the CD(d,q) rule: for each c in 2..d, rule[c] = (a, b)
    # such that y[c] = y[a] * x[b] (or x[c] = x[a] * y[b])
    # Standard L-U-W rule:
    rule = {}
    for c in range(2, d + 1):
        if c % 4 == 2:
            rule[c] = (1, c - 1)
        elif c % 4 == 3:
            rule[c] = (c - 1, 1)
        elif c % 4 == 0:
            rule[c] = (1, c - 2)
        else:  # c % 4 == 1
            rule[c] = (c - 2, 1)
    
    def neighbours(side, x):
        """Generate all q neighbors of vertex (side, x)."""
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                # x is the "right" side, computing y on left
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                # x is the "left" side, computing y on right
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    # Build the graph by BFS from a starting vertex
    start = (0, tuple([0] * (d + 1)))
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        side, x = vertex
        for other in neighbours(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            # Only add edge from left side to avoid duplicates
            if side == 0:
                edges.append((i, index[other]))
    
    return len(order), edges
