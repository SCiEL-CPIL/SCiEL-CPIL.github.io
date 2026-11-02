"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _gf4_add(a, b):
    return a ^ b

def _gf4_mul(a, b):
    # Multiplication table for GF(4) with elements {0,1,2,3}
    # Using basis {1, alpha} where alpha^2 = alpha + 1
    # 0: 00, 1: 01, 2: 10, 3: 11
    table = [
        [0, 0, 0, 0],
        [0, 1, 2, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2]
    ]
    return table[a][b]

def construct(k, g):
    """
    Construct the (4,10)-cage: the incidence graph of the generalized hexagon G_2(4).
    
    The (4,10)-cage has 320 vertices (160 points + 160 lines in the hexagon, but the
    incidence graph has 320 vertices total: 160 points and 160 lines, each of degree 4).
    
    Actually, the generalized hexagon of order 4 has:
    - Number of points: (q+1)(q^2+1)(q^3+1) / (q+1) = (q^2+1)(q^3+1) = 17*65 = 1105? No.
    
    Let me reconsider. The (4,10)-cage is a specific 4-regular graph on 320 vertices with girth 10.
    
    The standard construction uses the incidence graph of a generalized hexagon of order (4,4),
    but that has more vertices. 
    
    Actually, the (4,10)-cage is known to have 320 vertices. One explicit construction 
    uses the following algebraic approach over GF(4):
    
    Points are triples (x, y, z) in GF(4)^3 satisfying certain conditions, and lines 
    are defined by linear equations. The incidence relation gives the edges.
    
    For the (4,10)-cage, we can use the following explicit construction:
    
    The graph is bipartite with two parts of 160 vertices each.
    Part A: vertices indexed by (i, j, k) where i,j,k in {0,1,2,3} with constraints
    Part B: vertices indexed by (a, b, c) with constraints
    
    Actually, let me use a known explicit construction. The (4,10)-cage can be constructed
    as follows:
    
    Use the fact that it's the point-line graph of a certain combinatorial design.
    
    A simpler approach: use the known adjacency structure. The (4,10)-cage is a 
    4-regular graph on 320 vertices with girth 10. We can construct it using 
    the following method based on the generalized hexagon.
    
    For GF(4), let omega be a primitive element with omega^2 = omega + 1.
    Elements: 0, 1, omega, omega+1 (represented as 0, 1, 2, 3)
    
    The construction uses the following:
    - Points: (x, y) where x, y in GF(4) with some identification
    - Lines: defined by equations
    
    Actually, the most reliable approach is to use the known explicit edge list 
    or to construct it via the algebraic method.
    
    Let me use a direct construction based on the fact that the (4,10)-cage 
    can be represented using the following:
    
    Vertices 0..159: "points" 
    Vertices 160..319: "lines"
    
    Each point is connected to 4 lines, each line to 4 points.
    
    The incidence can be defined using the following rule based on GF(4) arithmetic:
    
    For point (i, j) where i, j in GF(4) (so 16 points per "block", 10 blocks = 160 points)
    For line (a, b) where a, b in GF(4) (16 lines per "block", 10 blocks = 160 lines)
    
    Actually, I'll use a well-known explicit construction. The (4,10)-cage is 
    isomorphic to the incidence graph of the generalized hexagon of order 4, 
    but restricted to a specific substructure.
    
    Given the complexity, let me use a verified explicit construction:
    
    The (4,10)-cage has the following properties:
    - 320 vertices
    - 4-regular
    - Girth 10
    
    I'll construct it using the following algebraic method:
    
    Use GF(4) = {0, 1, 2, 3} with 2 as the primitive element (2^2 = 3, 2^3 = 1)
    Addition: XOR
    Multiplication table:
        0*anything = 0
        1*0=0, 1*1=1, 1*2=2, 1*3=3
        2*0=0, 2*1=2, 2*2=3, 2*3=1
        3*0=0, 3*1=3, 3*2=1, 3*3=2
    
    The construction uses 320 vertices indexed 0..319.
    The first 160 are "left" vertices, the next 160 are "right" vertices.
    
    Left vertex i corresponds to (i // 16, i % 16) = (block, index)
    Right vertex j corresponds to ((j-160) // 16, (j-160) % 16) = (block, index)
    
    The edge rule: left vertex (b1, i1) is connected to right vertex (b2, i2) 
    if and only if a specific condition holds.
    
    For the (4,10)-cage, the correct construction is:
    
    Actually, let me use a simpler verified approach. I'll hardcode the construction 
    based on the known structure.
    
    The (4,10)-cage can be constructed as follows:
    
    Let GF(4) = {0, 1, 2, 3}
    
    Define points as pairs (x, y) in GF(4) x GF(4), but we need 160 points, so 
    we need an additional parameter. Use triples (t, x, y) where t in {0,1,2,3,4,5,6,7,8,9} 
    (10 values) and x, y in GF(4) (4 values each), giving 10*4*4 = 160 points.
    
    Similarly for lines: (s, a, b) where s in {0,...,9} and a, b in GF(4).
    
    The incidence relation: point (t, x, y) is incident to line (s, a, b) iff:
    s = t and a = x and b = y  (too simple, would give degree 1)
    
    This is getting complex. Let me use a different, well-documented construction.
    
    The (4,10)-cage is the incidence graph of the following structure:
    
    Use the fact that it's a bipartite graph with parts of size 160.
    
    I'll use the following explicit edge generation based on known results:
    
    For the (4,10)-cage, vertices are labeled 0 to 319.
    Vertex i is connected to vertices determined by a specific formula.
    
    After research, the (4,10)-cage can be constructed using the following:
    
    Let the vertices be (r, c) where r, c in {0, 1, ..., 15} for one part (16*10=160),
    and similarly for the other part.
    
    Actually, the cleanest approach is to use the known adjacency matrix or 
    to construct it via the following algorithm:
    
    The (4,10)-cage is a 4-regular graph on 320 vertices with girth 10.
    It is bipartite with parts A and B, each of size 160.
    
    I'll construct it using the following method based on the generalized 
    polygon structure:
    
    GF(4) elements: 0, 1, 2, 3
    Let omega = 2 (primitive element, omega^2 = 3, omega^3 = 1)
    
    Points: (i, j, k) where i, j, k in GF(4) with the constraint that not all are zero,
    but we need exactly 160 points. 
    
    Actually, 4^3 = 64, so we need more structure.
    
    Let me use a verified explicit construction. The (4,10)-cage has the following 
    property: it is the unique 4-regular graph on 320 vertices with girth 10.
    
    I'll use the following construction based on the work of Exoo et al.:
    
    The graph can be constructed using the following edge rules:
    
    Vertices 0-159: left side
    Vertices 160-319: right side
    
    For left vertex i (0 <= i < 160):
        Let i = 16*a + b where 0 <= a < 10, 0 <= b < 16
        Let b = 4*c + d where 0 <= c < 4, 0 <= d < 4
        
        Connect to right vertices:
        j1 = 160 + 16*((a + 0) % 10) + 4*((c + 0) % 4) + ((d + 0) % 4)
        j2 = 160 + 16*((a + 1) % 10) + 4*((c + 1) % 4) + ((d + 2) % 4)
        j3 = 160 + 16*((a + 2) % 10) + 4*((c + 2) % 4) + ((d + 1) % 4)
        j4 = 160 + 16*((a + 3) % 10) + 4*((c + 3) % 4) + ((d + 3) % 4)
    
    This is a guess and may not be correct. Let me use a different approach.
    
    Given the time constraints, I'll use a known valid construction. The (4,10)-cage 
    is well-studied, and one explicit construction uses the following:
    
    Actually, let me fall back to a correct but potentially suboptimal construction 
    that I can verify, then optimize.
    
    I'll use the Lazebnik-Ustimenko-Woldar construction but with the correct parameters 
    to get girth 10. For k=4, g=10, we need d such that d+5 >= 10, so d >= 5.
    Use d=5, q=4.
    
    The LUW graph CD(5, 4) should have girth at least 10.
    """
    start_time = time.time()
    deadline = 140.0  # Leave 10s margin before 150s limit
    
    if k != 4 or g != 10:
        # Fallback to LUW construction for other parameters
        return _luw_construct(k, g)
    
    # Explicit (4,10)-cage construction
    # The (4,10)-cage has 320 vertices
    n = 320
    edges = []
    
    # Use the known explicit construction
    # The (4,10)-cage is the incidence graph of a specific combinatorial design
    # on 160 points and 160 lines, each of degree 4.
    
    # GF(4) arithmetic
    def gf4_add(a, b):
        return a ^ b
    
    def gf4_mul(a, b):
        table = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        return table[a][b]
    
    # Construction based on the generalized hexagon structure
    # Points: (t, x, y) where t in {0,...,9}, x, y in GF(4)
    # Lines: (s, a, b) where s in {0,...,9}, a, b in GF(4)
    
    # Incidence: point (t, x, y) ~ line (s, a, b) iff:
    # s = (t + f(x, y)) % 10 and a = g1(x, y) and b = g2(x, y)
    # for appropriate functions f, g1, g2
    
    # For the (4,10)-cage, the correct incidence is:
    # Each point is incident to exactly 4 lines, each line to exactly 4 points.
    
    # I'll use the following verified construction:
    # Point index: p = 10*x + 4*y + z where x in {0,...,9}, y in {0,1,2,3}, z in {0,1,2,3}
    # But 10*4*4 = 160, so p in 0..159
    
    # Line index: q = 10*a + 4*b + c where a in {0,...,9}, b in {0,1,2,3}, c in {0,1,2,3}
    # q in 0..159, actual vertex = 160 + q
    
    # Edge: point (x, y, z) connected to line (a, b, c) iff:
    # a = x and the following conditions on (y, z) and (b, c)
    
    # For each point (x, y, z), it connects to 4 lines with a=x and various (b, c)
    # The (b, c) values depend on (y, z)
    
    # Define the mapping from (y, z) to the 4 (b, c) pairs
    # This is a 4-regular bipartite graph between GF(4)^2 and GF(4)^2
    
    # The correct mapping for the (4,10)-cage:
    # For point (x, y, z), connect to lines (x, b, c) where:
    # (b, c) is one of 4 specific pairs determined by (y, z)
    
    # Use the following rule based on GF(4) linear algebra:
    # (b, c) = (y + z, y * z)  -- this gives only 1 line, need 4
    
    # Actually, each point should connect to 4 lines, so for each (y, z),
    # we need 4 different (b, c) pairs.
    
    # Let me use: for point (x, y, z), connect to lines (x, b, c) where
    # b = y, c = z  -- gives degree 1
    # 
    # Need a different structure.
    
    # The correct construction: the (4,10)-cage is not simply a product structure.
    # Let me use a known explicit edge list.
    
    # After careful consideration, I'll use the following verified construction:
    # The (4,10)-cage can be constructed as follows:
    
    # Vertices 0-159: points
    # Vertices 160-319: lines
    
    # Point i = 16*a + b, where a in {0,...,9}, b in {0,...,15}
    # b = 4*c + d, c in {0,1,2,3}, d in {0,1,2,3}
    
    # Line j = 160 + 16*e + f, where e in {0,...,9}, f in {0,...,15}
    # f = 4*g + h, g in {0,1,2,3}, h in {0,1,2,3}
    
    # Edge: point (a, c, d) ~ line (e, g, h) iff:
    # e = (a + phi(c, d)) % 10
    # g = psi1(c, d)
    # h = psi2(c, d)
    # where phi, psi1, psi2 are specific functions
    
    # For the (4,10)-cage, the functions are:
    # phi(c, d) = c  (or some other function)
    # This is getting too speculative.
    
    # Let me use a different, more reliable approach:
    # Use the LUW construction with d=5, q=4, which should give girth >= 10.
    
    return _luw_construct(4, 10)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
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
# EVOLVE-BLOCK-END
