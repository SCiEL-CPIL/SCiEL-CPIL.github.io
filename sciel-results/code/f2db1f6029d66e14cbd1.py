"""Construct 4-regular girth-10 cage graph (n(4,10)).

Strategy:
- Use the known (4,10)-cage structure: a bipartite graph with 320 vertices.
- The Lazebnik-Ustimenko-Woldar (LUW) family gives explicit q-regular bipartite
  graphs of large girth. For k=4 (q=4, GF(4)) and girth 10, we need d such that
  girth >= d+5 (for odd d) or d+6 (for even d).
  
  For g=10: we need girth >= 10. With odd d: d+5 >= 10 => d >= 5.
  With even d: d+6 >= 10 => d >= 4.
  
  The LUW graph with q=4, d=5 should have girth >= 10.
  
  However, the LUW construction may not give the minimal n. The known (4,10)-cage
  has 320 vertices. Let me check: the LUW graph with q=4, d=5 has q^d points and
  q^d lines = 4^5 + 4^5 = 2048 vertices total, but we take the connected component
  of the zero vertex.
  
  Actually, for the cage problem, I should try to find the actual (4,10)-cage.
  Since constructing it from scratch is hard, let me use the LUW construction
  and see what size component we get. If it's larger than 320, I'll try to
  optimize.

  Actually, let me reconsider. The task says the smallest known is 320 vertices.
  The LUW graph might give a larger component. Let me implement it correctly
  and see.

  For GF(4) with d=5:
  - Points: vectors in GF(4)^d, so 4^5 = 1024 points
  - Lines: vectors in GF(4)^d, so 4^5 = 1024 lines
  - Total: 2048 vertices
  - The connected component of the zero vertex might be much smaller.

  Let me implement and check. If the component is too large, I'll try other approaches.

  Alternative: use the Hoffman-Singleton-type or other known cage constructions.
  For (4,10), one known construction is based on the 320-vertex cage.

  Let me just implement the LUW construction properly and return the component.
"""

import time
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """Construct a k-regular graph with girth at least g, minimizing n."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    if k == 4 and g == 10:
        # Use the LUW construction with q=4, d=5
        # This should give a 4-regular bipartite graph with girth >= 10
        n, edges = _luw_cage(4, 5)
        if n > 0:
            return n, edges
    
    # Fallback: LUW construction with appropriate parameters
    q = k
    # For girth g, we need d such that:
    # If d is odd: girth >= d + 5
    # If d is even: girth >= d + 6
    # So for g=10: odd d >= 5, even d >= 4
    if g <= 5:
        d = 3
    elif g <= 6:
        d = 4
    elif g <= 9:
        d = 5
    elif g <= 10:
        d = 5
    else:
        d = g - 5 if (g - 5) % 2 == 1 else g - 4
        if d < 3:
            d = 3

    n, edges = _luw_cage(q, d)
    if n > 0:
        return n, edges

    # If LUW fails, try a simple cycle-based construction (won't meet girth, but returns something)
    # This should not happen for valid inputs
    return 0, []


def _luw_cage(q, d):
    """
    Construct the connected component of the zero vertex in the LUW graph CD(d, q).
    q must be 4 or a prime.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j).
    """
    # Set up GF(q) operations
    if q == 4:
        # GF(4) = {0, 1, alpha, alpha+1} where alpha^2 = alpha + 1
        # Represent as: 0->0, 1->1, alpha->2, alpha+1->3
        # Addition is XOR
        # Multiplication table:
        #   0*anything = 0
        #   1*x = x
        #   2*2 = 3, 2*3 = 2, 2*1 = 2
        #   3*3 = 1, 3*2 = 2, 3*1 = 3
        def gf_add(x, y):
            return x ^ y

        def gf_sub(x, y):
            return x ^ y  # In characteristic 2, subtraction = addition

        def gf_mul(x, y):
            if x == 0 or y == 0:
                return 0
            if x == 1:
                return y
            if y == 1:
                return x
            # x, y in {2, 3}
            if x == 2 and y == 2:
                return 3
            if x == 2 and y == 3:
                return 2
            if x == 3 and y == 2:
                return 2
            if x == 3 and y == 3:
                return 1
            return 0  # Should not reach here
    elif q == 2:
        def gf_add(x, y):
            return x ^ y
        def gf_sub(x, y):
            return x ^ y
        def gf_mul(x, y):
            return x & y
    else:
        # Prime field
        def gf_add(x, y):
            return (x + y) % q
        def gf_sub(x, y):
            return (x - y) % q
        def gf_mul(x, y):
            return (x * y) % q

    # Define the rule for LUW graph
    # For c = 2, 3, ..., d:
    # rule[c] = (a, b) such that the edge condition is:
    #   line[c] - point[c] = line[a] * point[b]
    # The standard LUW rule:
    # For d >= 2:
    #   rule[2] = (1, 1)
    #   rule[3] = (2, 1) if d >= 3
    #   For c >= 4: rule[c] depends on the pattern

    # Standard LUW rule from the paper:
    # The rule is defined as follows:
    # r_2 = (1,1)
    # r_3 = (2,1)
    # r_4 = (1,2)
    # For c >= 5: r_c = (i_c, j_c) where the pattern cycles

    # Actually, let me use the standard definition from Lazebnik-Ustimenko-Woldar:
    # The graph CD(d, q) has points and lines as d-tuples over GF(q).
    # Point p = (p_1, ..., p_d), Line l = (l_1, ..., l_d)
    # p is incident to l iff for all c = 2, ..., d:
    #   l_c - p_c = l_{a_c} * p_{b_c}
    # where (a_c, b_c) is a fixed pair with 1 <= a_c, b_c <= d-1.

    # The specific rule used in the LUW paper for large girth:
    # r_2 = (1, 1)
    # r_3 = (2, 1)
    # r_4 = (1, 2)
    # r_5 = (3, 1)
    # r_6 = (1, 3)
    # r_7 = (4, 1)
    # r_8 = (1, 4)
    # ...
    # Pattern: for odd c >= 3: r_c = (c//2 + 1, 1) if c//2 + 1 <= d-1
    #           for even c >= 4: r_c = (1, c//2) if c//2 <= d-1

    # Let me use a simpler, well-known rule:
    # r_c = (min(c, d-1), min(c//2 + 1, d-1)) for appropriate c
    # Actually, let me use the rule that's known to work:

    rule = {}
    rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    if d >= 5:
        rule[5] = (3, 1)
    if d >= 6:
        rule[6] = (1, 3)
    if d >= 7:
        rule[7] = (4, 1)
    if d >= 8:
        rule[8] = (1, 4)
    if d >= 9:
        rule[9] = (5, 1)
    if d >= 10:
        rule[10] = (1, 5)

    # For general c, extend the pattern
    for c in range(2, d + 1):
        if c not in rule:
            if c % 2 == 1:  # odd
                a = c // 2 + 1
                b = 1
            else:  # even
                a = 1
                b = c // 2
            # Clamp to valid range
            a = min(a, d - 1)
            b = min(b, d - 1)
            rule[c] = (a, b)

    # Build the graph using BFS from the zero vertex
    # Vertex representation: (side, coords) where side=0 for points, side=1 for lines
    # coords is a tuple of length d

    zero_point = (0, tuple([0] * d))
    zero_line = (1, tuple([0] * d))

    # BFS
    index = {}
    order = []
    queue = deque()

    index[zero_point] = 0
    order.append(zero_point)
    queue.append(zero_point)

    edges = []

    while queue:
        vertex = queue.popleft()
        side, coords = vertex
        v_idx = index[vertex]

        for other in _luw_neighbours(vertex, d, gf_add, gf_sub, gf_mul, rule):
            if other not in index:
                index[other] = len(order)
                order.append(other)
                queue.append(other)
            # Only add edge from point to line to avoid duplicates
            if side == 0:
                o_idx = index[other]
                if v_idx < o_idx:
                    edges.append((v_idx, o_idx))
                else:
                    edges.append((o_idx, v_idx))

    n = len(order)
    return n, edges


def _luw_neighbours(vertex, d, gf_add, gf_sub, gf_mul, rule):
    """
    Generate neighbors of a vertex in the LUW graph.
    vertex = (side, coords) where coords is a tuple of length d.
    side=0: point, side=1: line.
    """
    side, coords = vertex
    coords_list = list(coords)

    if side == 0:
        # Point p, find all lines l such that p is incident to l
        # For each line coordinate l_1 (free variable), compute l_c for c=2..d
        # l_c = p_c + l_{a_c} * p_{b_c}
        for l1 in range(len(set(coords)) if False else _get_field_size(coords)):
            l_coords = [None, l1] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                l_coords[c] = gf_add(coords[c], gf_mul(l_coords[a], coords[b]))
            yield (1, tuple(l_coords))
    else:
        # Line l, find all points p such that p is incident to l
        # For each point coordinate p_1 (free variable), compute p_c for c=2..d
        # p_c = l_c - l_{a_c} * p_{b_c}
        for p1 in range(_get_field_size(coords)):
            p_coords = [None, p1] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                p_coords[c] = gf_sub(coords[c], gf_mul(coords[a], p_coords[b]))
            yield (0, tuple(p_coords))


def _get_field_size(coords):
    """Determine the field size from the coordinate values."""
    max_val = max(coords) if coords else 0
    if max_val <= 1:
        return 2
    elif max_val <= 3:
        return 4
    else:
        # For prime fields, the field size is max_val + 1 (assuming 0..q-1)
        return max_val + 1


# EVOLVE-BLOCK-END
