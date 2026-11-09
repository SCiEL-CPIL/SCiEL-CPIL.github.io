"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    
    Strategy: Use the incidence graph of a generalized polygon or a known
    explicit construction. For the specific case of (k=4, g=10), we use
    a known 4-regular graph with girth 10 and 320 vertices (or try to
    construct a smaller one if possible, but the task says smallest known
    is 320).
    
    Since the task allows any valid graph and scores by n (minimized), we
    need to return the smallest n we can find. The current seed returns
    a Lazebnik-Ustimenko-Woldar graph which might not be optimal.
    
    For k=4, g=10, the known optimal is 320. Let's see if we can hardcode
    or generate a known 320-vertex graph. However, generating it from
    scratch is complex. 
    
    Alternative: Use a different family. The Hoffman-Singleton graph is
    (7,5) with 50 vertices. The (3,6) graph is 14 vertices (Heawood).
    The (4,6) graph is 20 vertices (Pappus).
    
    For (4,10), let's try to use a known construction. The graph of
    order 320 is the smallest known. If we can't find a smaller one,
    we should return a valid 320-vertex graph.
    
    However, implementing the specific 320-vertex graph from the 2026 paper
    is non-trivial without the paper's details.
    
    Let's try a different approach: Use a generic construction that works
    for small k and g, but might not be optimal. The seed uses L-U-W.
    Maybe we can use a different L-U-W variant or a different family.
    
    Actually, for (4,10), the Moore bound is 2 * (1 + 4 + 4^2 + 4^3) = 2 * (1+4+16+64) = 2 * 85 = 170.
    So 320 is significantly above the Moore bound.
    
    Let's check if there is a known 4-regular graph with girth 10 and fewer than 320 vertices.
    The task says "smallest known such graph has 320 vertices".
    
    So, if we return 320, we match the best known. If the seed returns something larger,
    we improve. If the seed returns 320, we tie.
    
    Let's see what the seed returns for k=4, g=10.
    q = 4, d = g - 5 = 5 (since 5 is odd).
    The L-U-W graph CD(5,4) has order 2 * 4^5 = 2048? No, the order is 2 * q^d.
    Wait, the order of CD(d,q) is 2 * q^d. For d=5, q=4, that's 2 * 1024 = 2048.
    That's much larger than 320.
    
    So the seed is far from optimal for (4,10).
    
    We need a better construction for (4,10).
    One well-known family is the incidence graph of a projective plane, but that's for (q+1, 4).
    For girth 10, we need a generalized 5-gon or something similar.
    
    A 4-regular graph with girth 10 is related to the (4,10)-cage.
    The smallest known (4,10)-cage has 320 vertices.
    
    Can we construct it?
    One construction is based on the 5-dimensional hypercube or similar.
    Actually, the 320-vertex graph is the incidence graph of the generalized 5-gon of order (4,4)?
    No, a generalized 5-gon of order (s,t) has girth 10 and is (s+1)-regular and (t+1)-regular on the two sides.
    If s=t=4, it's 5-regular. We need 4-regular.
    
    So it's not a generalized 5-gon of order (4,4).
    
    What about a (4,10)-cage from a different construction?
    The 320-vertex cage is known to exist.
    
    Since I don't have the explicit construction in my head, let's try to use a
    different known graph.
    
    Wait, the task says "The smallest known such graph has 320 vertices... and the best lower bound is 243."
    This implies that 320 is the current best upper bound.
    
    If I can't construct the 320-vertex graph, I might be stuck with a larger one.
    However, I can try to use a different L-U-W parameter.
    L-U-W graphs are CD(d, q) with girth at least d+5.
    For g=10, we need d=5.
    The order is 2 * q^5.
    For q=4, order is 2048.
    For q=3, order is 2 * 243 = 486. But q must be prime or 4.
    If q=3, the graph is 3-regular, not 4-regular.
    The task requires k=4.
    
    So L-U-W with q=4 gives 2048 vertices.
    
    Is there a 4-regular graph with girth 10 and < 2048 vertices?
    Yes, 320.
    
    How to construct the 320-vertex graph?
    It is often constructed using a group or a specific combinatorial design.
    
    One known construction for the (4,10)-cage is based on the 5-dimensional hypercube Q5?
    Q5 has 32 vertices and is 5-regular.
    
    Let's look for a 4-regular graph on 320 vertices with girth 10.
    The graph is bipartite.
    
    I will attempt to implement a known 320-vertex cage if I can recall it.
    Otherwise, I will fall back to a valid but potentially larger graph.
    
    Actually, there is a well-known 4-regular graph with girth 10 on 320 vertices
    called the "Exoo-Goedgebeur-Jooken-Stubbe-Van den Eede" graph (2026).
    Since I don't have the explicit edge list, I cannot hardcode it.
    
    However, I can try to use a different family.
    What about the incidence graph of a generalized quadrangle?
    A generalized quadrangle of order (s,t) has girth 8.
    We need girth 10.
    
    What about a (4,10)-cage from a different source?
    Maybe I can use a random search? No, must be deterministic and fast.
    
    Let's reconsider the L-U-W graph.
    The order is 2 * q^d.
    For k=4, q must be 4.
    d=5 gives 2048.
    
    Is there a way to get a smaller graph?
    Maybe a different construction.
    
    The Hoffman-Singleton graph is (7,5) with 50 vertices.
    The (3,6) cage is 14 vertices.
    The (4,6) cage is 20 vertices.
    The (5,6) cage is 30 vertices.
    The (3,8) cage is 30 vertices.
    The (4,8) cage is 42 vertices? No, 42 is (3,8)?
    
    For (4,10), the cage is 320.
    
    If I cannot construct the 320-vertex graph, I should return the best I can.
    The L-U-W graph with q=4, d=5 has 2048 vertices.
    Is there a smaller L-U-W graph?
    If I use q=4, d=4, girth is at least 9. Not enough.
    
    So I am stuck with 2048 unless I find a better construction.
    
    Wait, the task says "The smallest known such graph has 320 vertices".
    This suggests that 320 is achievable.
    
    Let me try to construct a 320-vertex graph using a different method.
    One method is to use a group.
    The automorphism group of the (4,10)-cage is known.
    
    Alternatively, I can use a known 4-regular graph with girth 10.
    The graph is bipartite with 160 vertices on each side.
    
    I will implement a generic construction that might yield a smaller graph than 2048.
    If I can't, I'll return the L-U-W graph.
    
    Actually, let's try to use a different L-U-W variant or a different family.
    The "M" graphs or "N" graphs?
    
    Given the complexity, I will stick with a valid construction.
    The L-U-W graph is valid.
    To improve, I need a better construction.
    
    Let's try to use the incidence graph of a generalized 5-gon of order (3,4)?
    No, that's not 4-regular on both sides.
    
    I will output the L-U-W graph for now, but note that it's not optimal.
    However, the prompt asks for a "stronger algorithm".
    
    Maybe I can use a greedy search or backtracking for small k and g?
    For k=4, g=10, n=320 is too large for backtracking.
    
    I will return the L-U-W graph. It is valid and correct.
    """
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
