"""Rewritten program using a deterministic construction based on finite field structure."""
# EVOLVE-BLOCK-START
def construct():
    """
    Return 120 integer colours for edges of K16 in lexicographic order.
    
    Strategy: Use a construction based on the finite field GF(16) = GF(2^4).
    We label vertices 0..15 with elements of GF(16). The edge colour between
    vertices a and b is determined by the "slope" (a+b)/(a+b+1) in GF(16),
    mapped to an integer 0..14 (using 15 colours, or fewer if possible).
    
    Actually, let's use a simpler and well-known approach:
    We use the fact that for n = 2^k, we can use a construction based on
    the affine geometry AG(k, 2). However, the constraint is about K4s having
    at least 5 colours.
    
    Let me think more carefully. The seed uses (i+j) % 17 which gives 17 colours
    but we need to minimize. The bounds say 13 <= f(16) <= 15.
    
    Let me try a construction with 15 colours using GF(16):
    - Vertices are elements of GF(16)
    - For edge {a, b} where a != b, define colour = discrete_log((a-b)^(15)) 
      or some function of the difference.
    
    Actually, let me use a different approach. In GF(16), the multiplicative group
    has order 15. So we can map each non-zero element to a value in 0..14.
    
    For edge {a, b}, let d = a XOR b (addition in GF(2^4)). If d != 0, then
    d is in the multiplicative group of order 15. We can assign colour = 
    index of d in a fixed generator's powers.
    
    But wait: in GF(2^4), addition is XOR. The difference a-b = a+b (since char 2).
    So for any two distinct vertices a, b, a+b is non-zero, and we can take its
    discrete log base a primitive element to get a value in 0..14.
    
    Let's verify the K4 constraint: For any 4 vertices a, b, c, d, we have 6 edges.
    The colours are disc_log(a+b), disc_log(a+c), disc_log(a+d), disc_log(b+c),
    disc_log(b+d), disc_log(c+d).
    
    We need at least 5 distinct colours among these 6.
    
    The sum of all 6 "differences" in the multiplicative sense... this is tricky.
    
    Let me just implement this and see if it works. If it gives 15 colours and
    satisfies the constraint, that's a valid answer (and better than 17).
    """
    # Build GF(16) using primitive polynomial x^4 + x + 1 = 0x13
    # Elements are 0..15, multiplication defined by reduction mod x^4 + x + 1
    
    # Precompute multiplication table
    def gf_mul(a, b):
        result = 0
        for i in range(4):
            if b & (1 << i):
                result ^= a
            a <<= 1
            if a & 16:  # x^4
                a ^= 0x13  # x^4 + x + 1
        return result
    
    # Find a primitive element (generator of the multiplicative group of order 15)
    # Try element 3 (binary 0011)
    def is_primitive(g):
        if g == 0:
            return False
        # Check if g^15 == 1 and g^5 != 1, g^3 != 1, g^1 != 1
        x = g
        for _ in range(14):
            x = gf_mul(x, g)
        return x == 1 and gf_mul(g, g) != 1 and gf_mul(gf_mul(g, g), g) != 1
    
    # Find primitive element
    primitive = None
    for g in range(1, 16):
        if is_primitive(g):
            primitive = g
            break
    
    # Build discrete log table
    # disc_log[x] = k such that primitive^k = x for x in 1..15
    disc_log = {}
    x = 1
    for k in range(15):
        disc_log[x] = k
        x = gf_mul(x, primitive)
    
    # For edge {i, j}, colour = disc_log[i ^ j] (since in GF(2^4), i+j = i XOR j)
    colours = []
    for i in range(16):
        for j in range(i + 1, 16):
            d = i ^ j
            # d is non-zero since i != j
            c = disc_log[d]
            colours.append(c)
    
    # Verify: every K4 must have at least 5 distinct colours
    # Let's check a few K4s to make sure
    import itertools
    valid = True
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        edge_cols = set()
        for u, v in itertools.combinations(quad, 2):
            edge_cols.add(disc_log[u ^ v])
        if len(edge_cols) < 5:
            valid = False
            break
    
    # If valid, return 15 colours. If not, fall back to seed.
    if valid:
        return colours
    else:
        # Fallback to seed construction
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
# EVOLVE-BLOCK-END
