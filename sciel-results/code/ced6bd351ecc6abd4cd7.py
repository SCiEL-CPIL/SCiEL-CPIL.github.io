"""Construct the Witt design S(5,6,12) using PSL(2,11) group action."""

def construct():
    """
    Construct the Steiner system S(5,6,12) - the Witt design.
    
    Uses the action of PSL(2,11) on P^1(11) = {0, 1, ..., 10, ∞}.
    PSL(2,11) has order 660. The stabilizer of a 6-block has order 5,
    so the orbit has size 660/5 = 132 blocks.
    
    Points are 0..11 where 11 represents ∞.
    A Möbius transformation x -> (ax+b)/(cx+d) acts on P^1(11).
    """
    # Generate all 2x2 matrices over GF(11) with determinant 1 (SL(2,11))
    sl_elements = []
    for a in range(11):
        for b in range(11):
            for c in range(11):
                if a != 0:
                    bc = (b * c) % 11
                    rhs = (1 + bc) % 11
                    a_inv = pow(a, 10, 11)
                    d = (rhs * a_inv) % 11
                    sl_elements.append((a, b, c, d))
                else:
                    if b == 0:
                        continue
                    b_inv = pow(b, 10, 11)
                    c_val = (10 * b_inv) % 11
                    for d in range(11):
                        sl_elements.append((0, b, c_val, d))
    
    # Quotient by ±I to get PSL(2,11)
    psl_elements = set()
    for elem in sl_elements:
        neg_elem = ((11 - elem[0]) % 11, (11 - elem[1]) % 11,
                    (11 - elem[2]) % 11, (11 - elem[3]) % 11)
        if elem <= neg_elem:
            psl_elements.add(elem)
    
    psl_list = list(psl_elements)
    
    # Define Möbius action on P^1(11)
    def mobius_action(a, b, c, d, x):
        """Apply (ax+b)/(cx+d) to point x, where 11 represents ∞."""
        if x == 11:  # x is ∞
            if c != 0:
                return (a * pow(c, 10, 11)) % 11
            else:
                return 11  # ∞
        else:
            # x is finite
            denom = (c * x + d) % 11
            if denom == 0:
                return 11  # ∞
            num = (a * x + b) % 11
            return (num * pow(denom, 10, 11)) % 11
    
    # Base block: {∞, 0, 1, 3, 4, 5}
    base_block = frozenset([11, 0, 1, 3, 4, 5])
    
    # Generate all blocks by applying each group element to the base block
    blocks_set = set()
    for (a, b, c, d) in psl_list:
        new_block = frozenset(mobius_action(a, b, c, d, x) for x in base_block)
        blocks_set.add(new_block)
    
    # Convert to sorted lists
    result = [sorted(b) for b in blocks_set]
    
    return result
