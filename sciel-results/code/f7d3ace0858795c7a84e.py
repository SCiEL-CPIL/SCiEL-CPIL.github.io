def _yoshida4(n):
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - 2.0 * x1
    # Base 4th order: A(x3), B(x2), A(x1), B(x2), A(x3)
    # Note: Standard Yoshida 4th order for A+B is often A(x1)B(x2)A(x3)B(x2)A(x1)
    # Let's use the symmetric one: A(x3) B(x2) A(x1) B(x2) A(x3)
    # Sum A: x3+x1+x3. Sum B: x2+x2.
    # We will normalize later.
    a_base = np.array([x3, x1, x3])
    b_base = np.array([x2, x2])
    
    # Target length n (12)
    # We need to expand a_base (len 3) to len n and b_base (len 2) to len n.
    # To maintain symmetry, we should expand in a symmetric way.
    # Simple tiling might break symmetry if n is not a multiple.
    # n=12 is multiple of 3 and 4? No, 12 is multiple of 3 and 2.
    # Tile a_base 4 times -> len 12.
    # Tile b_base 6 times -> len 12.
    
    a_expanded = np.tile(a_base, n // len(a_base))
    b_expanded = np.tile(b_base, n // len(b_base))
    
    # Ensure lengths are exactly n
    if len(a_expanded) < n:
        a_expanded = np.concatenate([a_expanded, np.ones(n - len(a_expanded)) * (a_base.sum() / len(a_base))])
    if len(b_expanded) < n:
        b_expanded = np.concatenate([b_expanded, np.ones(n - len(b_expanded)) * (b_base.sum() / len(b_base))])
        
    a = a_expanded[:n]
    b = b_expanded[:n]
    
    # Normalize to sum 1
    a = a / np.sum(a)
    b = b / np.sum(b)
    
    return a, b
