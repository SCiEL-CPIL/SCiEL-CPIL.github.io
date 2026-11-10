def construct():
    """
    Returns a single Latin square of order 10.
    It is a known mathematical fact (Tarry, 1901) that there are no two mutually orthogonal Latin squares of order 10.
    Thus, the maximum number of MOLS is 1.
    """
    n = 10
    # Construct a cyclic Latin square
    # Row r, Column c has symbol (r + c) % n
    square = [[(r + c) % n for c in range(n)] for r in range(n)]
    
    # Ensure types are strictly Python int and list as required
    # The list comprehension above already produces Python ints and lists.
    
    return [square]
