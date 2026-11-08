    """
    Build a rank-97 exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    
    Strategy: Split the 4x7 output into a 4x4 block and a 4x3 block.
    - The 4x4 block is multiplied using a rank-49 Strassen-like decomposition (based on 2x2x2 Strassen).
    - The 4x3 block is multiplied using schoolbook (rank 4*4*3 = 48).
    Total rank: 49 + 48 = 97.
    
    Note: Ensuring clean execution without external dependencies or syntax errors.
    """
