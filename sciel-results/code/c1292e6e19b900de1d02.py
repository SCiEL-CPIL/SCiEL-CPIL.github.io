"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_latin_square(sq, n):
    """Check if sq is a valid Latin square of order n with symbols 0..n-1."""
    for i in range(n):
        row = sq[i]
        if len(row) != n:
            return False
        if sorted(row) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _is_orthogonal(sq1, sq2, n):
    """Check if two Latin squares are orthogonal."""
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def _mols_5():
    """
    Construct 4 mutually orthogonal Latin squares of order 5 using GF(5).
    Returns list of 4 Latin squares, each a list of 5 lists of ints 0..4.
    """
    n = 5
    squares = []
    # GF(5) operations: addition and multiplication mod 5
    # L_k(i, j) = (i + k*j) mod 5 for k in 0,1,2,3
    for k in range(4):
        sq = [[(i + k * j) % n for j in range(n)] for i in range(n)]
        squares.append(sq)
    return squares

def _construct_10_from_5(mols5):
    """
    Construct Latin squares of order 10 from MOLS of order 5.
    Uses the standard construction: if we have m MOLS of order n,
    we can construct m MOLS of order 2n using a 2x2 Latin square.
    
    For order 10 = 2*5, given 4 MOLS of order 5, we can construct
    4 MOLS of order 10. Additionally, we can add more squares.
    
    The construction: Given Latin squares L1,...,Lm of order n,
    construct new squares of order 2n:
    For row (a, i) and column (b, j) where a,b in {0,1}, i,j in 0..n-1:
    The new square M has:
    M[(a,i),(b,j)] = 2*L[(a+i)%n, (b+j)%n] + (a+b)%2  ... this isn't quite right.
    
    Actually, the standard construction for MOLS of order 2n from MOLS of order n:
    If L1,...,Lm are MOLS of order n, then we can form MOLS of order 2n.
    
    Let me use a simpler approach. For order 10, I'll use the fact that
    there exist 9 MOLS of order 10 (since 10 is not a prime power, but we
    know at least some exist).
    
    Actually, let me use a direct construction. For order 10, we can use
    the fact that 10 = 2*5 and construct squares using the group Z_10 or
    other methods.
    
    Let me try a different approach: use the finite field GF(5) to get 4 MOLS
    of order 5, then "blow up" to order 10.
    
    The blow-up: given a Latin square L of order n, construct a Latin square
    of order 2n by:
    For position (2i+a, 2j+b) where a,b in {0,1}:
    value = 2*L[i,j] + (a+b)%2
    This gives a Latin square of order 2n with symbols 0..2n-1.
    
    But we need to check orthogonality. If L1 and L2 are orthogonal, are
    their blow-ups orthogonal? Not necessarily.
    
    Let me instead try to directly construct MOLS of order 10.
    
    Actually, I recall that for n=10, there are known constructions. Let me
    try using the group-based construction or a direct search.
    
    For a simpler approach, let me just return 1 square for now and ensure
    it's valid, then try to add more.
    """
    n = 10
    
    # Start with a simple Latin square of order 10
    sq0 = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Try to construct additional orthogonal squares
    # Using the GF(5) approach: partition rows and columns into pairs
    # Row i has "group" i//5 and "index" i%5
    # Col j has "group" j//5 and "index" j%5
    
    # For each MOLS of order 5, construct a corresponding MOLS of order 10
    squares = [sq0]
    
    for mols_idx, L5 in enumerate(mols5):
        # Construct a Latin square of order 10 from L5
        # Position (2a+i, 2b+j) -> use L5[i,j] combined with something
        # Let's try: value = 2*L5[i,j] + (a+b)%2
        # But this might not be a Latin square...
        
        # Alternative: value = 2*L5[(i+a)%5, (j+b)%5] + (a+b)%2
        # Let me check if this is a Latin square
        
        sq = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                a = i // 5
                b = j // 5
                ii = i % 5
                jj = j % 5
                # Use L5 with some mixing
                val = 2 * L5[ii, jj] + (a + b) % 2
                sq[i][j] = val
        
        # Check if this is a valid Latin square
        if _is_latin_square(sq, n):
            # Check orthogonality with all existing squares
            orthogonal = True
            for existing in squares:
                if not _is_orthogonal(existing, sq, n):
                    orthogonal = False
                    break
            if orthogonal:
                squares.append(sq)
    
    return squares

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 second safety margin
    
    # Get MOLS of order 5
    mols5 = _mols_5()
    
    # Construct MOLS of order 10
    squares = _construct_10_from_5(mols5)
    
    # Ensure we have at least 1 square
    if not squares:
        n = 10
        squares = [[[(i + j) % n for j in range(n)] for i in range(n)]]
    
    # Try to add more squares via random search if time permits
    n = 10
    max_squares = n - 1  # Maximum possible is n-1 = 9
    
    while len(squares) < max_squares and time.time() < deadline:
        # Try to find a new Latin square orthogonal to all existing ones
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Generate a random Latin square
            # Start with a random permutation for the first row
            first_row = list(range(n))
            random.shuffle(first_row)
            
            # Build the square row by row
            sq = [first_row[:]]
            for i in range(1, n):
                # For each row, we need a permutation that is orthogonal to all existing squares
                # This is complex, so let's try a simpler approach
                pass
        
        # If we can't easily add more, break
        if not found:
            break
    
    # Convert to pure Python lists of ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    return result
# EVOLVE-BLOCK-END
