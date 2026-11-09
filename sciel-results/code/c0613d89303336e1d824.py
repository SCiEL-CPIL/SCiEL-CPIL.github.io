"""Construct mutually orthogonal Latin squares of order 10."""
import time
import random

# EVOLVE-BLOCK-START

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

def _are_orthogonal(sq1, sq2, n):
    """Check if two Latin squares are orthogonal."""
    seen = set()
    for i in range(n):
        for j in range(n):
            pair = (sq1[i][j], sq2[i][j])
            if pair in seen:
                return False
            seen.add(pair)
    return len(seen) == n * n

def _construct_latin_from_perm(base, n):
    """
    Construct a Latin square from a base row permutation.
    base is a permutation of 0..n-1.
    Row i is base shifted by i positions (cyclic).
    This only works for prime n or when using appropriate construction.
    For n=10 (composite), we need a different approach.
    """
    # For composite n like 10, simple cyclic shift doesn't work.
    # Instead, we'll use the standard construction for n = p*q where p,q are primes.
    # 10 = 2 * 5, so we can use a product construction.
    pass

def _latin_square_from_offset(i, n):
    """
    Generate a Latin square of order n using a specific construction.
    For n=10, we use the fact that 10 = 2*5.
    We can represent each cell (r, c) as a pair and use modular arithmetic.
    
    Actually, let's use a simpler approach: generate Latin squares using
    the standard method for any n by using a complete mapping.
    """
    # For n=10, use the construction based on Z_2 x Z_5
    # Represent each element as (a, b) where a in Z_2, b in Z_5
    # Latin square L_k[r, c] where r, c are elements of Z_2 x Z_5
    # L_k[r, c] = (r[0] + c[0], r[1] + k*c[1]) mod (2, 5)
    # We need k to be a unit in Z_5 for this to be a Latin square.
    
    # Actually, let me use a more direct approach.
    # For n = p*q, we can construct Latin squares using:
    # L_k[i, j] where i, j in 0..n-1
    # Write i = 5*a + b, j = 5*c + d where a,c in 0..1, b,d in 0..4
    # L_k[i, j] = 5*(a + c mod 2) + (b + k*d mod 5)
    
    def encode(a, b):
        return 5 * a + b
    
    sq = [[0] * n for _ in range(n)]
    for i in range(n):
        a = i // 5
        b = i % 5
        for j in range(n):
            c = j // 5
            d = j % 5
            val = encode((a + c) % 2, (b + k * d) % 5)
            sq[i][j] = val
    return sq

def _generate_latin_squares(n, max_squares=9):
    """
    Generate a set of mutually orthogonal Latin squares of order n.
    For n=10, we use the product construction with n=2*5.
    """
    if n != 10:
        # Fallback for other n
        return [[[(i + j) % n for j in range(n)] for i in range(n)]]
    
    squares = []
    
    # For n=10=2*5, we can construct Latin squares L_k for k in {1,2,3,4}
    # L_k[i,j] where i=5a+b, j=5c+d
    # L_k[i,j] = 5*((a+c)%2) + ((b + k*d)%5)
    # These are orthogonal for different k values that are distinct mod 5.
    
    for k in range(1, 5):
        sq = [[0] * n for _ in range(n)]
        for i in range(n):
            a = i // 5
            b = i % 5
            for j in range(n):
                c = j // 5
                d = j % 5
                val = 5 * ((a + c) % 2) + ((b + k * d) % 5)
                sq[i][j] = val
        squares.append(sq)
    
    # Verify they are all Latin squares
    for sq in squares:
        if not _is_latin_square(sq, n):
            raise ValueError("Generated square is not a valid Latin square")
    
    # Verify mutual orthogonality
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if not _are_orthogonal(squares[i], squares[j], n):
                raise ValueError(f"Squares {i} and {j} are not orthogonal")
    
    return squares

def construct():
    """
    Construct mutually orthogonal Latin squares of order 10.
    Returns a list of 4 mutually orthogonal Latin squares.
    """
    n = 10
    squares = _generate_latin_squares(n)
    
    # Ensure all values are Python ints
    result = []
    for sq in squares:
        result.append([[int(sq[i][j]) for j in range(n)] for i in range(n)])
    
    return result

# EVOLVE-BLOCK-END
