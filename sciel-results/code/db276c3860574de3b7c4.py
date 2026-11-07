"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def _build_squares_from_base(base, n=10):
    """
    Given a base Latin square L (as list of lists), generate a set of mutually
    orthogonal Latin squares using the method: for a group G of order n that is
    the direct product of two groups of coprime order, we can sometimes construct
    MOLS. For n=10 = 2*5, we use the fact that Z_10 has a special structure.
    
    Actually, for n=10, a well-known construction gives exactly 2 MOLS.
    Here's the approach: we'll use the known fact that for prime powers we have
    p^k - 1 MOLS. For n=10, the maximum is 2 (it's known that there are exactly
    2 MOLS of order 10).
    
    So we need to construct 2 mutually orthogonal Latin squares of order 10.
    
    Let me use a concrete construction. One approach:
    - Use the group Z_2 x Z_5 with a specific orthomorphism.
    
    Actually, let me just hardcode two known MOLS of order 10.
    
    Here's a known pair of MOLS of order 10:
    L1[i][j] = (i + j) % 10  -- this is a Latin square
    L2[i][j] = (2i + 3j) % 10 -- need to check if this is orthogonal to L1
    
    Let me verify: For L1 and L2 to be orthogonal, the map (i,j) -> (L1[i][j], L2[i][j])
    must be a bijection on Z_10 x Z_10.
    
    L1 = i + j
    L2 = 2i + 3j
    
    Given (a, b) where a = i+j mod 10, b = 2i+3j mod 10.
    From a: j = a - i.
    Substitute: b = 2i + 3(a-i) = 2i + 3a - 3i = 3a - i.
    So i = 3a - b, j = a - (3a-b) = b - 2a.
    
    So for any (a,b), we get i = 3a - b mod 10, j = b - 2a mod 10.
    This is unique, so L1 and L2 are orthogonal!
    
    But wait, is L2 a Latin square? Let me check:
    Row i: L2[i][j] = 2i + 3j mod 10. As j varies, 3j mod 10 takes values {0,3,6,9,2,5,8,1,4,7} = all of Z_10.
    Yes, because gcd(3,10)=1. So L2 is a Latin square.
    
    So we have 2 MOLS! Let me construct them properly.
    """
    # L1[i][j] = (i + j) % 10
    # L2[i][j] = (2*i + 3*j) % 10
    
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    L2 = [[(2*i + 3*j) % n for j in range(n)] for i in range(n)]
    
    # Verify L1 is Latin
    for i in range(n):
        row = L1[i]
        col = [L1[k][i] for k in range(n)]
        assert sorted(row) == list(range(n)), f"Row {i} of L1 not Latin: {row}"
        assert sorted(col) == list(range(n)), f"Col {i} of L1 not Latin: {col}"
    
    # Verify L2 is Latin
    for i in range(n):
        row = L2[i]
        col = [L2[k][i] for k in range(n)]
        assert sorted(row) == list(range(n)), f"Row {i} of L2 not Latin: {row}"
        assert sorted(col) == list(range(n)), f"Col {i} of L2 not Latin: {col}"
    
    # Verify orthogonality
    pairs = set()
    for i in range(n):
        for j in range(n):
            pair = (L1[i][j], L2[i][j])
            assert pair not in pairs, f"Pair {pair} repeated at ({i},{j})"
            pairs.add(pair)
    assert len(pairs) == n*n, f"Only {len(pairs)} distinct pairs"
    
    return [L1, L2]


def construct():
    n = 10
    squares = _build_squares_from_base(None, n)
    # Ensure all values are Python ints and lists are proper lists
    result = []
    for sq in squares:
        result.append([[int(x) for x in row] for row in sq])
    return result
# EVOLVE-BLOCK-END
