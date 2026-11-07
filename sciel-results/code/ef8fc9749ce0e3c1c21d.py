import time
from itertools import permutations, product

def construct():
    n = 10
    deadline = time.time() + 50  # safety margin

    def make_square(k):
        """Create Latin square with Z5 slope k and Z2 slope 1."""
        sq = []
        for i in range(n):
            a_i = i // 2
            b_i = i % 2
            row = []
            for j in range(n):
                a_j = j // 2
                b_j = j % 2
                f = (a_i + k * a_j) % 5
                e = (b_i + b_j) % 2
                s = 5 * e + f
                row.append(int(s))
            sq.append(row)
        return sq

    def is_latin_square(sq):
        for i in range(n):
            if len(set(sq[i])) != n:
                return False
            for j in range(n):
                col = [sq[i][j] for i in range(n)]
                if len(set(col)) != n:
                    return False
        return True

    def are_orthogonal(sq1, sq2):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (sq1[i][j], sq2[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return len(pairs) == n * n

    def is_valid_mols(squares):
        """Check if all squares are Latin and pairwise orthogonal."""
        if not squares:
            return False
        # Check Latin property
        for sq in squares:
            if not is_latin_square(sq):
                return False
        # Check pairwise orthogonality
        for i in range(len(squares)):
            for j in range(i + 1, len(squares)):
                if not are_orthogonal(squares[i], squares[j]):
                    return False
        return True

    # Generate 4 candidate MOLS using the Z5 x Z2 construction
    squares = [make_square(k) for k in [1, 2, 3, 4]]
    
    # Verify all 4 are valid
    if is_valid_mols(squares):
        return squares
    
    # If 4 don't work, try subsets
    # Try first 3
    if len(squares) >= 3 and is_valid_mols(squares[:3]):
        return squares[:3]
    
    # Try first 2
    if len(squares) >= 2 and is_valid_mols(squares[:2]):
        return squares[:2]
    
    # Try single square
    if len(squares) >= 1 and is_latin_square(squares[0]):
        return [squares[0]]
    
    # Fallback: try to construct using a different method
    # Use the standard construction for MOLS when n is prime power
    # For n=10, we can't use the standard field construction, but we can try
    # a brute force search for a small number of squares
    
    # Try the cyclic group construction
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]
    if is_latin_square(L1):
        return [L1]
    
    # Last resort: return a single trivial Latin square
    return [[[j for j in range(n)] for i in range(n)]]
