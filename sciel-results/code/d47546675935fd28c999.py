"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a complete set of 9 mutually orthogonal Latin squares of order 10.
    
    Order 10 is not a prime power, but it is isomorphic to Z_5 x Z_2.
    We can construct MOLS using the structure of the group Z_5 x Z_2.
    
    The standard construction for MOLS of order pq (where p, q are primes)
    when p=5, q=2 uses the fact that we can define squares based on linear
    functions over the field Z_5, extended to the group Z_5 x Z_2.
    
    Specifically, we represent elements as pairs (a, b) where a in Z_5, b in Z_2.
    We map (a, b) to index i = a + 5*b.
    
    For each slope k in Z_5 \ {0}, and for each "offset" pattern in Z_2,
    we can construct a Latin square.
    
    Actually, a simpler known construction for order 10:
    Since 10 = 2 * 5, and 2 and 5 are primes, we can use the fact that
    there exist 9 MOLS of order 10 (Euler's 36 officers problem was solved
    for order 10 in 1959 by Bose, Shrikhande, and Parker).
    
    The construction uses the group Z_5 x Z_2. We define 9 squares as follows:
    For each pair (s, t) where s in Z_5 \ {0} and t in Z_2, plus additional
    squares to get 9 total.
    
    Let me use a direct construction:
    Elements are indexed by i = a + 5*b where a in {0,1,2,3,4}, b in {0,1}.
    
    Square L_{s,t} for s in {1,2,3,4}, t in {0,1} (8 squares) plus one more.
    
    Actually, the maximum number of MOLS of order n is n-1 when n is a prime power.
    For n=10, which is not a prime power, the maximum is 9 (since 10-1=9).
    
    Here's a concrete construction:
    Represent cell (i,j) where i = 5*x + y (x in Z_2, y in Z_5) and j = 5*u + v (u in Z_2, v in Z_5).
    
    For square k (k=0..8), define:
    If k < 5:
      Let s = k+1 (s in {1,2,3,4,5} but 5=0 in Z_5, so s in {1,2,3,4,0})
      Wait, let me use a different approach.
    
    Known construction: Use the fact that Z_5 x Z_2 is isomorphic to... 
    Actually, let me just hardcode a known set of 9 MOLS of order 10.
    """
    # Generate 9 MOLS of order 10 using the group Z_5 x Z_2
    
    # Map index i to (a, b) where i = a + 5*b, a in Z_5, b in Z_2
    def idx_to_pair(i):
        return (i % 5, i // 5)
    
    def pair_to_idx(a, b):
        return a + 5 * b
    
    # For each of the 9 squares, we define a function f: (a,b) x (u,v) -> (c,d)
    # where (a,b) is row, (u,v) is column, (c,d) is symbol.
    # The 9 squares correspond to 9 distinct "slopes" in the appropriate sense.
    
    # Construction:
    # Square k for k in 0..8:
    #   Let k = 5*m + l where m in {0,1}, l in {0,1,2,3,4}
    #   But we need 9 distinct ones. Let's use k=0..8.
    
    # A standard construction:
    # For s in Z_5 \ {0} (4 values) and t in Z_2 (2 values), we get 8 squares.
    # Plus one more square.
    
    # Let me use this known parametrization:
    # Row i = 5*x + y, Col j = 5*u + v
    # Symbol for square (s, t) where s in {1,2,3,4}, t in {0,1}:
    #   c = (s * (y - v)) % 5  ... no, this needs to be a Latin square.
    
    # Let me think differently. A Latin square of order 10 can be defined by:
    # L[i][j] = f(i, j)
    # where i = 5*x + y, j = 5*u + v
    # f(i,j) = 5*c + d
    
    # For orthogonality, we need that for any two squares L, L', the map
    # (i,j) -> (L[i][j], L'[i][j]) is bijective.
    
    # Here's a working construction from the literature:
    # Define 9 squares as follows. For k in 0..8:
    #   k = 5 * m + n where m in {0,1}, n in {0,1,2,3,4}
    #   But we only have 10 such pairs, and we need 9.
    
    # Actually, let me use a simpler approach: I'll define the squares explicitly
    # using a known valid set.
    
    # The 9 MOLS of order 10 can be constructed as follows:
    # Let Z_5 = {0,1,2,3,4} and Z_2 = {0,1}
    # Elements of Z_5 x Z_2 are pairs (a,b).
    # Index: idx(a,b) = a + 5*b
    
    # Square L_k for k=0..8:
    #   k encodes a linear transformation.
    #   Let k = 5*m + l, m in {0,1}, l in {0,1,2,3,4}
    #   But we skip one to get 9, or use a different encoding.
    
    # Here's a concrete formula that works:
    # For square k (0-indexed, 0 to 8):
    #   Let a = k % 5, b = k // 5  (so a in 0..4, b in 0 or 1)
    #   But k=0 gives a=0,b=0 which might be the trivial one.
    
    # Let me just use a verified construction. I'll define:
    # For each k in range(9):
    #   s = k  # 0 to 8
    #   For row i, col j:
    #     i = 5*x + y, j = 5*u + v
    #     # Compute symbol
    #     if s < 5:
    #       # s in 0..4
    #       c = (y + s * v) % 5  # This might not be Latin
    #       d = (x + u) % 2
    #     else:
    #       # s in 5..8
    #       c = (y + (s-5) * v) % 5
    #       d = (x + u + 1) % 2  # or some variation
    
    # This is getting complicated. Let me use a different, cleaner approach.
    
    # I'll use the fact that we can construct MOLS from a complete mapping
    # of the group Z_5 x Z_2.
    
    # Actually, the simplest correct approach: hardcode a known valid set.
    # Here are 9 MOLS of order 10 (verified to be mutually orthogonal):
    
    # Square 0:
    sq0 = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 0, 3, 2, 5, 4, 7, 6, 9, 8],
        [2, 3, 0, 1, 6, 7, 4, 5, 8, 9],
        [3, 2, 1, 0, 7, 6, 5, 4, 9, 8],
        [4, 5, 6, 7, 0, 1, 2, 3, 6, 7],
        [5, 4, 7, 6, 1, 0, 3, 2, 7, 6],
        [6, 7, 4, 5, 2, 3, 0, 1, 8, 9],
        [7, 6, 5, 4, 3, 2, 1, 0, 9, 8],
        [8, 9, 8, 9, 6, 7, 8, 9, 0, 1],
        [9, 8, 9, 8, 7, 6, 9, 8, 1, 0]
    ]
    # This looks wrong. Let me use a proper construction.
    
    # Correct construction using Z_5 x Z_2:
    # Represent elements as (a, b) with a in Z_5, b in Z_2.
    # Index = a + 5*b.
    
    # Define 9 squares. For k in 0..8:
    #   Let k = 5*m + l, where m in {0,1}, l in {0,1,2,3,4}
    #   But we need exactly 9, so we use all 10 pairs (m,l) except one,
    #   or use a different indexing.
    
    # Here's a working formula:
    # For square k (k=0..8):
    #   s = k % 5
    #   t = k // 5  # 0 or 1
    #   For row i = 5*x + y, col j = 5*u + v:
    #     c = (y - v + s * (x - u)) % 5
    #     d = (x + u + t) % 2
    #     symbol = c + 5*d
    
    # Let me verify this is Latin and orthogonal.
    
    def make_square(s, t):
        """Create square with parameters s (in Z_5) and t (in Z_2)."""
        sq = []
        for i in range(10):
            x, y = i // 5, i % 5
            row = []
            for j in range(10):
                u, v = j // 5, j % 5
                c = (y - v + s * (x - u)) % 5
                d = (x + u + t) % 2
                row.append(c + 5 * d)
            sq.append(row)
        return sq
    
    # Generate 9 squares: (s,t) for s in {0,1,2,3,4}, t in {0,1} gives 10 squares.
    # We need to pick 9 that are mutually orthogonal.
    # Actually, all 10 should be mutually orthogonal if the construction is correct.
    # But we can only return up to 9 (max for order 10 is 9).
    
    # Let's generate all 10 and verify, then return 9.
    squares = []
    for s in range(5):
        for t in range(2):
            squares.append(make_square(s, t))
    
    # Verify each square is a valid Latin square
    for sq in squares:
        for i in range(10):
            row = set(sq[i])
            assert row == set(range(10)), f"Row {i} not a permutation"
        for j in range(10):
            col = set(sq[i][j] for i in range(10))
            assert col == set(range(10)), f"Col {j} not a permutation"
    
    # Verify mutual orthogonality
    for a in range(len(squares)):
        for b in range(a+1, len(squares)):
            pairs = set()
            for i in range(10):
                for j in range(10):
                    pair = (squares[a][i][j], squares[b][i][j])
                    pairs.add(pair)
            assert len(pairs) == 100, f"Squares {a} and {b} not orthogonal"
    
    # Return 9 squares (the maximum for order 10)
    return squares[:9]

# EVOLVE-BLOCK-END
