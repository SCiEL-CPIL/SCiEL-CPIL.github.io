"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """Construct mutually orthogonal Latin squares of order 10.
    
    Uses the algebraic structure Z_10 ≅ Z_2 × Z_5 to construct
    two orthogonal Latin squares.
    """
    n = 10
    deadline = time.time() + 55.0  # Safety margin before 60s limit
    
    # Represent indices and symbols as pairs (a, b) where a in Z_2, b in Z_5
    # Map: idx = a*5 + b, sym = a*5 + b
    
    def pair_to_int(a, b):
        return a * 5 + b
    
    def int_to_pair(x):
        return (x // 5, x % 5)
    
    # Construct first Latin square L0:
    # L0[(r1,r2)][(c1,c2)] = (r1+c1 mod 2, r2+c2 mod 5)
    L0 = [[0]*n for _ in range(n)]
    for i in range(n):
        r1, r2 = int_to_pair(i)
        for j in range(n):
            c1, c2 = int_to_pair(j)
            L0[i][j] = pair_to_int((r1+c1) % 2, (r2+c2) % 5)
    
    # Construct second Latin square L1:
    # L1[(r1,r2)][(c1,c2)] = (r1+c1 mod 2, r2 + r1*c1 + c2 mod 5)
    # Wait, this needs to be a valid Latin square. Let me use a different approach.
    # 
    # For the second square, use: L1[(r1,r2)][(c1,c2)] = (r1+c1 mod 2, r2 + k*r1 + c2 mod 5)
    # where k is a non-zero element of Z_5.
    
    k = 1
    L1 = [[0]*n for _ in range(n)]
    for i in range(n):
        r1, r2 = int_to_pair(i)
        for j in range(n):
            c1, c2 = int_to_pair(j)
            L1[i][j] = pair_to_int((r1+c1) % 2, (r2 + k*r1 + c2) % 5)
    
    # Verify both are Latin squares
    def is_latin(sq):
        for i in range(n):
            row = set(sq[i])
            if len(row) != n:
                return False
            col = set(sq[i][j] for i in range(n))
            if len(col) != n:
                return False
        return True
    
    # Verify orthogonality
    def are_orthogonal(A, B):
        pairs = set()
        for i in range(n):
            for j in range(n):
                pair = (A[i][j], B[i][j])
                if pair in pairs:
                    return False
                pairs.add(pair)
        return len(pairs) == n*n
    
    # Check validity
    if not is_latin(L0) or not is_latin(L1):
        # Fallback to single square
        return [L0]
    
    if not are_orthogonal(L0, L1):
        # Try different k values
        for k_try in range(1, 5):
            L1_try = [[0]*n for _ in range(n)]
            for i in range(n):
                r1, r2 = int_to_pair(i)
                for j in range(n):
                    c1, c2 = int_to_pair(j)
                    L1_try[i][j] = pair_to_int((r1+c1) % 2, (r2 + k_try*r1 + c2) % 5)
            if is_latin(L1_try) and are_orthogonal(L0, L1_try):
                L1 = L1_try
                break
        else:
            # No valid second square found, return just L0
            return [L0]
    
    # Ensure all values are Python ints
    L0 = [[int(v) for v in row] for row in L0]
    L1 = [[int(v) for v in row] for row in L1]
    
    return [L0, L1]
# EVOLVE-BLOCK-END
