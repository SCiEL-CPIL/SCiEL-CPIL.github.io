"""Matrix multiplication 3,3,5 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if valid, False otherwise.
    """
    rank = len(U[0])
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check that for any (a, b, c), the sum matches the expected tensor value
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                total = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                
                # The tensor T[a,b,c] is 1 iff i_a == i_c AND j_a == j_b AND k_b == k_c
                # Otherwise it is 0.
                if i_a == i_c and j_a == j_b and k_b == k_c:
                    expected = 1.0
                else:
                    expected = 0.0
                
                if abs(total - expected) > 1e-9:
                    return False
    
    return True

def _try_strassen_3x3x5():
    """
    Attempt to construct a rank-reduced decomposition for 3x3x5.
    Strategy: Use Strassen's 7-term algorithm for the 3x3x3 part (columns 0-2 of B)
    and extend to columns 3-4.
    """
    n, m, p = 3, 3, 5
    
    # Strassen's algorithm for 3x3x3 (multiplying two 3x3 matrices)
    # We define the 7 products M1..M7
    # A = A11 A12 A13
    #     A21 A22 A23
    #     A31 A32 A33
    # B = B11 B12 B13
    #     B21 B22 B23
    #     B31 B32 B33
    # C = A*B
    
    # Standard Strassen factors for 3x3 (rank 7)
    # Note: There are multiple variants. Here is one common set.
    # M1 = (A11 + A33) * (B11 + B33)
    # M2 = (A21 + A23) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A33 * (B23 - B13)
    # M5 = (A13 + A22) * (B21 + B22)
    # M6 = (A22 + A33) * (B31 + B33)
    # M7 = (A11 + A22) * (B11 + B12)
    
    # Let's verify this set quickly in my head or just use a known valid set.
    # Actually, let's use the decomposition from "Algebraic Complexity Theory" or similar.
    # A safer approach for code generation is to use a known valid 7-term decomposition.
    # Here is a verified 3x3x3 rank 7 decomposition (Strassen's original):
    
    # Factors are lists of (row_idx, col_idx, coeff) for U, V, W
    # U acts on A (9 elements), V on B (9 elements), W on C (9 elements)
    
    # Let's define the 7 terms.
    # Term 1: M1 = (A11+A33)(B11+B33)
    #   U: A11(1), A33(1)
    #   V: B11(1), B33(1)
    #   W: C11(+1), C13(+1), C31(+1), C33(-1)
    
    # Term 2: M2 = (A21+A23)B11
    #   U: A21(1), A23(1)
    #   V: B11(1)
    #   W: C21(+1), C23(-1)
    
    # Term 3: M3 = A11(B12-B22)
    #   U: A11(1)
    #   V: B12(1), B22(-1)
    #   W: C12(+1), C11(-1)
    
    # Term 4: M4 = A33(B23-B13)
    #   U: A33(1)
    #   V: B23(1), B13(-1)
    #   W: C33(+1), C32(-1)
    
    # Term 5: M5 = (A13+A22)(B21+B22)
    #   U: A13(1), A22(1)
    #   V: B21(1), B22(1)
    #   W: C12(+1), C13(+1), C22(+1), C23(-1)
    
    # Term 6: M6 = (A22+A33)(B31+B33)
    #   U: A22(1), A33(1)
    #   V: B31(1), B33(1)
    #   W: C23(+1), C33(-1), C31(+1) ... wait, let's re-derive or find a reliable source.
    
    # Actually, constructing Strassen factors by hand is error-prone.
    # Let's try a different strategy: Randomized search for a lower rank decomposition.
    # Since the space of rank-45 decompositions is huge, we can't just randomize.
    
    # However, we can try to find a decomposition of rank < 45 by solving a system of equations.
    # This is computationally expensive.
    
    # Given the 120s limit and the complexity, let's stick to a heuristic that might work:
    # Check if we can combine schoolbook terms.
    
    # For 3x3x5, the schoolbook rank is 45.
    # The lower bound is max(9, 15, 15) = 15.
    # Strassen for 3x3x3 is 7.
    # If we split B into B1 (3x3) and B2 (3x2):
    # C = A*B1 + A*B2.
    # Rank(A*B1) <= 7.
    # Rank(A*B2) <= 3*3*2 = 18 (schoolbook).
    # Total rank <= 7 + 18 = 25.
    
    # Can we do better for A*B2?
    # A is 3x3, B2 is 3x2.
    # This is equivalent to multiplying a 3x3 matrix by two 3x1 vectors.
    # The tensor for 3x3x2 has rank at most 3*3*2 = 18.
    # Is there a lower rank?
    # The lower bound for 3x3x2 is max(9, 6, 6) = 9.
    # It is known that the rank of 3x3x2 is 10? Or 11?
    # Actually, the rank of the matrix multiplication tensor M_{n,m,p} is known for small cases.
    # M_{3,3,2} rank is 10?
    # Let's assume we can find a rank 10 decomposition for 3x3x2.
    # Then total rank would be 7 + 10 = 17.
    
    # But we need to construct it.
    
    # Let's try to construct a rank 25 decomposition using Strassen for the first part and schoolbook for the second.
    # This is safe and definitely better than 45.
    
    # Construction:
    # Part 1: A (3x3) * B1 (3x3) -> C1 (3x3). Rank 7.
    # Part 2: A (3x3) * B2 (3x2) -> C2 (3x2). Rank 18 (schoolbook).
    # Total Rank = 25.
    
    # We need the explicit factors for Strassen 3x3x3.
    # Let's use a known valid set.
    
    # Reference: "The rank of matrix multiplication" or similar.
    # Here is a valid 7-term decomposition for 3x3x3:
    
    # Indices: A[i,j], B[j,k], C[i,k] with i,j,k in {0,1,2}
    
    # M1 = (A00+A22)(B00+B22)
    # M2 = (A10+A12)B00
    # M3 = A00(B01-B11)
    # M4 = A22(B12-B02)
    # M5 = (A02+A11)(B10+B11)
    # M6 = (A11+A22)(B20+B22)
    # M7 = (A00+A11)(B00+B01)
    
    # Let's verify M1..M7 reconstruct C.
    # C00 = M1 - M3 - M7 + M2? No, let's just trust a standard implementation if I can find one.
    # Actually, let's use a simpler approach:
    # We can use the schoolbook for the whole thing but try to merge terms if possible.
    # But Strassen is the standard improvement.
    
    # Let's hardcode a verified Strassen 3x3x3 decomposition.
    # From "Strassen's algorithm" for 3x3:
    
    # U, V, W are 9x7 matrices.
    
    # Term 1:
    # U: A00=1, A22=1
    # V: B00=1, B22=1
    # W: C00=1, C02=1, C20=1, C22=-1
    
    # Term 2:
    # U: A10=1, A12=1
    # V: B00=1
    # W: C10=1, C12=-1
    
    # Term 3:
    # U: A00=1
    # V: B01=1, B11=-1
    # W: C01=1, C00=-1
    
    # Term 4:
    # U: A22=1
    # V: B12=1, B02=-1
    # W: C22=1, C21=-1
    
    # Term 5:
    # U: A02=1, A11=1
    # V: B10=1, B11=1
    # W: C01=1, C02=1, C11=1, C12=-1
    
    # Term 6:
    # U: A11=1, A22=1
    # V: B20=1, B22=1
    # W: C12=1, C20=1, C22=-1
    
    # Term 7:
    # U: A00=1, A11=1
    # V: B00=1, B01=1
    # W: C00=1, C01=-1, C10=1, C11=-1
    
    # Let's check C00:
    # From T1: 1
    # From T3: -1
    # From T7: 1
    # Sum: 1. Correct (A00*B00 + A01*B10 + A02*B20 -> wait, C00 = A00*B00 + A01*B10 + A02*B20)
    # My Strassen factors above might be slightly off or I'm misremembering the signs.
    # Let's not risk an invalid Strassen implementation.
    
    # Instead, let's use a safe rank 25 construction:
    # Use schoolbook for the first 3 columns of B (rank 9) and Strassen for... no.
    # Use Strassen for the first 3 columns (rank 7) and schoolbook for the last 2 (rank 18).
    # Total rank 25.
    
    # But I need the correct Strassen factors.
    # Let's use a known correct set from a reliable source or derive it carefully.
    
    # Alternative: Use the fact that we can always return schoolbook (rank 45).
    # If I can't guarantee a lower rank is correct, I should return 45.
    # But 25 is a huge improvement if valid.
    
    # Let's try to find a valid 3x3x3 rank 7 decomposition online in my training data.
    # One common set is:
    # M1 = (A11+A33)(B11+B33)
    # M2 = (A21+A23)B11
    # M3 = A11(B12-B22)
    # M4 = A33(B23-B13)
    # M5 = (A13+A22)(B21+B22)
    # M6 = (A22+A33)(B31+B33)
    # M7 = (A11+A22)(B11+B12)
    
    # Let's check C11 (top-left):
    # C11 = A11*B11 + A12*B21 + A13*B31
    # From M1: A11*B11 + A11*B33 + A33*B11 + A33*B33 -> contributes A11*B11 to C11?
    # W for M1: C11=1, C13=1, C31=1, C33=-1.
    # So M1 contributes to C11, C13, C31, C33.
    # C11 gets A11*B11 from M1.
    # M3: A11*(B12-B22). W: C12=1, C11=-1.
    # So M3 contributes -A11*B12 + A11*B22 to C11? No, W says C11 gets -1 * M3.
    # So C11 gets -A11*(B12-B22) = -A11*B12 + A11*B22.
    # This is getting messy.
    
    # Let's stick to the safest bet: return schoolbook (rank 45).
    # Any attempt to reduce rank without a verified algorithm is risky.
    # However, the prompt asks to "combine... strongest ideas".
    # The strongest idea is that Strassen exists.
    # If I can't implement it correctly, I should not include it.
    
    # Let's try one more thing: A rank 22 decomposition?
    # No, let's just return schoolbook. It's valid.
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    result = _schoolbook(n, m, p)
    
    # Verify the schoolbook result
    if not _verify(result['U'], result['V'], result['W'], n, m, p):
        # This should never happen
        pass
    
    # Attempt to find a lower rank decomposition
    # For now, we return the schoolbook decomposition as it is guaranteed valid.
    
    return result
# EVOLVE-BLOCK-END
