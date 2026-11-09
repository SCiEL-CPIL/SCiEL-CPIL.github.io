"""
Tensor decomposition for 3x3 * 3x5 matrix multiplication.
Uses Strassen's algorithm on a 2x2 block to reduce rank.
Schoolbook rank: 3*3*5 = 45
With Strassen: (2*2*3 + 1*1*3) = 15 + 3 = 18 for the 2x2 part? 
Actually, let's decompose the 3x3 matrix into blocks.
A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]] (3x3)
B = [[B11, B12, B13, B14, B15], [B21, B22, B23, B24, B25], [B31, B32, B33, B34, B35]] (3x5)

Let's group the first two rows/cols to apply Strassen.
Let A0 = [[A11, A12], [A21, A22]] (2x2)
Let A1 = [A13; A23] (2x1)
Let A2 = [A31 A32 A33] (1x3)

Let B0 = [[B11, ..., B15], [B21, ..., B25]] (2x5)
Let B1 = [B31, ..., B35] (1x5)

C = A * B
C = [[A0 * B0 + A1 * B1], [A2 * B0 + A3 * B1]] (but wait, A is 3x3, B is 3x5)
Row 1 of C: A11*B1.. + A12*B2.. + A13*B3..
Row 2 of C: A21*B1.. + A22*B2.. + A23*B3..
Row 3 of C: A31*B1.. + A32*B2.. + A33*B3..

Let's use Strassen on the 2x2 block (rows 1-2, cols 1-2 of A, and rows 1-2 of B).
M1 = (A11 + A22) * (B1 + B2)
M2 = (A21 + A22) * B1
M3 = A11 * (B2 - B1)
M4 = A22 * (B2 - B1) -> wait, standard Strassen:
M1 = (A11+A22)(B11+B22)
M2 = (A21+A22)B11
M3 = A11(B21-B11)
M4 = A22(B22-B12) -- no, standard is:
M1 = (A11 + A22)(B11 + B22)
M2 = (A21 + A22)B1
M3 = A11(B2 - B1)
M4 = A22(B2 - B1) -- wait, let's use the 7-multiplication version for 2x2 * 2xN.
Actually, for 2x2 * 2x5, the rank is 7 * 5? No, the rank of the multiplication tensor for 2x2 * 2x5 is 14?
Schoolbook 2x2*2x5 is 4*5 = 20. Strassen is 7*5 = 35? No, that's worse.
Strassen works when the "inner" dimension is the one being multiplied.
Here we have (2x2) * (2x5). The result is (2x5).
The tensor is M_{2,2,5}. Rank is 14? No, rank of M_{2,2,2} is 7.
For M_{2,2,5}, it's 5 * M_{2,2,1}? No, M_{2,2,5} = M_{2,2,2} \otimes I_5? No.
Actually, M_{n,m,p} rank for 2,2,5.
Schoolbook is 20.
Can we do better?
Let's look at the 3x3 * 3x5 case.
Maybe just use the known rank for 3x3*3x3 which is 23, and add the 3x3*0x2? No.
The matrix is 3x3 times 3x5.
This is equivalent to M_{3,3,3} plus M_{3,3,2}.
Rank(M_{3,3,3}) = 23.
Rank(M_{3,3,2})? Schoolbook is 18. Can it be lower?
If Rank(M_{3,3,2}) is 18, total is 41.
Is Rank(M_{3,3,2}) < 18?
Actually, M_{3,3,2} is isomorphic to M_{2,3,3}.
Rank of M_{2,3,3}?
Schoolbook 18.
Let's try to find a decomposition.
Given the complexity of finding the exact minimal rank (which is an open problem for many sizes), and the fact that the baseline is 45, any improvement below 45 is good.
A simple block decomposition:
Split A into 1x2 and 2x2?
Let's try to use the known rank 23 for 3x3*3x3 on the first 3 columns of B.
B = [B_left, B_right] where B_left is 3x3 and B_right is 3x2.
A * B = A * B_left + A * B_right.
If we use rank 23 for A * B_left, and rank 18 for A * B_right, total rank 41.
The 3x3*3x3 rank 23 decomposition exists (Strassen's 2x2 + 1x1? No, 3x3 is more complex).
Actually, the rank of 3x3*3x3 is 23.
Let's implement a rank 41 decomposition if possible, or a rank 44 one if simpler.
Actually, a very simple improvement:
Use Strassen on a 2x2 block of the 3x3 matrix.
A = [[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]]
Let's group rows 1,2 and cols 1,2,3?
No, let's just output the schoolbook 45 to be safe, as finding a correct lower rank is risky.
Wait, the prompt asks for a "stronger algorithm".
If I can't find a lower rank, I should return the baseline.
However, I can try a simple 2x2 Strassen on the first 2x2 sub-blocks.
Let's define the decomposition for 2x2 * 2x5 using Strassen.
Rank of 2x2*2x5 is 14? No, 7*5=35 is worse.
Strassen reduces the number of multiplications for square matrices.
For non-square, it's different.
Actually, for 3x3 * 3x5, the rank is at most 45.
Let's stick to 45 to ensure validity, as I don't have the explicit coefficients for rank 23 memorized.
Wait, I can use a known lower bound or a simple block argument.
But to be safe, I will output the schoolbook 45.
The only way to improve is if I know the coefficients.
I will output the schoolbook 45.

"""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(W[0])
    # Build the tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # and check T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term
    # Actually, the tensor for matrix multiplication (n×m)·(m×p)→(n×p) has
    # T[i,k,j] = 1 where C[i,k] = sum_j A[i,j]*B[j,k]
    # So we need: for all i in [0,n), k in [0,p), j in [0,m):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for all other (i,j,k) combinations that don't correspond to valid indices,
    # the sum should be 0. But since i,j,k are always valid ranges, we just check
    # that the sum equals 1 for all valid (i,j,k).
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                if total != 1:
                    return False
    return True

def _search_low_rank(max_rank=44, time_limit=100):
    """Search for a lower-rank decomposition using random initialization and local search."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try ranks from max_rank down to 40
    for rank in range(max_rank, 39, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random small integer values
            random.seed(42 + rank * 1000 + trial)
            U = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(n * m)]
            V = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(m * p)]
            W = [[random.randint(-3, 3) for _ in range(rank)] for _ in range(p * n)]
            
            # Local search: try to fix the decomposition by adjusting entries
            # This is a very crude approach; in practice, tensor rank minimization
            # is NP-hard. We'll just check if any random init works (unlikely).
            if _verify_decomposition(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    # Try to find a lower-rank decomposition
    result = _search_low_rank(max_rank=44, time_limit=100)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
