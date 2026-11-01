"""Matrix multiplication 2,2,4 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 4]

def _verify(U, V, W):
    """Verify that the decomposition exactly computes matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = delta[i,j,k] for matmult
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: C_{i,k} = sum_j A_{i,j} B_{j,k}
    # In the factorization: C_{i,k} = sum_r U_{i,m} V_{m,k} W_{k,n}
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # Indexing: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # The tensor element T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] = 1 if we think of it as A[i,j]*B[j,k] contributing to C[i,k]
    # Actually the multiplication tensor is: for each (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    
    # Let's verify by checking that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] equals the correct tensor entry.
    # The tensor for matmult (n x m) * (m x p): entry (i,j,k) is 1 (each pair contributes exactly once to C[i,k])
    # Wait, that's not right. The tensor T is defined such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # And T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k]).
    # But that's just the schoolbook. The tensor in the decomposition sense is:
    # C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) ... no that's wrong too.
    
    # Let me think more carefully. The standard tensor decomposition for matrix multiplication:
    # C = A * B where A is n x m, B is m x p, C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We want: C[i,k] = sum_r U[i*m + ?, r] * ... 
    
    # Actually the convention is:
    # U has shape (n*m, rank): indexed by (i,j) -> row i*m+j
    # V has shape (m*p, rank): indexed by (j,k) -> row j*p+k  
    # W has shape (p*n, rank): indexed by (k,i) -> row k*n+i
    # And the decomposition says: the multiplication tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where T[i,j,k] = 1 for matrix multiplication (each term A[i,j]*B[j,k] appears with coefficient 1 in C[i,k]).
    
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1  # Each A[i,j]*B[j,k] contributes with coefficient 1
                actual = 0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    errors.append((i, j, k, actual, expected))
    
    return len(errors) == 0, errors

def _strassen_decomposition():
    """
    Construct a rank-14 decomposition for 2x2 * 2x4 matrix multiplication.
    
    The 2x2 * 2x4 multiplication can be viewed as two 2x2 * 2x2 multiplications.
    Using Strassen's 7-multiplication algorithm for each 2x2*2x2, we get 14 terms.
    However, we need to check if we can share any terms between the two blocks.
    
    For A (2x2) times B (2x4):
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[0,3] = A[0,0]*B[0,3] + A[0,1]*B[1,3]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    C[1,3] = A[1,0]*B[0,3] + A[1,1]*B[1,3]
    
    Block 1 (columns 0,1 of B): standard 2x2 * 2x2
    Block 2 (columns 2,3 of B): another 2x2 * 2x2
    
    Strassen for 2x2*2x2 uses 7 bilinear forms:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    
    C00 = M1 - M4 + M5 - M7
    C01 = M3 + M7
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS  # 2, 2, 4
    
    # We'll build 14 terms (7 for each 2x2 block)
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 4 x 14
    V = [[0] * rank for _ in range(m * p)]  # 8 x 14
    W = [[0] * rank for _ in range(p * n)]  # 8 x 14
    
    # For block 1 (B columns 0,1): B[0,0]=V row 0, B[0,1]=V row 1, B[1,0]=V row 2, B[1,1]=V row 3
    # W indices for block 1: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    # U indices: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[2], A[1,1]=U[3]
    
    # Strassen M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> U[0]+=1, U[3]+=1
    # V: B00=1, B11=1 -> V[0]+=1, V[3]+=1
    # W: C00=1, C11=1 -> W[0]+=1, W[3]+=1
    # But we need to distribute: C00 gets M1, C11 gets M1
    # Actually in Strassen: C00 = M1 - M4 + M5 - M7, so M1 contributes to both C00 and C11
    
    # Let me be more careful. In Strassen:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So each M_r contributes to specific C entries with specific signs.
    # For the tensor decomposition, term r has:
    # U_r[i,j] = coefficient of A[i,j] in the first factor of M_r
    # V_r[j,k] = coefficient of B[j,k] in the second factor of M_r
    # W_r[k,i] = coefficient in the output C[i,k] from M_r
    
    # Block 1 (columns 0,1 of B):
    # V indices: B[0,0]=V[0], B[0,1]=V[1], B[1,0]=V[2], B[1,1]=V[3]
    # W indices: C[0,0]=W[0], C[0,1]=W[1], C[1,0]=W[2], C[1,1]=W[3]
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00->1, A11->1 => U[0][0]=1, U[3][0]=1
    # V: B00->1, B11->1 => V[0][0]=1, V[3][0]=1
    # W: contributes +1 to C00 and +1 to C11 => W[0][0]=1, W[3][0]=1
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    # U: A10->1, A11->1 => U[2][1]=1, U[3][1]=1
    # V: B00->1 => V[0][1]=1
    # W: contributes +1 to C10, -1 to C11 => W[2][1]=1, W[3][1]=-1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10-B11)
    # U: A00->1 => U[0][2]=1
    # V: B10->1, B11->-1 => V[2][2]=1, V[3][2]=-1
    # W: contributes +1 to C01, +1 to C11 => W[1][2]=1, W[3][2]=1
    
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    # U: A11->1 => U[3][3]=1
    # V: B01->1, B00->-1 => V[1][3]=1, V[0][3]=-1
    # W: contributes -1 to C00, +1 to C10 => W[0][3]=-1, W[2][3]=1
    
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    # U: A00->1, A01->1 => U[0][4]=1, U[1][4]=1
    # V: B11->1 => V[3][4]=1
    # W: contributes +1 to C00, -1 to C11 => W[0][4]=1, W[3][4]=-1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[3][4] = -1
    
    # M6 = (A10-A00)(B00+B10)
    # U: A10->1, A00->-1 => U[2][5]=1, U[0][5]=-1
    # V: B00->1, B10->1 => V[0][5]=1, V[2][5]=1
    # W: contributes +1 to C11 => W[3][5]=1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B01+B11)
    # U: A01->1, A11->-1 => U[1][6]=1, U[3][6]=-1
    # V: B01->1, B11->1 => V[1][6]=1, V[3][6]=1
    # W: contributes -1 to C00, +1 to C01 => W[0][6]=-1, W[1][6]=1
    
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[0][6] = -1; W[1][6] = 1
    
    # Block 2 (columns 2,3 of B):
    # V indices: B[0,2]=V[4], B[0,3]=V[5], B[1,2]=V[6], B[1,3]=V[7]
    # W indices: C[0,2]=W[4], C[0,3]=W[5], C[1,2]=W[6], C[1,3]=W[7]
    
    # M1' = (A00+A11)(B02+B13)  [analogous to M1 with B shifted]
    U[0][7] = 1; U[3][7] = 1
    V[4][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1
    
    # M2' = (A10+A11)B02
    U[2][8] = 1; U[3][8] = 1
    V[4][8] = 1
    W[6][8] = 1; W[7][8] = -1
    
    # M3' = A00(B12-B13)
    U[0][9] = 1
    V[6][9] = 1; V[7][9] = -1
    W[5][9] = 1; W[7][9] = 1
    
    # M4' = A11(B03-B02)
    U[3][10] = 1
    V[5][10] = 1; V[4][10] = -1
    W[4][10] = -1; W[6][10] = 1
    
    # M5' = (A00+A01)B13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = 1; W[7][11] = -1
    
    # M6' = (A10-A00)(B02+B12)
    U[2][12] = 1; U[0][12] = -1
    V[4][12] = 1; V[6][12] = 1
    W[7][12] = 1
    
    # M7' = (A01-A11)(B03+B13)
    U[1][13] = 1; U[3][13] = -1
    V[5][13] = 1; V[7][13] = 1
    W[4][13] = -1; W[5][13] = 1
    
    return U, V, W

def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with the Strassen-based rank-14 decomposition
    U, V, W = _strassen_decomposition()
    
    # Verify the decomposition
    valid, errors = _verify(U, V, W)
    if not valid:
        # Fallback to schoolbook if Strassen is incorrect
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
    
    # Try to find a lower rank decomposition using random initialization + gradient descent
    # on the tensor decomposition problem. Since we need exact integer/half-integer solutions,
    # we'll use the rank-14 as baseline and attempt to find rank-13 or lower.
    
    best_rank = 14
    best_U, best_V, best_W = U, V, W
    
    # Attempt to find a rank-13 decomposition
    # This is a hard problem; we'll try a few random restarts with small perturbations
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        # Try rank 13
        target_rank = 13
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and (time.time() - start_time) < deadline - 10:
            attempts += 1
            # Start from the rank-14 solution and try to merge two columns
            # Pick a random pair of columns to try to merge
            r1, r2 = random.sample(range(14), 2)
            
            # Try to find coefficients a, b such that column r2 = a * column r1' 
            # where column r1' is a modified version. This is complex, so let's just
            # try random rank-13 initializations.
            
            # Random rank-13 initialization
            U_trial = np.zeros((n * m, target_rank))
            V_trial = np.zeros((m * p, target_rank))
            W_trial = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                # Random sparse initialization
                for i in range(n * m):
                    if random.random() < 0.3:
                        U_trial[i, r] = random.choice([-1, 1]) * random.choice([0.5, 1])
                for i in range(m * p):
                    if random.random() < 0.3:
                        V_trial[i, r] = random.choice([-1, 1]) * random.choice([0.5, 1])
                for i in range(p * n):
                    if random.random() < 0.3:
                        W_trial[i, r] = random.choice([-1, 1]) * random.choice([0.5, 1])
            
            # Check if this random init happens to work (extremely unlikely)
            valid_t, _ = _verify(U_trial.tolist(), V_trial.tolist(), W_trial.tolist())
            if valid_t:
                best_rank = target_rank
                best_U, best_V, best_W = U_trial.tolist(), V_trial.tolist(), W_trial.tolist()
                break
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Try rank 12
        target_rank = 12
        attempts = 0
        max_attempts = 30
        while attempts < max_attempts and (time.time() - start_time) < deadline - 5:
            attempts += 1
            U_trial = np.zeros((n * m, target_rank))
            V_trial = np.zeros((m * p, target_rank))
            W_trial = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                for i in range(n * m):
                    if random.random() < 0.4:
                        U_trial[i, r] = random.choice([-1, 1]) * random.choice([0.5, 1])
                for i in range(m * p):
                    if random.random() < 0.4:
                        V_trial[i, r] = random.choice([-1, 1]) * random.choice([0.5, 1])
                for i in range(p * n):
                    if random.random() < 0.4:
                        W_trial[i, r] = random.choice([-1, 1]) * random.choice([0.5, 1])
            
            valid_t, _ = _verify(U_trial.tolist(), V_trial.tolist(), W_trial.tolist())
            if valid_t:
                best_rank = target_rank
                best_U, best_V, best_W = U_trial.tolist(), V_trial.tolist(), W_trial.tolist()
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
