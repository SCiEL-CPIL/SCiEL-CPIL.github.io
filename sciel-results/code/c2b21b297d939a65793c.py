import numpy as np
import time
import sys

DIMENSIONS = [2, 4, 8]
DEADLINE = 145  # seconds, leave 5s safety margin

def construct():
    """
    Rank-56 tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Uses block-Strassen: splits the 4-dimensional inner space into 2 groups of 2,
    and the 8-dimensional output into 4 groups of 2, yielding 8 independent 2x2*2x2
    sub-problems each solved with rank-7 Strassen.
    
    Output factor follows the official trace convention W[k*n+i,r].
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    rank = 56
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    for s in range(2):  # s=0,1: split A columns / B rows into 2x2 blocks
        for j in range(4):  # j=0..3: split B columns into 2x2 blocks
            # A_s: rows 0-1, cols 2s, 2s+1 of A (2x4)
            a00 = 0 * m + (2*s + 0)
            a01 = 0 * m + (2*s + 1)
            a10 = 1 * m + (2*s + 0)
            a11 = 1 * m + (2*s + 1)
            
            # B_{s,j}: rows 2s, 2s+1; cols 2j, 2j+1 of B (4x8)
            b00 = (2*s + 0) * p + (2*j + 0)
            b01 = (2*s + 0) * p + (2*j + 1)
            b10 = (2*s + 1) * p + (2*j + 0)
            b11 = (2*s + 1) * p + (2*j + 1)
            
            # C block: rows 0-1, cols 2j, 2j+1; global index = col*n + row
            c00 = (2*j + 0) * n + 0
            c01 = (2*j + 1) * n + 0
            c10 = (2*j + 0) * n + 1
            c11 = (2*j + 1) * n + 1
            
            # Strassen U (A-side) coefficients
            # m1 = (a00+a11)*(b00+b11)
            U[a00][col] += 1
            U[a11][col] += 1
            # m2 = (a10+a11)*b00
            U[a10][col+1] += 1
            U[a11][col+1] += 1
            # m3 = a00*(b01-b11)
            U[a00][col+2] += 1
            # m4 = a11*b10
            U[a11][col+3] += 1
            # m5 = (a00+a01)*b11
            U[a00][col+4] += 1
            U[a01][col+4] += 1
            # m6 = (a10-a00)*(b00+b01)
            U[a10][col+5] += 1
            U[a00][col+5] += -1
            # m7 = (a01-a11)*(b10+b11)
            U[a01][col+6] += 1
            U[a11][col+6] += -1
            
            # Strassen V (B-side) coefficients
            V[b00][col] += 1
            V[b11][col] += 1
            V[b00][col+1] += 1
            V[b01][col+2] += 1
            V[b11][col+2] += -1
            V[b10][col+3] += 1
            V[b00][col+3] += -1
            V[b11][col+4] += 1
            V[b00][col+5] += 1
            V[b01][col+5] += 1
            V[b10][col+6] += 1
            V[b11][col+6] += 1
            
            # Strassen W (C-side) coefficients
            # c00 = m1 + m4 - m5 + m7
            W[c00][col] += 1
            W[c00][col+3] += 1
            W[c00][col+4] += -1
            W[c00][col+6] += 1
            # c01 = m3 + m5
            W[c01][col+2] += 1
            W[c01][col+4] += 1
            # c10 = m2 + m4
            W[c10][col+1] += 1
            W[c10][col+3] += 1
            # c11 = m1 - m2 + m3 + m6
            W[c11][col] += 1
            W[c11][col+1] += -1
            W[c11][col+2] += 1
            W[c11][col+5] += 1
            
            col += 7
    
    # Check deadline before attempting any improvement search
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return {"U": U, "V": V, "W": W}
    
    # Attempt to find a lower-rank decomposition using the existing structure
    # Try to merge Strassen blocks that share structure
    # The key insight: if we can find linear dependencies among the rank-56 vectors,
    # we can reduce the rank.
    
    # Convert to numpy for efficient linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The tensor T is defined by T[i,j,k] = U[i,r] * V[j,r] * W[k,r] summed over r
    # We need to verify the decomposition is correct first, then try to reduce rank
    
    # For 2x4 * 4x8:
    # A is 2x4, B is 4x8, C is 2x8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: C[i,k] = sum_j sum_r U[i*4+j, r] * V[j*8+k, r] * W[k*2+i, r]
    
    # Verify correctness with random matrices
    np.random.seed(42)
    A_test = np.random.randint(-2, 3, size=(2, 4))
    B_test = np.random.randint(-2, 3, size=(4, 8))
    C_test = A_test @ B_test
    
    # Compute using decomposition
    C_decomp = np.zeros((2, 8))
    for i in range(2):
        for k in range(8):
            val = 0
            for j in range(4):
                for r in range(56):
                    val += U_np[i*4+j, r] * V_np[j*8+k, r] * W_np[k*2+i, r]
            C_decomp[i, k] = val
    
    if not np.allclose(C_test, C_decomp):
        # Decomposition is incorrect, return as-is
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank reduction by checking for linear dependencies
    # among the rank vectors
    # Each rank vector is a triple (U[:,r], V[:,r], W[:,r])
    # We can try to find if any vector is a linear combination of others
    
    # This is a complex search, so we'll do a quick check
    # by looking for duplicate or zero vectors
    
    # Check for zero columns
    zero_cols = []
    for r in range(56):
        if np.all(U_np[:, r] == 0) or np.all(V_np[:, r] == 0) or np.all(W_np[:, r] == 0):
            zero_cols.append(r)
    
    if zero_cols:
        # Remove zero columns
        keep = [r for r in range(56) if r not in zero_cols]
        new_rank = len(keep)
        if new_rank < 56:
            U_new = U_np[:, keep].tolist()
            V_new = V_np[:, keep].tolist()
            W_new = W_np[:, keep].tolist()
            return {"U": U_new, "V": V_new, "W": W_new}
    
    # Check for duplicate columns
    seen = {}
    dup_cols = []
    for r in range(56):
        key = (tuple(U_np[:, r]), tuple(V_np[:, r]), tuple(W_np[:, r]))
        if key in seen:
            dup_cols.append(r)
        else:
            seen[key] = r
    
    if dup_cols:
        keep = [r for r in range(56) if r not in dup_cols]
        new_rank = len(keep)
        if new_rank < 56:
            U_new = U_np[:, keep].tolist()
            V_new = V_np[:, keep].tolist()
            W_new = W_np[:, keep].tolist()
            return {"U": U_new, "V": V_new, "W": W_new}
    
    # Try a more sophisticated approach: use the structure of the problem
    # The 2x4 * 4x8 multiplication can be viewed as two 2x4 * 4x4 multiplications
    # or other decompositions. Let's try different block structures.
    
    # Alternative: treat as 2x4 * 4x8 = (2x4 * 4x4) for first 4 cols + (2x4 * 4x4) for last 4 cols
    # Each 2x4 * 4x4 can use rank-? Let's compute.
    # 2x4 * 4x4: n=2, m=4, p=4
    # Schoolbook: 2*4*4 = 32
    # Strassen-like: split 4 into 2x2 blocks
    # A: 2x4 = [A1 A2] where A1, A2 are 2x2
    # B: 4x4 = [[B11, B12], [B21, B22]] where each is 2x2
    # C: 2x4 = [C1 C2] where C1, C2 are 2x2
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    # This is 4 matrix multiplications of 2x2 * 2x2, each rank 7, total 28
    # So 2x4 * 4x4 has rank at most 28
    # Two such blocks: 56, same as before
    
    # Try another decomposition: 2x4 * 4x8
    # Split A into rows: A = [A1; A2] where A1, A2 are 1x4
    # C = A1*B1 + A2*B2 where B1, B2 are 4x8
    # This doesn't help directly
    
    # Try: split the inner dimension differently
    # A: 2x4, B: 4x8
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # Group j into pairs: (0,1) and (2,3)
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # = (A[i,0]+A[i,1])*B[0,k] + (A[i,0]-A[i,1])*(B[0,k]-B[1,k]) + A[i,1]*B[1,k] + ...
    # This is getting complicated. Let's stick with the current rank-56 solution.
    
    # Final check: return the rank-56 decomposition
    return {"U": U, "V": V, "W": W}
