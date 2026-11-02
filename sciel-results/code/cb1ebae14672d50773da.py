"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # We aim for a rank-19 decomposition:
    # - 7 terms for the first two columns (k=0,1) using a Strassen-like structure
    # - 4 terms per remaining column (k=2,3,4) using trivial rank-4 decomposition
    # Total rank = 7 + 3*4 = 19
    
    rank = 19
    U = [[0] * rank for _ in range(n * m)]  # 4 x 19
    V = [[0] * rank for _ in range(m * p)]  # 10 x 19
    W = [[0] * rank for _ in range(p * n)]  # 10 x 19
    
    # Strassen-like decomposition for 2x2x2 (first two columns k=0,1)
    # Standard Strassen for 2x2 matrix multiplication uses 7 multiplications
    # We need S[i,j,k] = delta_{i,j,k} for k=0,1
    # The tensor for 2x2x2 multiplication has entries T[i,j,k] = 1 if the (i,j) entry of A times the j-th column of B contributes to output
    # Actually, for matrix multiplication C = A*B where A is 2x2, B is 2x2:
    # C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor T[i,k,j] = 1 for all i,k,j (each term A[i,k]*B[k,j] contributes)
    # So for 2x2x2, the tensor has 8 non-zero entries (all ones in a 2x2x2 box)
    # Strassen decomposes this into 7 rank-1 tensors
    
    # Let's use the known Strassen decomposition for 2x2x2
    # Indices: i in {0,1} (row of A), k in {0,1} (col of A / row of B), j in {0,1} (col of B)
    # T[i,k,j] = 1 for all i,k,j in {0,1}
    
    # Strassen's 7 factors:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # For our tensor T[i,k,j] where i=row of A, k=col of A/row of B, j=col of B:
    # We need to express T[i,k,j] = sum_r U[i,k,r] * V[k,j,r] * W[j,i,r]
    # Wait, the convention is W[k*n+i, r] where k is column of B and i is row of A
    # So W index is (j, i) where j is col of B, i is row of A
    
    # Let me re-examine: U has shape (n*m, rank) = (4, rank), indexed by (i*m+j) where i is row of A, j is col of A
    # V has shape (m*p, rank) = (10, rank), indexed by (j*p+k) where j is col of A, k is col of B  
    # Wait, m=2, p=5, so V is 2*5=10 entries
    # V[j*p + k] where j is column of A (0 or 1), k is column of B (0..4)
    # W has shape (p*n, rank) = (10, rank), indexed by W[k*n + i] where k is col of B, i is row of A
    
    # So the decomposition is: S[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # where i is row of A (0..n-1), j is col of A / row of B (0..m-1), k is col of B (0..p-1)
    # And we want S[i,j,k] = 1 for all valid i,j,k (since the multiplication tensor has all 1s)
    
    # For k=0,1 (first two columns), we use Strassen. For k=2,3,4, we use trivial.
    
    # Trivial decomposition for a single column k:
    # We need S[i,j,k] = 1 for i in {0,1}, j in {0,1}
    # Using 4 terms:
    # r1: U[0*m+0]=1, V[0*p+k]=1, W[k*n+0]=1 -> contributes to S[0,0,k]
    # r2: U[0*m+1]=1, V[1*p+k]=1, W[k*n+0]=1 -> contributes to S[0,1,k]
    # r3: U[1*m+0]=1, V[0*p+k]=1, W[k*n+1]=1 -> contributes to S[1,0,k]
    # r4: U[1*m+1]=1, V[1*p+k]=1, W[k*n+1]=1 -> contributes to S[1,1,k]
    
    # For k=2,3,4, use trivial with 4 terms each
    col = 0
    for k in range(2, 5):
        # r1
        U[0*m + 0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        # r2
        U[0*m + 1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        # r3
        U[1*m + 0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        # r4
        U[1*m + 1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
    
    # Now for k=0,1, use Strassen (7 terms)
    # We need S[i,j,0] = 1 and S[i,j,1] = 1 for i,j in {0,1}
    # This is equivalent to decomposing two 2x2 all-ones matrices
    # But we can share the Strassen structure
    
    # Let's use a combined approach. For the 2x2x2 subproblem (k=0,1),
    # the tensor T[i,j,k] for i,j in {0,1}, k in {0,1} has all 8 entries = 1
    # We need to decompose this into 7 rank-1 tensors
    
    # Strassen's decomposition for the 2x2x2 multiplication tensor:
    # The 7 rank-1 components are:
    # (Using standard Strassen, adapted to our index convention)
    
    # Let me define the 7 factors explicitly:
    # Each factor r has U[i,j,r], V[j,k,r], W[k,i,r]
    
    # Factor 1: (i=0,j=0,k=0) + (i=0,j=1,k=1) + (i=1,j=0,k=0) + (i=1,j=1,k=1)
    # U: (0,0)=1, (0,1)=1, (1,0)=1, (1,1)=1 -> all ones
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    U[0*m+0][col] = 1
    U[0*m+1][col] = 1
    U[1*m+0][col] = 1
    U[1*m+1][col] = 1
    V[0*p+0][col] = 1
    V[1*p+1][col] = 1
    W[0*n+0][col] = 1
    W[1*n+1][col] = 1
    col += 1
    
    # Factor 2: (i=1,j=0,k=0) + (i=1,j=1,k=0)
    # U: (1,0)=1, (1,1)=1
    # V: (0,0)=1
    # W: (0,1)=1
    U[1*m+0][col] = 1
    U[1*m+1][col] = 1
    V[0*p+0][col] = 1
    W[0*n+1][col] = 1
    col += 1
    
    # Factor 3: (i=0,j=0,k=1) - (i=0,j=0,k=0)
    # U: (0,0)=1
    # V: (0,1)=1, (0,0)=-1
    # W: (1,0)=1
    U[0*m+0][col] = 1
    V[0*p+1][col] = 1
    V[0*p+0][col] = -1
    W[1*n+0][col] = 1
    col += 1
    
    # Factor 4: (i=1,j=1,k=0) - (i=1,j=1,k=1)
    # U: (1,1)=1
    # V: (1,0)=1, (1,1)=-1
    # W: (0,1)=1
    U[1*m+1][col] = 1
    V[1*p+0][col] = 1
    V[1*p+1][col] = -1
    W[0*n+1][col] = 1
    col += 1
    
    # Factor 5: (i=0,j=0,k=0) + (i=0,j=1,k=1)
    # U: (0,0)=1, (0,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1
    U[0*m+0][col] = 1
    U[0*m+1][col] = 1
    V[0*p+0][col] = 1
    V[1*p+1][col] = 1
    W[0*n+0][col] = 1
    col += 1
    
    # Factor 6: (i=1,j=0,k=0) - (i=0,j=0,k=0) + (i=1,j=0,k=1)
    # U: (1,0)=1, (0,0)=-1
    # V: (0,0)=1
    # W: (0,1)=1, (1,1)=1
    U[1*m+0][col] = 1
    U[0*m+0][col] = -1
    V[0*p+0][col] = 1
    W[0*n+1][col] = 1
    W[1*n+1][col] = 1
    col += 1
    
    # Factor 7: (i=0,j=1,k=1) - (i=1,j=1,k=1) + (i=0,j=1,k=0)
    # U: (0,1)=1, (1,1)=-1
    # V: (1,1)=1
    # W: (1,0)=1, (0,0)=1
    U[0*m+1][col] = 1
    U[1*m+1][col] = -1
    V[1*p+1][col] = 1
    W[1*n+0][col] = 1
    W[0*n+0][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

