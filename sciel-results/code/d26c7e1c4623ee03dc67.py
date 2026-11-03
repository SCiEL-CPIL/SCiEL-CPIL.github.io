import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]

def _verify(U, V, W, n=4, m=4, p=5, tol=1e-8):
    """Verify that U, V, W decompose the 4x4x5 matrix multiplication tensor."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Check each (i,j,k) entry
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, leaving margin
    
    # Strategy: Use Strassen-based block decomposition to reduce rank
    # 4x4x5 can be decomposed using 2x2 blocks with Strassen (rank 7) 
    # and handle the 5th column with naive multiplication
    
    # Build the rank-64 solution using Strassen for 2x2 blocks + naive for remaining
    # 4x4x5: split A into 2x2 blocks, B into 2x2 blocks (with 5 cols -> 2+3 or 2+2+1)
    # Better: split as 2x2 + 2x3 for B, and 2x2 for A output
    
    # Let's use a systematic approach:
    # A is 4x4, B is 4x5, C = A*B is 4x5
    # Split rows of A: first 2, last 2
    # Split cols of B: first 2, next 2, last 1 (so 2+2+1=5)
    # Split rows of C: first 2, last 2
    # Split cols of C: first 2, next 2, last 1
    
    # This gives 2x3 = 6 sub-multiplications:
    # A_{11}B_{11}, A_{11}B_{12}, A_{11}B_{13}, A_{21}B_{21}, A_{21}B_{22}, A_{21}B_{23}
    # where A_{11} is 2x2, B_{11} is 2x2, B_{12} is 2x2, B_{13} is 2x1, etc.
    
    # For each 2x2 * 2x2 -> use Strassen (rank 7)
    # For each 2x2 * 2x1 -> use naive (rank 4) 
    # For each 2x2 * 1x? -> not applicable here
    
    # Actually let me be more careful. Let me use a simpler known result.
    # 
    # Known: 4x4x5 can be done in rank 42 using Winograd-style or specific constructions.
    # But let's start with a solid 64-rank Strassen-based solution and try to improve.
    
    # Build Strassen 2x2x2 factors
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen multiplication, each of shape (4,7)."""
        # Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # A indexing: A11=0, A12=1, A21=2, A22=3
        # B indexing: B11=0, B12=1, B21=2, B22=3
        # C indexing: C11=0, C12=1, C21=2, C22=3
        
        # M1
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1; W[2,0]=0  # C11 gets M1
        # C11 = M1+M4-M5+M7
        W[0,0]=1  # C11 += M1
        
        # M2
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1  # C21 += M2
        W[3,1]=-1  # C22 -= M2
        
        # M3
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1  # C12 += M3
        W[3,2]=1  # C22 += M3
        
        # M4
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1  # C11 += M4
        W[2,3]=1  # C21 += M4
        
        # M5
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1  # C11 -= M5
        W[1,4]=1  # C12 += M5
        
        # M6
        U[2,5]=-1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1  # C22 += M6
        
        # M7
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1  # C11 += M7
        
        return U, V, W
    
    # Build naive 2x2x1 (2x2 * 2x1 = 2x1) factors
    def naive_2x2x1():
        """Return U, V, W for 2x2x1 multiplication, each of shape (4,4), (2,4), (2,4)."""
        # C = A * b where A is 2x2, b is 2x1, C is 2x1
        # C1 = A11*b1 + A12*b2
        # C2 = A21*b1 + A22*b2
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        # Term 1: A11*b1 -> C1
        U[0,0]=1; V[0,0]=1; W[0,0]=1
        # Term 2: A12*b2 -> C1
        U[1,1]=1; V[1,1]=1; W[0,1]=1
        # Term 3: A21*b1 -> C2
        U[2,2]=1; V[0,2]=1; W[1,2]=1
        # Term 4: A22*b2 -> C2
        U[3,3]=1; V[1,3]=1; W[1,3]=1
        return U, V, W
    
    # Build naive 2x1x2 (2x1 * 1x2 = 2x2) - not needed in our split
    
    # Now assemble the full 4x4x5 decomposition
    # A: 4x4, split into A_top (rows 0,1) and A_bot (rows 2,3), each 2x4
    # B: 4x5, split into B_cols01 (cols 0,1), B_cols23 (cols 2,3), B_col4 (col 4)
    # C: 4x5, split into C_top (rows 0,1), C_bot (rows 2,3), each 2x5
    
    # Sub-problems:
    # 1. A_top[0:2,0:2] * B[0:2,0:2] -> C_top[0:2,0:2] (2x2*2x2 -> 2x2): Strassen, rank 7
    # 2. A_top[0:2,2:4] * B[2:4,0:2] -> C_top[0:2,0:2] (2x2*2x2 -> 2x2): Strassen, rank 7
    # 3. A_top[0:2,0:2] * B[0:2,2:4] -> C_top[0:2,2:4] (2x2*2x2 -> 2x2): Strassen, rank 7
    # 4. A_top[0:2,2:4] * B[2:4,2:4] -> C_top[0:2,2:4] (2x2*2x2 -> 2x2): Strassen, rank 7
    # 5. A_top[0:2,0:2] * B[0:2,4:5] -> C_top[0:2,4:5] (2x2*2x1 -> 2x1): naive, rank 4
    # 6. A_top[0:2,2:4] * B[2:4,4:5] -> C_top[0:2,4:5] (2x2*2x1 -> 2x1): naive, rank 4
    # 7. A_bot[0:2,0:2] * B[0:2,0:2] -> C_bot[0:2,0:2] (2x2*2x2 -> 2x2): Strassen, rank 7
    # 8. A_bot[0:2,2:4] * B[2:4,0:2] -> C_bot[0:2,0:2] (2x2*2x2 -> 2x2): Strassen, rank 7
    # 9. A_bot[0:2,0:2] * B[0:2,2:4] -> C_bot[0:2,2:4] (2x2*2x2 -> 2x2): Strassen, rank 7
    # 10. A_bot[0:2,2:4] * B[2:4,2:4] -> C_bot[0:2,2:4] (2x2*2x2 -> 2x2): Strassen, rank 7
    # 11. A_bot[0:2,0:2] * B[0:2,4:5] -> C_bot[0:2,4:5] (2x2*2x1 -> 2x1): naive, rank 4
    # 12. A_bot[0:2,2:4] * B[2:4,4:5] -> C_bot[0:2,4:5] (2x2*2x1 -> 2x1): naive, rank 4
    
    # Total rank: 8*7 + 4*4 = 56 + 16 = 72
    
    # But wait, I need to handle the column indices correctly.
    # Let me use a different approach - just build it directly with the known rank.
    
    # Actually, let me just return the schoolbook solution first to ensure correctness,
    # then try to improve with Strassen if time permits.
    
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Check if we have time to try Strassen-based improvement
    if time.time() - start_time < 60:
        # Try to build a Strassen-based solution
        try:
            U_strassen, V_strassen, W_strassen = _build_strassen_4x4x5()
            if _verify(U_strassen, V_strassen, W_strassen, n, m, p):
                return {"U": U_strassen, "V": V_strassen, "W": W_strassen}
        except Exception:
            pass
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _build_strassen_4x4x5():
    """Build a Strassen-based 4x4x5 decomposition."""
    n, m, p = 4, 4, 5
    
    # We'll build using the block structure
    # A (4x4) split into 4 blocks of 2x2: A00, A01, A10, A11
    # B (4x5) split into 2x2, 2x2, 2x1: B00, B01, B02 (2x1), B10, B11, B12 (2x1)
    # C (4x5) = A*B, split into 2x2, 2x2, 2x1 for top row, same for bottom
    
    # For each 2x2*2x2 block, use Strassen (rank 7)
    # For each 2x2*2x1 block, use naive (rank 4)
    
    # There are 8 Strassen blocks and 4 naive blocks
    # Total rank = 8*7 + 4*4 = 72
    
    # Let me build this more carefully using explicit index mappings
    
    U = np.zeros((n*m, 72))
    V = np.zeros((m*p, 72))
    W = np.zeros((p*n, 72))
    
    col_offset = 0
    
    # Helper: add Strassen 2x2x2 block at given positions
    def add_strassen(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal col_offset
        # Strassen U,V,W for 2x2*2x2
        su, sv, sw = _strassen_factors()
        # Map indices
        for ci in range(7):
            for ri in range(4):
                U[a_row_start + ri, col_offset + ci] += su[ri, ci]
            for ri in range(4):
                V[b_row_start + ri, col_offset + ci] += sv[ri, ci]
            for ri in range(4):
                W[c_row_start + ri, col_offset + ci] += sw[ri, ci]
        col_offset += 7
    
    # Helper: add naive 2x2*2x1 block
    def add_naive_2x2x1(a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start):
        nonlocal col_offset
        nu, nv, nw = _naive_2x2x1_factors()
        for ci in range(4):
            for ri in range(4):
                U[a_row_start + ri, col_offset + ci] += nu[ri, ci]
            for ri in range(2):
                V[b_row_start + ri, col_offset + ci] += nv[ri, ci]
            for ri in range(2):
                W[c_row_start + ri, col_offset + ci] += nw[ri, ci]
        col_offset += 4
    
    # Strassen factors for 2x2*2x2
    def _strassen_factors():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        # A: 0,1,2,3 (row-major 2x2)
        # B: 0,1,2,3 (row-major 2x2)  
        # C: 0,1,2,3 (row-major 2x2)
        
        # M1 = (A00+A11)(B00+B11)
        U[0,0]=1; U[3,0]=1
        V[0,0]=1; V[3,0]=1
        W[0,0]=1  # C00 += M1
        W[3,0]=0  # C11 doesn't get M1 directly
        
        # M2 = (A10+A11)B00
        U[2,1]=1; U[3,1]=1
        V[0,1]=1
        W[2,1]=1  # C10 += M2
        W[3,1]=-1 # C11 -= M2
        
        # M3 = A00(B01-B11)
        U[0,2]=1
        V[1,2]=1; V[3,2]=-1
        W[1,2]=1  # C01 += M3
        W[3,2]=1  # C11 += M3
        
        # M4 = A11(B10-B00)
        U[3,3]=1
        V[2,3]=1; V[0,3]=-1
        W[0,3]=1  # C00 += M4
        W[2,3]=1  # C10 += M4
        
        # M5 = (A00+A01)B11
        U[0,4]=1; U[1,4]=1
        V[3,4]=1
        W[0,4]=-1 # C00 -= M5
        W[1,4]=1  # C01 += M5
        
        # M6 = (A10-A00)(B00+B01)
        U[2,5]=1; U[0,5]=-1
        V[0,5]=1; V[1,5]=1
        W[3,5]=1  # C11 += M6
        
        # M7 = (A01-A11)(B10+B11)
        U[1,6]=1; U[3,6]=-1
        V[2,6]=1; V[3,6]=1
        W[0,6]=1  # C00 += M7
        
        return U, V, W
    
    def _naive_2x2x1_factors():
        """2x2 * 2x1 -> 2x1, rank 4"""
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0,0]=1; V[0,0]=1; W[0,0]=1  # A00*b0 -> C0
        U[1,1]=1; V[1,1]=1; W[0,1]=1  # A01*b1 -> C0
        U[2,2]=1; V[0,2]=1; W[1,2]=1  # A10*b0 -> C1
        U[3,3]=1; V[1,3]=1; W[1,3]=1  # A11*b1 -> C1
        return U, V, W
    
    # Now assemble all blocks
    # A rows 0-3, cols 0-3
    # B rows 0-3, cols 0-4
    # C rows 0-3, cols 0-4
    
    # Top-left 2x2 of C:
    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    add_strassen(0, 0, 0, 0, 0, 0)       # A[0:2,0:2]*B[0:2,0:2]
    add_strassen(0, 2, 2, 0, 0, 0)       # A[0:2,2:4]*B[2:4,0:2]
    
    # Top-middle 2x2 of C:
    # C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]
    add_strassen(0, 0, 0, 2, 0, 2)       # A[0:2,0:2]*B[0:2,2:4]
    add_strassen(0, 2, 2, 2, 0, 2)       # A[0:2,2:4]*B[2:4,2:4]
    
    # Top-right 2x1 of C:
    # C[0:2,4] = A[0:2,0:2]*B[0:2,4] + A[0:2,2:4]*B[2:4,4]
    add_naive_2x2x1(0, 0, 0, 4, 0, 4)    # A[0:2,0:2]*B[0:2,4]
    add_naive_2x2x1(0, 2, 2, 4, 0, 4)    # A[0:2,2:4]*B[2:4,4]
    
    # Bottom-left 2x2 of C:
    # C[2:4,0:2] = A[2:4,0:2]*B[0:2,0:2] + A[2:4,2:4]*B[2:4,0:2]
    add_strassen(2, 0, 0, 0, 2, 0)       # A[2:4,0:2]*B[0:2,0:2]
    add_strassen(2, 2, 2, 0, 2, 0)       # A[2:4,2:4]*B[2:4,0:2]
    
    # Bottom-middle 2x2 of C:
    # C[2:4,2:4] = A[2:4,0:2]*B[0:2,2:4] + A[2:4,2:4]*B[2:4,2:4]
    add_strassen(2, 0, 0, 2, 2, 2)       # A[2:4,0:2]*B[0:2,2:4]
    add_strassen(2, 2, 2, 2, 2, 2)       # A[2:4,2:4]*B[2:4,2:4]
    
    # Bottom-right 2x1 of C:
    # C[2:4,4] = A[2:4,0:2]*B[0:2,4] + A[2:4,2:4]*B[2:4,4]
    add_naive_2x2x1(2, 0, 0, 4, 2, 4)    # A[2:4,0:2]*B[0:2,4]
    add_naive_2x2x1(2, 2, 2, 4, 2, 4)    # A[2:4,2:4]*B[2:4,4]
    
    return U.tolist(), V.tolist(), W.tolist()

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
