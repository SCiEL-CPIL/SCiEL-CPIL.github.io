"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import sys
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    
    def make_schoolbook():
        U = [[0] * rank_school for _ in range(n * m)]
        V = [[0] * rank_school for _ in range(m * p)]
        W = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, rank_school
    
    # Verify a candidate decomposition
    def verify(U, V, W, rank):
        """Check if U, V, W correctly decompose the multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # Compute T[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        # We can do this by iterating over c or using einsum
        # T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
        # But einsum might be slow for small arrays? No, it's C-optimized.
        
        # Let's try einsum
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U_reshaped, V_reshaped, W_reshaped)
            if not np.allclose(T, 1.0, atol=1e-9):
                return False
            return True
        except Exception:
            # Fallback to loop if einsum fails (unlikely)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                        if abs(s - 1.0) > 1e-9:
                            return False
            return True
    
    # Strassen 2x2x2 decomposition
    # For multiplying 2x2 by 2x2, Strassen uses 7 multiplications
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our problem with n=2, we can use Strassen on 2x2 blocks of (j,k)
    # Split j into pairs: (0,1), (2,3), and single j=4
    # Split k into pairs: (0,1), (2,3), (4,5)
    
    def add_strassen_block(U, V, W, col_offset, j0, k0):
        """Add a Strassen 2x2x2 block for j indices (j0, j0+1) and k indices (k0, k0+1)"""
        # 7 Strassen terms
        # Term 1: (A[i,j0]+A[i,j0+1]) * (B[j0,k0]+B[j0+1,k0+1]) -> C[i,k0] += , C[i,k0+1] -= ... wait
        # Let me be more careful.
        
        # A is 2x5, B is 5x6, C is 2x6
        # For this block, we use A[:, j0:j0+2] (2x2), B[j0:j0+2, k0:k0+2] (2x2)
        # Result contributes to C[:, k0:k0+2] (2x2)
        
        # Strassen for 2x2*2x2:
        # Let A_sub = A[:, j0:j0+2], B_sub = B[j0:j0+2, k0:k0+2]
        # A_sub has entries A[0,j0], A[0,j0+1], A[1,j0], A[1,j0+1]
        # B_sub has entries B[j0,k0], B[j0,k0+1], B[j0+1,k0], B[j0+1,k0+1]
        
        # M1 = (A00+A11)(B00+B11) where A00=A[0,j0], A11=A[1,j0+1], etc.
        # But we need to track which (i,j) and (j,k) pairs contribute.
        
        # For each Strassen multiplication, we create one rank-1 term
        # The U vector has non-zero entries at specific (i,j) positions
        # The V vector has non-zero entries at specific (j,k) positions
        # The W vector has non-zero entries at specific (k,i) positions
        
        # M1: (A[0,j0] + A[1,j0+1]) * (B[j0,k0] + B[j0+1,k0+1])
        # U: U[0*m+j0] += 1, U[1*m+j0+1] += 1
        # V: V[j0*p+k0] += 1, V[(j0+1)*p+(k0+1)] += 1
        # W: W[k0*n+0] += 1, W[(k0+1)*n+1] += 1
        # But wait, this doesn't work directly because the Strassen formula combines results.
        
        # Actually, for the tensor decomposition, each Strassen "multiplication" corresponds to
        # one rank-1 term where U, V, W are vectors that when multiplied give the correct
        # contribution to the tensor.
        
        # Let me think differently. The Strassen algorithm computes:
        # C = A*B where A is 2x2, B is 2x2
        # Using 7 multiplications M1..M7
        # Each Mk is a product of a linear combination of A entries and a linear combination of B entries
        # Then C entries are linear combinations of M1..M7
        
        # For the tensor decomposition, we need:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all valid (i,j,k)
        
        # In Strassen, the "multiplication" Mk = (sum_a u_a * A[a]) * (sum_b v_b * B[b])
        # contributes to C entries via: C[c,d] += sum_mk w_mk * Mk
        # where w_mk are coefficients from the Strassen reconstruction
        
        # So for each Strassen multiplication Mk, we create a rank-1 term where:
        # U[:, c] has the coefficients for A entries
        # V[:, c] has the coefficients for B entries  
        # W[:, c] has the coefficients for C entries (the reconstruction weights)
        
        # For M1 = (A00+A11)(B00+B11):
        # U: A00=1, A11=1 -> U[0,j0]=1, U[1,j0+1]=1 (for i=0,j=j0 and i=1,j=j0+1)
        # Wait, A00 means A[0,0] in the submatrix, which is A[0,j0] in the full matrix
        # A11 means A[1,1] in the submatrix, which is A[1,j0+1] in the full matrix
        # B00 means B[0,0] in the submatrix, which is B[j0,k0] in the full matrix
        # B11 means B[1,1] in the submatrix, which is B[j0+1,k0+1] in the full matrix
        
        # So for M1:
        # U[0*m+j0] = 1, U[1*m+j0+1] = 1
        # V[j0*p+k0] = 1, V[(j0+1)*p+(k0+1)] = 1
        # W: C00 += M1, C11 += M1 (from C00 = M1+M4-M5+M7, C11 = M1-M2+M3+M6)
        # So W[k0*n+0] = 1, W[(k0+1)*n+1] = 1
        
        # Let me define the 7 Strassen terms properly:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Where in the submatrix:
        # A00 = A[0,0] -> full: A[0, j0]
        # A01 = A[0,1] -> full: A[0, j0+1]
        # A10 = A[1,0] -> full: A[1, j0]
        # A11 = A[1,1] -> full: A[1, j0+1]
        # B00 = B[0,0] -> full: B[j0, k0]
        # B01 = B[0,1] -> full: B[j0, k0+1]
        # B10 = B[1,0] -> full: B[j0+1, k0]
        # B11 = B[1,1] -> full: B[j0+1, k0+1]
        # C00 = C[0,0] -> full: C[0, k0]
        # C01 = C[0,1] -> full: C[0, k0+1]
        # C10 = C[1,0] -> full: C[1, k0]
        # C11 = C[1,1] -> full: C[1, k0+1]
        
        terms = []
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00+=1, C11+=1
        terms.append({
            'U': {0*m+j0: 1, 1*m+j0+1: 1},
            'V': {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
            'W': {k0*n+0: 1, (k0+1)*n+1: 1}
        })
        
        # M2 = (A10+A11)B00
        # U: A10=1, A11=1
        # V: B00=1
        # W: C10+=1, C11-=1
        terms.append({
            'U': {1*m+j0: 1, 1*m+j0+1: 1},
            'V': {j0*p+k0: 1},
            'W': {k0*n+1: 1, (k0+1)*n+1: -1}
        })
        
        # M3 = A00(B01-B11)
        # U: A00=1
        # V: B01=1, B11=-1
        # W: C01+=1, C11+=1
        terms.append({
            'U': {0*m+j0: 1},
            'V': {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1},
            'W': {(k0+1)*n+0: 1, (k0+1)*n+1: 1}
        })
        
        # M4 = A11(B10-B00)
        # U: A11=1
        # V: B10=1, B00=-1
        # W: C00+=1, C10+=1
        terms.append({
            'U': {1*m+j0+1: 1},
            'V': {(j0+1)*p+k0: 1, j0*p+k0: -1},
            'W': {k0*n+0: 1, k0*n+1: 1}
        })
        
        # M5 = (A00+A01)B11
        # U: A00=1, A01=1
        # V: B11=1
        # W: C00-=1, C01+=1
        terms.append({
            'U': {0*m+j0: 1, 0*m+j0+1: 1},
            'V': {(j0+1)*p+(k0+1): 1},
            'W': {k0*n+0: -1, (k0+1)*n+0: 1}
        })
        
        # M6 = (A10-A00)(B00+B01)
        # U: A10=1, A00=-1
        # V: B00=1, B01=1
        # W: C11+=1
        terms.append({
            'U': {1*m+j0: 1, 0*m+j0: -1},
            'V': {j0*p+k0: 1, j0*p+(k0+1): 1},
            'W': {(k0+1)*n+1: 1}
        })
        
        # M7 = (A01-A11)(B10+B11)
        # U: A01=1, A11=-1
        # V: B10=1, B11=1
        # W: C00+=1
        terms.append({
            'U': {0*m+j0+1: 1, 1*m+j0+1: -1},
            'V': {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
            'W': {k0*n+0: 1}
        })
        
        for term in terms:
            for idx, val in term['U'].items():
                U[idx][col_offset] = val
            for idx, val in term['V'].items():
                V[idx][col_offset] = val
            for idx, val in term['W'].items():
                W[idx][col_offset] = val
            col_offset += 1
        
        return col_offset
    
    def add_schoolbook_block(U, V, W, col_offset, j_indices, k_indices, i_indices=None):
        """Add schoolbook terms for given j and k indices"""
        if i_indices is None:
            i_indices = range(n)
        
        for i in i_indices:
            for j in j_indices:
                for k in k_indices:
                    U[i*m+j][col_offset] = 1
                    V[j*p+k][col_offset] = 1
                    W[k*n+i][col_offset] = 1
                    col_offset += 1
        return col_offset
    
    # Build the Strassen-based decomposition
    # j pairs: (0,1), (2,3), and single j=4
    # k pairs: (0,1), (2,3), (4,5)
    
    U = [[0] * 0 for _ in range(n * m)]
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    # First, determine total rank
    # 6 Strassen blocks (2x2 j) x (3 k pairs) = 6 blocks * 7 = 42
    # 3 schoolbook blocks (j=4) x (3 k pairs) x 2 i = 3*3*2 = 18? No...
    # For j=4, k-pair (0,1): 2 i * 1 j * 2 k = 4 terms
    # For j=4, k-pair (2,3): 4 terms
    # For j=4, k-pair (4,5): 4 terms
    # Total: 42 + 12 = 54
    
    total_rank = 54
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Add Strassen blocks for j-pairs (0,1) and (2,3) with each k-pair
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            col = add_strassen_block(U, V, W, col, j0, k0)
    
    # Add schoolbook for j=4 with each k-pair
    for k0 in [0, 2, 4]:
        col = add_schoolbook_block(U, V, W, col, [4], [k0, k0+1])
    
    # Verify
    if not verify(U, V, W, total_rank):
        # Fallback to schoolbook
        U, V, W, rank_school = make_schoolbook()
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
