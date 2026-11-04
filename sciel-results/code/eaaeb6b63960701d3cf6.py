"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24

    def build_schoolbook():
        """Build the schoolbook rank-120 decomposition."""
        rank = n * m * p  # 120
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return U, V, W, rank

    def strassen_2x2x2():
        """Standard Strassen 2x2x2 decomposition, rank 7.
        
        For multiplying A(2x2) * B(2x2) -> C(2x2):
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        In our tensor notation:
        U[i,j] corresponds to A[i,j] -> U[i*m+j, r]
        V[j,k] corresponds to B[j,k] -> V[j*p+k, r]  
        W[k,i] corresponds to C[i,k] -> W[k*n+i, r]
        
        For a local 2x2x2 block at offset (i0, j0, k0):
        U[(i0+i)*m + (j0+j), r]
        V[(j0+j)*p + (k0+k), r]
        W[(k0+k)*n + (i0+i), r]
        """
        # Standard Strassen formulas:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B21
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M6 + M7
        
        entries = []
        # Each entry: (i, j, k, weight) for local indices
        # M1: (A11+A22)(B11+B22) -> contributes to C11, C22
        entries.append((0, 0, 0, 1))   # A11*B11 -> C11
        entries.append((1, 1, 0, 1))   # A22*B11 -> C11
        entries.append((0, 0, 1, 1))   # A11*B22 -> C22
        entries.append((1, 1, 1, 1))   # A22*B22 -> C22
        
        # M2: (A21+A22)B21 -> contributes to C21
        entries.append((1, 1, 0, 1))   # A21*B21 -> C21
        entries.append((0, 1, 0, 1))   # A22*B21 -> C21
        
        # M3: A11(B12-B22) -> contributes to C12
        entries.append((0, 0, 1, 1))   # A11*B12 -> C12
        entries.append((0, 1, 1, -1))  # A11*(-B22) -> C12
        
        # M4: A22(B21-B11) -> contributes to C11, C21
        entries.append((1, 1, 0, 1))   # A22*B21 -> C11, C21
        entries.append((1, 0, 0, -1))  # A22*(-B11) -> C11, C21
        
        # M5: (A11+A12)B22 -> contributes to C11(-), C12(+)
        entries.append((0, 0, 1, 1))   # A11*B22 -> C11(-), C12(+)
        entries.append((0, 1, 1, 1))   # A12*B22 -> C11(-), C12(+)
        
        # M6: (A21-A11)(B11+B12) -> contributes to C22
        entries.append((1, 0, 0, 1))   # A21*B11 -> C22
        entries.append((1, 0, 1, 1))   # A21*B12 -> C22
        entries.append((0, 0, 0, -1))  # (-A11)*B11 -> C22
        entries.append((0, 0, 1, -1))  # (-A11)*B12 -> C22
        
        # M7: (A12-A22)(B21+B22) -> contributes to C11, C22
        entries.append((0, 1, 0, 1))   # A12*B21 -> C11, C22
        entries.append((0, 1, 1, 1))   # A12*B22 -> C11, C22
        entries.append((1, 1, 0, -1))  # (-A22)*B21 -> C11, C22
        entries.append((1, 1, 1, -1))  # (-A22)*B22 -> C11, C22
        
        # Now we need to group these into 7 rank-1 terms.
        # The standard Strassen grouping:
        # M1: A11*B11, A11*B22, A22*B11, A22*B22 (4 terms in one product)
        # But each rank-1 term is a single (i,j,k) triple with a weight.
        # Actually, each rank-1 term in the tensor decomposition corresponds to
        # a single (i,j,k) with a scalar weight.
        #
        # For Strassen, we have 7 "multiply" operations, but each multiply
        # involves products of linear combinations. The tensor rank-1 terms
        # are:
        
        # Let me use the explicit rank-7 decomposition for 2x2x2:
        # The 2x2x2 matrix multiplication tensor has rank 7 (Strassen).
        # The explicit decomposition:
        
        # Term 1: (A11+A22)(B11+B22)
        # This gives 4 contributions to C:
        #   A11*B11 -> C11
        #   A11*B22 -> C12  
        #   A22*B11 -> C21
        #   A22*B22 -> C22
        # But in tensor form, this is NOT a rank-1 term because it involves
        # multiple (i,j,k) triples.
        
        # I need to think about this differently. The tensor T_{i,j,k} for 2x2x2
        # has T_{i,j,k} = 1 for all i,j,k in {0,1}. We need to decompose this
        # as sum_r u_r ⊗ v_r ⊗ w_r where u_r[i]*v_r[j]*w_r[k] summed over r = T_{i,j,k}.
        
        # Actually, the matrix multiplication tensor T has entries:
        # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 for all valid (i,j,k), and the decomposition is:
        # sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k] = 1
        
        # For 2x2x2, T[i,j,k] = 1 for all i,j,k in {0,1} (8 entries).
        # Schoolbook: 8 rank-1 terms, each with U[i,r]=1, V[j,r]=1, W[k,r]=1
        # for specific (i,j,k).
        
        # Strassen rank-7: We need 7 vectors u_r, v_r, w_r in R^2 such that
        # sum_r u_r[i]*v_r[j]*w_r[k] = 1 for all i,j,k.
        
        # One explicit Strassen decomposition:
        # Let me use the standard one. The key insight is that we can write:
        # 1 = sum of products of linear combinations.
        
        # Standard Strassen for 2x2x2 tensor:
        # r=0: u=(1,1), v=(1,1), w=(1,0)  -> contributes 1 to T[0,0,0], T[0,1,0], T[1,0,0], T[1,1,0]
        #       Actually u[0]*v[0]*w[0] = 1*1*1 = 1 for (0,0,0)
        #       u[0]*v[1]*w[0] = 1*1*1 = 1 for (0,1,0)
        #       u[1]*v[0]*w[0] = 1*1*1 = 1 for (1,0,0)
        #       u[1]*v[1]*w[0] = 1*1*1 = 1 for (1,1,0)
        #       This alone covers 4 of the 8 entries with value 1, but we need
        #       the sum over all r to equal 1, not 4.
        
        # I think I'm overcomplicating this. Let me just use the schoolbook
        # decomposition for all blocks. The rank will be 120, which is the
        # baseline. Without explicit verified Strassen formulas, I'll stick
        # with schoolbook to ensure correctness.
        
        return None

    def verify(U, V, W, rank):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    # Build schoolbook decomposition
    U, V, W, rank = build_schoolbook()
    
    # Verify
    if not verify(U, V, W, rank):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    return best_result
# EVOLVE-BLOCK-END
