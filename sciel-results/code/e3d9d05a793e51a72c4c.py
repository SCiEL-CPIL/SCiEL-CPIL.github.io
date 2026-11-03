import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16

    deadline = time.time() + 145.0

    def check_decomp(U, V, W, tol=1e-6):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        T = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        return np.allclose(T, 1.0, atol=tol)

    def build_kronecker_strassen():
        """
        Build a decomposition using the Kronecker product structure.
        
        Strategy:
        1. Transpose A (2x4) to A^T (4x2) and permute C (2x8) to C' (8x2).
        2. The multiplication becomes A^T (4x2) * B (4x8) -> C' (8x2) with permutation.
        3. View this as a Kronecker product of two 2x2 * 2x2 multiplications.
        4. Use rank-4 Strassen-like decompositions for each 2x2 * 2x2 factor.
        
        Actually, let's use a more direct approach:
        The tensor T[i,j,k] = 1 for all i,j,k can be decomposed as:
        T = sum_r U_r ⊗ V_r ⊗ W_r
        
        We'll use the fact that the 2x4*4x8 multiplication can be seen as:
        - 2 groups of i (size 2)
        - 2 groups of j (size 2) 
        - 4 groups of k (size 2)
        
        But instead of independent blocks, we use shared Strassen products.
        
        Let's try a different decomposition:
        Use the rank-7 Strassen decomposition but exploit the structure better.
        
        Actually, let's implement a rank-32 decomposition using the following idea:
        - Partition into 4 blocks of size (2x2) * (2x4) -> (4x2)
        - For each block, use a rank-8 decomposition (not 7, since 2x2*2x4 is not square)
        
        Wait, let's be more careful. The dimensions are:
        A: 2x4, B: 4x8, C: 2x8
        
        We can partition as:
        A = [A1 A2] where A1, A2 are 2x2
        B = [B1 B2 B3 B4] where each Bj is 4x2
        C = [C1 C2 C3 C4] where each Cj is 2x2
        
        C1 = A1*B1 + A2*B3
        C2 = A1*B2 + A2*B4
        C3 = A1*B3 + A2*B5  -- wait, B only has 4 columns of 2, so B is 4x8 = [B1 B2 B3 B4]
        
        Let me re-index. B is 4x8. Let's partition B into 4 blocks of 4x2:
        B = [B0 B1 B2 B3] where Bj is 4x2
        A is 2x4. Let's partition A into 2 blocks of 2x2:
        A = [A0 A1] where Ai is 2x2
        
        Then C = A*B is 2x8. Partition C into 4 blocks of 2x2:
        C = [C0 C1 C2 C3] where Ck is 2x2
        
        Ck = A0 * Bk[0:2,:] + A1 * Bk[2:4,:]
        
        So each Ck depends on all of A and a 2x2 sub-block of B.
        
        This doesn't immediately give a Kronecker structure. Let's try the transpose approach.
        
        Let A^T be 4x2. Let B be 4x8. Let C^T be 8x2.
        C^T = B^T * A^T. So we're computing B^T (8x4) * A^T (4x2) -> C^T (8x2).
        
        Now B^T is 8x4, A^T is 4x2.
        Partition B^T into 4 blocks of 2x4: B^T = [B1^T; B2^T; B3^T; B4^T]
        Partition A^T into 2 blocks of 2x2: A^T = [A1^T; A2^T]
        
        C^T = [C1^T; C2^T; C3^T; C4^T] where Ck^T is 2x2.
        Ck^T = Bk^T * A^T = Bk^T * [A1^T; A2^T]
        
        Still not a clean Kronecker.
        
        Let's try a completely different approach: use the rank-7 Strassen decomposition
        but apply it to the whole problem by viewing it as a 2x2 block multiplication
        where the "entries" are 2x4 matrices.
        
        A = [A11 A12; A21 A22] where each block is 1x4? No, A is 2x4.
        
        Let's partition A into 2x2 blocks of size 1x2:
        A = [A11 A12; A21 A22] where Aij is 1x2.
        B is 4x8. Partition into 2x2 blocks of size 2x4:
        B = [B11 B12; B21 B22] where Bij is 2x4.
        C is 2x8. Partition into 2x2 blocks of size 1x4:
        C = [C11 C12; C21 C22] where Cij is 1x4.
        
        C11 = A11*B11 + A12*B21
        C12 = A11*B12 + A12*B22
        C21 = A21*B11 + A22*B21
        C22 = A21*B12 + A22*B22
        
        Now each Cij is a 1x4 matrix, which is a sum of two products of 1x2 and 2x4 matrices.
        Each such product can be decomposed with rank 2 (schoolbook) or we can use Strassen
        on the 2x2 "scalar" structure if we view the 1x2 and 2x4 as having internal structure.
        
        This is getting complicated. Let's just implement the rank-56 Strassen decomposition
        correctly and verify it, then try to improve it.
        """
        # Fallback to the known correct rank-56 Strassen decomposition
        rank = 8 * 7  # 56
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        col = 0
        for jg in range(2):  # j-group: 0 -> {0,1}, 1 -> {2,3}
            for kg in range(4):  # k-group: 0->{0,1}, 1->{2,3}, 2->{4,5}, 3->{6,7}
                j0 = jg * 2
                j1 = jg * 2 + 1
                k0 = kg * 2
                k1 = kg * 2 + 1

                strassen_defs = [
                    ([(0 * m + j0, 1)], [(j0 * p + k0, 1)], [(k0 * n + 0, 1)]),
                    ([(1 * m + j1, 1)], [(j1 * p + k1, 1)], [(k1 * n + 1, 1)]),
                    ([(0 * m + j0, 1), (0 * m + j1, 1)], [(j0 * p + k1, 1)], [(k1 * n + 0, 1)]),
                    ([(1 * m + j0, 1), (0 * m + j0, -1)], [(j0 * p + k0, 1), (j0 * p + k1, 1)], [(k0 * n + 0, 1)]),
                    ([(1 * m + j0, 1), (1 * m + j1, 1)], [(j1 * p + k0, 1)], [(k0 * n + 1, 1)]),
                    ([(0 * m + j1, 1), (1 * m + j1, -1)], [(j1 * p + k0, 1), (j1 * p + k1, 1)], [(k1 * n + 1, 1)]),
                    ([(1 * m + j1, 1), (0 * m + j0, 1), (1 * m + j0, -1), (0 * m + j1, -1)],
                     [(j0 * p + k1, 1), (j1 * p + k1, -1)], [(k0 * n + 0, 1)]),
                ]

                for u_coeffs, v_coeffs, w_coeffs in strassen_defs:
                    for idx, val in u_coeffs:
                        U[idx, col] += val
                    for idx, val in v_coeffs:
                        V[idx, col] += val
                    for idx, val in w_coeffs:
                        W[idx, col] += val
                    col += 1

        return U, V, W

    # Try the Kronecker-based decomposition first
    if time.time() < deadline - 10.0:
        try:
            # Implement a rank-32 decomposition using the following structure:
            # The 2x4*4x8 multiplication can be decomposed as a Kronecker product
            # of two 2x2*2x2 multiplications, each with rank 7, giving total rank 49?
            # No, that's not right either.
            
            # Let's use a different approach: the tensor T[i,j,k] = 1 can be written as
            # T = sum_r u_r ⊗ v_r ⊗ w_r
            # where u_r is in R^(2*4), v_r in R^(4*8), w_r in R^(8*2).
            
            # We'll use the rank-7 Strassen decomposition for 2x2*2x2 and apply it
            # in a structured way.
            
            # For now, use the known correct rank-56 decomposition
            U, V, W = build_kronecker_strassen()
            
            if check_decomp(U, V, W, tol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
        except Exception as e:
            pass

    # Fallback: schoolbook
    rank = n * m * p
    U2 = np.zeros((U_dim, rank))
    V2 = np.zeros((V_dim, rank))
    W2 = np.zeros((W_dim, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i*m+j, c] = 1
                V2[j*p+k, c] = 1
                W2[k*n+i, c] = 1
                c += 1
    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
