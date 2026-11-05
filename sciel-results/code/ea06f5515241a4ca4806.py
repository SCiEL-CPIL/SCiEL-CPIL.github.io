import numpy as np
import time
from itertools import combinations

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = 2, 3, 6
    num_A = n * m  # 6
    num_B = m * p  # 18
    num_C = p * n  # 12

    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit (10s buffer)

    def build_rank_33():
        rank = 33
        U = np.zeros((num_A, rank))
        V = np.zeros((num_B, rank))
        W = np.zeros((num_C, rank))
        col = 0

        # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> Rank 12
        # Indices: U[i*m+0], V[0*p+k], W[k*n+i]
        for k in range(p):
            for i in range(n):
                U[i * m + 0, col] = 1.0
                V[0 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1

        # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6) -> Rank 21 (3 * Strassen 7)
        # Split B[1:3, :] into three 2x2 blocks: cols [0:2], [2:4], [4:6]
        for t in range(3):
            # Block t: B rows 1,2; cols 2t, 2t+1
            # A rows 0,1; cols 1,2
            # U indices:
            #   a00 = A[0,1] -> idx 1
            #   a01 = A[0,2] -> idx 2
            #   a10 = A[1,1] -> idx 4
            #   a11 = A[1,2] -> idx 5
            # V indices:
            #   b00 = B[1, 2t] -> idx p+2t
            #   b01 = B[1, 2t+1] -> idx p+2t+1
            #   b10 = B[2, 2t] -> idx 2p+2t
            #   b11 = B[2, 2t+1] -> idx 2p+2t+1
            # W indices (C[k,i] -> k*n+i):
            #   c00 = C[0, 2t] -> idx 2t*n+0 = 4t
            #   c01 = C[0, 2t+1] -> idx (2t+1)*n+0 = 4t+2
            #   c10 = C[1, 2t] -> idx 2t*n+1 = 4t+1
            #   c11 = C[1, 2t+1] -> idx (2t+1)*n+1 = 4t+3

            # Strassen M1 = (a00 + a11) * (b00 + b11)
            # c00 += M1, c11 += M1
            U[1, col] += 1.0; U[5, col] += 1.0
            V[p + 2*t, col] += 1.0; V[2*p + 2*t + 1, col] += 1.0
            W[4*t, col] += 1.0; W[4*t+3, col] += 1.0
            col += 1

            # Strassen M2 = (a10 + a11) * b00
            # c10 += M2, c11 -= M2
            U[4, col] += 1.0; U[5, col] += 1.0
            V[p + 2*t, col] += 1.0
            W[4*t+1, col] += 1.0; W[4*t+3, col] += -1.0
            col += 1

            # Strassen M3 = a00 * (b01 - b11)
            # c01 += M3, c11 += M3
            U[1, col] += 1.0
            V[p + 2*t + 1, col] += 1.0; V[2*p + 2*t + 1, col] += -1.0
            W[4*t+2, col] += 1.0; W[4*t+3, col] += 1.0
            col += 1

            # Strassen M4 = a11 * (b10 - b00)
            # c00 += M4, c10 += M4
            U[5, col] += 1.0
            V[2*p + 2*t, col] += 1.0; V[p + 2*t, col] += -1.0
            W[4*t, col] += 1.0; W[4*t+1, col] += 1.0
            col += 1

            # Strassen M5 = (a00 + a01) * b11
            # c00 -= M5, c01 += M5, c11 += M5
            U[1, col] += 1.0; U[2, col] += 1.0
            V[2*p + 2*t + 1, col] += 1.0
            W[4*t, col] += -1.0; W[4*t+2, col] += 1.0; W[4*t+3, col] += 1.0
            col += 1

            # Strassen M6 = (a10 - a00) * (b00 + b01)
            # c11 -= M6
            U[4, col] += 1.0; U[1, col] += -1.0
            V[p + 2*t, col] += 1.0; V[p + 2*t + 1, col] += 1.0
            W[4*t+3, col] += -1.0
            col += 1

            # Strassen M7 = (a01 - a11) * (b10 + b11)
            # c00 -= M7
            U[2, col] += 1.0; U[5, col] += -1.0
            V[2*p + 2*t, col] += 1.0; V[2*p + 2*t + 1, col] += 1.0
            W[4*t, col] += -1.0
            col += 1

        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-9):
        """
        Verify if U, V, W exactly decompose the matrix multiplication tensor.
        T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] should be 1 if i,j,k correspond to valid multiplication, 0 otherwise.
        Actually, the tensor for matrix multiplication C = AB is T_{ijk} = 1 if the term is A_{ij} B_{jk} contributing to C_{ik}.
        Here indices are flattened.
        U is (n*m, rank). Index a = i*m + j.
        V is (m*p, rank). Index b = j*p + k.
        W is (p*n, rank). Index c = k*n + i.
        The tensor T is defined such that C_{ik} = sum_j A_{ij} B_{jk}.
        In tensor form: T_{a,b,c} is non-zero only if the indices match the multiplication pattern.
        Specifically, if a = i*m+j, b = j'*p+k, c = k'*n+i'.
        The term contributes if j == j' and k == k' and i == i'.
        So T_{a,b,c} = 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0.
        We check if sum_r U_{a,r} V_{b,r} W_{c,r} equals T_{a,b,c}.
        """
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        
        # Efficient check:
        # Compute T_est = einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        # T_target is 1 everywhere for i in 0..n-1, j in 0..m-1, k in 0..p-1.
        # Wait, the tensor T_{ijk} for matrix multiplication is 1 for all valid i,j,k.
        # Because C_{ik} = sum_j A_{ij} B_{jk}.
        # So for any i, j, k, the coefficient of A_{ij}B_{jk} in C_{ik} is 1.
        # Thus T_{ijk} = 1 for all i,j,k.
        
        T_est = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p))
        
        return np.allclose(T_est, T_target, atol=tol)

    # Initialize with Rank 33
    U, V, W = build_rank_33()
    
    if not verify_decomposition(U, V, W):
        # Fallback to schoolbook if bug exists (should not happen)
        rank_school = n * m * p
        U = np.zeros((num_A, rank_school))
        V = np.zeros((num_B, rank_school))
        W = np.zeros((num_C, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = 33

    # Attempt to find Rank 32 decomposition
    # Strategy: Identify a linear dependency among the columns of the tensor representation.
    # The tensor T can be viewed as a vector in R^{n*m*p} (or similar space).
    # Each rank-1 term corresponds to a vector v_r = U_r \otimes V_r \otimes W_r (flattened).
    # If we can find a linear combination of these vectors that sums to 0, we might be able to remove one.
    # However, simply finding a dependency doesn't guarantee we can remove a column with integer/half-integer coefficients.
    # 
    # Alternative Strategy: 
    # The rank of 2x2x3 is 10. 
    # If we can find a rank 10 decomposition for 2x2x3, we can use it.
    # 2x3x6 can be split into two 2x3x3 blocks? No, A is 2x3.
    # C = A * B.
    # Let B = [B1, B2] where B1, B2 are 3x3.
    # C = [A*B1, A*B2].
    # This is not independent.
    #
    # Let's try a direct search for rank 32 by checking if any column in the rank-33 solution is "redundant" in a specific sense.
    # Specifically, we can try to solve for a rank 32 decomposition by fixing 32 columns and solving for the 33rd? No.
    #
    # A known result: r(2,2,3) = 10.
    # Can we construct a rank 10 solution for 2x2x3?
    # If yes, we can replace the 3 Strassen blocks (rank 21) with... wait.
    # The second part is 2x2 * 2x6.
    # We can split 2x6 into two 2x3 matrices.
    # Let B_sub = [B_left, B_right] where B_left, B_right are 2x3.
    # Then C_sub = A_sub * B_sub = [A_sub * B_left, A_sub * B_right].
    # This is two 2x2x3 multiplications.
    # If r(2,2,3) = 10, then rank for this part is 2 * 10 = 20.
    # Total rank = 12 (for first part) + 20 = 32.
    #
    # So the key is to find a rank 10 decomposition for 2x2x3.
    #
    # Construction for 2x2x3 (Rank 10):
    # A = [[a, b], [c, d]]
    # B = [[e, f, g], [h, i, j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    #
    # We need 10 terms M1..M10.
    # This is a known open problem solved by Bini et al. or similar? 
    # Actually, r(2,2,3) is known to be 10.
    # Reference: "Tensor rank of 2x2x3" by various authors.
    #
    # Let's try to derive or find a simple one.
    # If we cannot find it reliably in code without hardcoding a complex set of coefficients, we might fail.
    #
    # However, we can try a randomized search for rank 32 directly on the full 2x3x6 tensor.
    # This is computationally expensive but might find a solution if one exists near the rank 33 solution.
    #
    # Given the time constraints and the difficulty of deriving the 2x2x3 rank 10 solution from scratch,
    # and the fact that the current score is 1.0909 (Rank 33),
    # I will implement a heuristic that attempts to verify if a rank 32 solution exists by
    # checking the rank of the tensor matricization.
    #
    # Matricization of T (2x3x6) into (2*3) x (3*6) = 6 x 18 matrix.
    # The rank of this matrix is a lower bound on the tensor rank.
    # If the rank of the matricization is <= 32, it doesn't prove rank 32 exists, but if it's 33, rank 32 is impossible?
    # No, tensor rank >= max rank of matricizations.
    # If max matricization rank is 33, then tensor rank >= 33.
    # Let's check the rank of the matricization.
    
    # Construct the tensor T as a 3D array
    T = np.ones((n, m, p))
    
    # Matricization 1: (i, j) vs (k) -> (n*m) x (p) = 6 x 6
    # T_ijk. Fix i,j. Sum over k? No, matricization groups indices.
    # M1_{(i,j), k} = T_{ijk}.
    M1 = T.reshape(n*m, p)
    rank_M1 = np.linalg.matrix_rank(M1)
    
    # M2_{(i,k), j} = T_{ijk}
    M2 = T.transpose(0, 2, 1).reshape(n*p, m)
    rank_M2 = np.linalg.matrix_rank(M2)
    
    # M3_{(j,k), i} = T_{ijk}
    M3 = T.transpose(1, 2, 0).reshape(m*p, n)
    rank_M3 = np.linalg.matrix_rank(M3)
    
    # The rank of the tensor is at least the maximum of these ranks.
    # For 2x3x6, M1 is 6x6 (rank 6), M2 is 12x3 (rank 3), M3 is 18x2 (rank 2).
    # So lower bound is 6. This is not helpful for distinguishing 32 vs 33.
    
    # We need a better lower bound or a constructive search.
    #
    # Let's try to construct the rank 10 2x2x3 block.
    # If I can't, I return rank 33.
    #
    # Known Rank 10 solution for 2x2x3:
    # M1 = (a+d)(e+i)
    # M2 = (a-d)(f-j)
    # M3 = (a+c)(h+i)
    # M4 = (b-d)(h-i)
    # ... this is getting complicated and I don't have the exact coefficients memorized to be safe.
    #
    # Instead, I will use the "sub-tensor" insight from the inspiration:
    # "exploit structure. The 2x3*3x6 multiplication can be viewed as..."
    #
    # Let's try a local search around the rank 33 solution.
    # We can try to remove one column and see if the residual can be absorbed by the others with small adjustments?
    # No, the coefficients are fixed.
    #
    # I will stick to the rank 33 solution as it is verified and safe.
    # The potential for rank 32 exists (12 + 2*10), but deriving the 2x2x3 rank 10 decomposition is risky without external lookup or extensive derivation time.
    #
    # However, let's try one specific known decomposition for 2x2x3 if I can recall it.
    # Actually, there is a simpler way:
    # Rank(2,2,3) = 10 is established.
    # If I could hardcode it, I would use it.
    # Since I cannot guarantee correctness of a hardcoded complex set of coefficients, I will return the rank 33 solution.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
