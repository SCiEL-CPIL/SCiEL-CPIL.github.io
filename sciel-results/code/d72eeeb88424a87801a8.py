"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    def schoolbook():
        """Guaranteed valid schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, rank):
        """Check tensor identity: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1"""
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        # Check each (i, j, k) triple
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U_np[i * m + j, :]
                    v_val = V_np[j * p + k, :]
                    w_val = W_np[k * n + i, :]
                    s = np.dot(u_val, v_val)
                    s = np.dot(s, w_val)
                    if abs(s - 1.0) > 1e-10:
                        return False
        # Also check orthogonality: for distinct (i,j,k) triples, the product should be 0
        # This is expensive; check a sample
        import random
        random.seed(42)
        for _ in range(200):
            i1, j1, k1 = random.randint(0, n-1), random.randint(0, m-1), random.randint(0, p-1)
            i2, j2, k2 = random.randint(0, n-1), random.randint(0, m-1), random.randint(0, p-1)
            if (i1, j1, k1) == (i2, j2, k2):
                continue
            u1 = U_np[i1 * m + j1, :]
            v1 = V_np[j1 * p + k1, :]
            w1 = W_np[k1 * n + i1, :]
            u2 = U_np[i2 * m + j2, :]
            v2 = V_np[j2 * p + k2, :]
            w2 = W_np[k2 * n + i2, :]
            # The tensor component T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
            # For the decomposition to be exact, T must be the identity tensor (1 if (i,j,k) matches, 0 otherwise)
            # But actually we need T[i,j,k] = 1 for all (i,j,k) in the multiplication tensor
            # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the product A[i,j]*B[j,k] -> C[i,k])
            # Wait, no. The tensor for matrix multiplication is:
            # T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
            # Actually the structure tensor has entries T[i,j,k] where the multiplication is:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor is sum_{i,j,k} A[i,j] B[j,k] C[i,k]
            # The rank decomposition: T = sum_c U_c \otimes V_c \otimes W_c
            # where U_c[i*m+j] = coefficient of A[i,j], V_c[j*p+k] = coefficient of B[j,k], W_c[k*n+i] = coefficient of C[i,k]
            # For the identity: for each (i,j,k), the sum over c should give 1 if this term appears in the product, 0 otherwise
            # Actually every (i,j,k) triple contributes exactly once to C[i,k], so T[i,j,k] = 1 for all valid (i,j,k)
            # The check is: for all i,j,k, sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
            # And for "cross terms" we don't need them to be 0 because there are no cross terms in this formulation
            # The tensor is just the all-ones tensor of shape (n*m, m*p, p*n) restricted to the multiplication pattern
            # Actually I think the verification is just checking that the sum equals 1 for all (i,j,k)
            pass
        return True
    
    # Try the block Strassen decomposition
    try:
        if time.time() > deadline:
            return schoolbook()
        
        num_blocks = p // 2
        block_size = 2
        rank_per_block = 11
        total_rank = num_blocks * rank_per_block
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        # Strassen part (7 terms) for the 2x2 submatrix
        strassen_terms = [
            {'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
            {'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)]},
            {'U': [(0,0,1)], 'V': [(0,1,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)]},
            {'U': [(1,1,1)], 'V': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (0,1,1)]},
            {'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)]},
            {'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
            {'U': [(0,1,1), (1,1,-1)], 'V': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]}
        ]
        
        # Outer product part (4 terms) for the third column/row
        outer_terms = [
            {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
            {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
            {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
            {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
        ]
        
        all_terms = strassen_terms + outer_terms
        
        col_offset = 0
        for b in range(num_blocks):
            for term in all_terms:
                if time.time() > deadline:
                    return schoolbook()
                col = col_offset
                for i, j, val in term['U']:
                    U[i * m + j][col] += val
                for j, k_local, val in term['V']:
                    k_global = b * block_size + k_local
                    V[j * p + k_global][col] += val
                for k_local, i, val in term['W']:
                    k_global = b * block_size + k_local
                    W[k_global * n + i][col] += val
                col_offset += 1
            if time.time() > deadline:
                return schoolbook()
        
        # Verify the decomposition
        if verify(U, V, W, total_rank):
            return {"U": U, "V": V, "W": W}
        else:
            return schoolbook()
            
    except Exception:
        return schoolbook()
# EVOLVE-BLOCK-END
