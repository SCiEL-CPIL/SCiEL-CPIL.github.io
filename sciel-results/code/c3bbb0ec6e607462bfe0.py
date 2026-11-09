"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    # Dimensions: A (2x4), B (4x8), C (2x8)
    # U shape: (n*m, rank) = (8, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (16, rank)
    
    # Strategy: Tiling into 8 blocks of 2x2x2 Strassen multiplications.
    # A is split into 2 column-blocks of 2x2: A0 (cols 0-1), A1 (cols 2-3)
    # B is split into 4 row-blocks of 2x8? No.
    # B is 4x8.
    # We split the shared dimension m=4 into two blocks of size 2: m0 (rows 0-1), m1 (rows 2-3).
    # A = [A0, A1] where A0, A1 are 2x2.
    # B = [B0; B1] where B0, B1 are 2x8.
    # C = A0 B0 + A1 B1.
    
    # Now we need to compute A0 B0 (2x2 * 2x8) and A1 B1 (2x2 * 2x8).
    # A0 B0: Split B0 (2x8) into 4 blocks of 2x2: B0_0, B0_1, B0_2, B0_3.
    # A0 B0 = A0 B0_0 + A0 B0_1 + A0 B0_2 + A0 B0_3.
    # Each A0 B0_j is a 2x2x2 multiplication.
    # Rank of each is 7.
    # Total for A0 B0 is 4 * 7 = 28.
    # Total for A1 B1 is 4 * 7 = 28.
    # Total rank = 56.
    
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 decomposition (Rank 7)
    # Indices for 2x2 matrix: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # U_vec, V_vec, W_vec for each of the 7 terms.
    # Term 1: U=[1,0,0,0], V=[0,1,-1,0], W=[0,1,0,1]
    # Term 2: U=[1,0,1,0], V=[0,0,0,1], W=[-1,0,1,0]
    # Term 3: U=[0,1,0,-1], V=[1,0,0,0], W=[1,0,0,-1]
    # Term 4: U=[0,0,0,1], V=[0,-1,1,0], W=[1,0,1,0]
    # Term 5: U=[1,1,0,0], V=[1,0,0,1], W=[1,1,0,0]
    # Term 6: U=[0,0,1,1], V=[1,1,0,0], W=[0,0,1,1]
    # Term 7: U=[1,0,-1,0], V=[0,0,1,1], W=[0,1,0,-1]
    
    strassen_terms = [
        # (U_indices, V_indices, W_indices)
        # Values are implicitly 1 unless specified, but we need signs.
        # Let's store (U_dict, V_dict, W_dict)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({0: 1, 2: 1}, {3: 1}, {0: -1, 2: 1}),
        ({1: 1, 3: -1}, {0: 1}, {0: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {0: 1, 3: 1}, {0: 1, 1: 1}),
        ({2: 1, 3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}),
        ({0: 1, 2: -1}, {2: 1, 3: 1}, {1: 1, 3: -1}),
    ]
    
    col = 0
    # Iterate over the 8 blocks
    # a_block: 0 or 1 (which 2x2 block of A)
    # b_col_block: 0, 1, 2, 3 (which 2x2 block of B's columns)
    # The row block of B is determined by a_block (A0 pairs with B top, A1 with B bottom)
    
    for a_block in range(2):
        # A column offset
        a_col_off = a_block * 2
        # B row offset (A0 uses B rows 0-1, A1 uses B rows 2-3)
        b_row_off = a_block * 2
        
        for b_col_block in range(4):
            # B column offset
            b_col_off = b_col_block * 2
            
            # C column offset (C has 8 cols, 4 blocks of 2)
            c_col_off = b_col_block * 2
            
            # Apply 7 Strassen terms
            for u_dict, v_dict, w_dict in strassen_terms:
                # Fill U
                for u_idx, val in u_dict.items():
                    # u_idx 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    # A index: row u_idx//2, col a_col_off + u_idx%2
                    a_row = u_idx // 2
                    a_col = a_col_off + (u_idx % 2)
                    U[a_row * m + a_col, col] = val
                
                # Fill V
                for v_idx, val in v_dict.items():
                    # v_idx 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    # B index: row b_row_off + v_idx//2, col b_col_off + v_idx%2
                    b_row = b_row_off + (v_idx // 2)
                    b_col = b_col_off + (v_idx % 2)
                    V[b_row * p + b_col, col] = val
                
                # Fill W
                for w_idx, val in w_dict.items():
                    # w_idx 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
                    # C index: row w_idx//2, col c_col_off + w_idx%2
                    # W shape (p*n, rank). Index k*n+i.
                    # k is col in C (0..7), i is row in C (0..1).
                    # Index = c_col * n + c_row.
                    c_row = w_idx // 2
                    c_col = c_col_off + (w_idx % 2)
                    W[c_col * n + c_row, col] = val
                
                col += 1
    
    # Verify the decomposition
    # Check T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} == 1
    # Note: W index in tensor is (k, i). In matrix W, it is stored as W[k*n + i, c].
    # U index is (i, j). Stored as U[i*m + j, c].
    # V index is (j, k). Stored as V[j*p + k, c].
    
    # We can do a fast check using numpy
    # Tensor T is sparse: T[i, j, k] = 1 if j is the shared index?
    # Actually, the multiplication tensor is T[i, j, k] = 1 for all i, j, k?
    # No, the tensor defines the multiplication: C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The tensor entries are 1 if the indices (i,j,k) form a valid term in the sum.
    # Since every (i,j,k) is a valid term (A_{i,j} * B_{j,k}), T[i,j,k] = 1 for all i,j,k.
    # Wait, if T[i,j,k] = 1 for all i,j,k, then the rank is just the number of non-zero entries?
    # No, the rank of the tensor is the minimal number of rank-1 tensors summing to it.
    # The tensor M_{n,m,p} has entries 1 at (i,j,k) corresponding to the multiplication.
    # For standard matrix multiplication, T[i,j,k] = 1 if j is the contraction index.
    # Here, for fixed i, k, we sum over j.
    # So T[i,j,k] = 1 for all i, j, k.
    # Is that right?
    # If T[i,j,k] = 1 for all i,j,k, then the tensor is the all-ones tensor.
    # The rank of the all-ones tensor of size n x m x p is max(n,m,p)? No.
    # The rank of the all-ones tensor is 1? No, it's not rank 1.
    # Actually, the matrix multiplication tensor is NOT the all-ones tensor.
    # The tensor is defined such that the trilinear form is sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}?
    # No, the decomposition is sum_c U_c A V_c B W_c C = 0?
    # The condition is: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k].
    # For matrix multiplication, T[i,j,k] = 1 if j is the shared index?
    # Yes, T[i,j,k] = 1 for all valid i,j,k.
    # So we need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    
    # Let's verify with numpy
    # Create T (2, 4, 8) filled with 1s
    T = np.ones((n, m, p))
    
    # Compute sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # This is a tensor contraction.
    # We can do this by looping or using einsum.
    # np.einsum('ijc,jkc,kic->ijk', U, V, W)
    
    try:
        computed_T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        if not np.allclose(computed_T, T, atol=1e-9):
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j, c] = 1
                        V[j*p + k, c] = 1
                        W[k*n + i, c] = 1
                        c += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        # Fallback
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, c] = 1
                    V[j*p + k, c] = 1
                    W[k*n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
