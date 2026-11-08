import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank
    rank = n * m * p
    
    # We will build the decomposition block by block.
    # We use Strassen's algorithm for the 2x2x2 block at indices i=0,1; j=0,1; k=0,1.
    # The rest is schoolbook.
    
    # Indices for the Strassen block
    i_range = [0, 1]
    j_range = [0, 1]
    k_range = [0, 1]
    
    # Total rank will be (n*m*p - 8) + 7 = 104
    # Let's construct the matrices
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # 1. Add Strassen block for i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen terms M1..M7
    # M1: (A00+A11)(B00+B11)
    # A00 is i=0,j=0. A11 is i=1,j=1.
    # B00 is j=0,k=0. B11 is j=1,k=1.
    # C00 is k=0,i=0. C11 is k=1,i=1.
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # M1: (A00+A11)(B00+B11)
    # U: A00=1, A11=1. V: B00=1, B11=1. W: C00=1, C11=1.
    # Wait, Strassen computes C from M.
    # M1 contributes to C00 and C11.
    # The tensor term is sum over r of U[i,j,r] V[j,k,r] W[k,i,r] = C[i,k].
    # For M1:
    # U[0,0] += 1, U[1,1] += 1
    # V[0,0] += 1, V[1,1] += 1
    # W[0,0] += 1, W[1,1] += 1  (C00 and C11)
    # But this is one rank-1 term? No, M1 is a product of sums.
    # (A00+A11)(B00+B11) is not a single rank-1 tensor in the CP decomposition sense of the multiplication tensor.
    # The multiplication tensor is T[i,j,k] = delta_{j,k} (for A*B).
    # Strassen replaces the 8 rank-1 terms of the 2x2x2 block with 7 rank-1 terms.
    # The 7 terms are linear combinations of the input entries.
    # The output is a linear combination of the M's.
    # So we need to express the 7 M's as rank-1 terms in the U, V, W decomposition.
    
    # Let's define the 7 terms explicitly.
    # Term 1: (A00+A11) * (B00+B11)
    # U[0,0]=1, U[1,1]=1
    # V[0,0]=1, V[1,1]=1
    # W[0,0]=1, W[1,1]=1
    # This is NOT a rank-1 term in the decomposition of the identity tensor.
    # The decomposition is C = U A V^T W? No.
    # The tensor is T_{ijk}. We want T = sum_r U_{ij,r} V_{jk,r} W_{ki,r}.
    # For the 2x2x2 block, T_{ijk} is 1 if j=k (in the local sense? No, j is inner index).
    # In the global tensor, T_{i,j,k} is non-zero only if the indices match the multiplication structure.
    # Actually, the tensor for matrix multiplication is T_{i,j,k} = 1 if we are computing C_{ik} from A_{ij} B_{jk}.
    # So for a fixed (i,j,k), the tensor entry is 1.
    # The schoolbook decomposition uses 1 term for each (i,j,k).
    # Strassen uses 7 terms for the 8 (i,j,k) combinations in the 2x2x2 block.
    
    # The 7 terms correspond to the 7 intermediate values M1..M7.
    # Each M is a linear combination of A's and B's.
    # So for each M, we have a rank-1 term in the decomposition?
    # No, the decomposition is of the tensor T.
    # T = sum_r u_r v_r w_r.
    # The entries of u_r, v_r, w_r are the coefficients of the linear forms.
    
    # Let's construct the 7 terms for the 2x2x2 block.
    # Local indices: i' in {0,1}, j' in {0,1}, k' in {0,1}.
    # Global indices: i=i', j=j', k=k'.
    
    # M1: (A00+A11)(B00+B11)
    # u1: U[0,0]=1, U[1,1]=1. Others 0.
    # v1: V[0,0]=1, V[1,1]=1. Others 0.
    # w1: W[0,0]=1, W[1,1]=1. Others 0. (Contributes to C00 and C11)
    # Wait, w1 must be a vector.
    # The term is u1 v1 w1^T?
    # The contribution to C_{ik} is sum_j A_{ij} B_{jk}.
    # The term r contributes U_{ij,r} V_{jk,r} W_{ki,r}.
    # So for M1:
    # U_{00,1}=1, U_{11,1}=1.
    # V_{00,1}=1, V_{11,1}=1.
    # W_{00,1}=1, W_{11,1}=1.
    # This term contributes to C_{00} and C_{11}.
    # C_{00} gets 1*1*1 = 1. C_{11} gets 1*1*1 = 1.
    # This matches M1 contributing to C00 and C11.
    
    add_term(0*2+0, 0*2+0, 0*2+0, 1, 1, 1) # U[0,0], V[0,0], W[0,0] -> C[0,0]
    add_term(1*2+1, 1*2+1, 1*2+1, 1, 1, 1) # U[1,1], V[1,1], W[1,1] -> C[1,1]
    # Wait, add_term adds to a single column.
    # U[0,0] and U[1,1] are in the same column r.
    # So I need to set U[0,0][r]=1, U[1,1][r]=1, etc.
    
    # Let's rewrite add_term to handle multiple indices.
    def add_rank1_term(r, u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        for idx, val in zip(u_indices, u_vals):
            U[idx][r] += val
        for idx, val in zip(v_indices, v_vals):
            V[idx][r] += val
        for idx, val in zip(w_indices, w_vals):
            W[idx][r] += val

    # Reset col
    col = 0
    
    # M1: (A00+A11)(B00+B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    add_rank1_term(col, [0*5+0, 1*5+1], [1, 1], [0*7+0, 1*7+1], [1, 1], [0*3+0, 1*3+1], [1, 1])
    col += 1
    
    # M2: (A10+A11)B00
    # U: (1,0)=1, (1,1)=1
    # V: (0,0)=1
    # W: (0,1)=1 (C10)
    add_rank1_term(col, [1*5+0, 1*5+1], [1, 1], [0*7+0], [1], [0*3+1], [1])
    col += 1
    
    # M3: A00(B10-B11)
    # U: (0,0)=1
    # V: (1,0)=1, (1,1)=-1
    # W: (0,0)=1 (C00)
    add_rank1_term(col, [0*5+0], [1], [1*7+0, 1*7+1], [1, -1], [0*3+0], [1])
    col += 1
    
    # M4: A11(B01-B00)
    # U: (1,1)=1
    # V: (0,1)=1, (0,0)=-1
    # W: (0,1)=1 (C10)
    add_rank1_term(col, [1*5+1], [1], [0*7+1, 0*7+0], [1, -1], [0*3+1], [1])
    col += 1
    
    # M5: (A00+A01)B11
    # U: (0,0)=1, (0,1)=1
    # V: (1,1)=1
    # W: (0,0)=1 (C00)
    add_rank1_term(col, [0*5+0, 0*5+1], [1, 1], [1*7+1], [1], [0*3+0], [1])
    col += 1
    
    # M6: (A10-A00)(B00+B01)
    # U: (1,0)=1, (0,0)=-1
    # V: (0,0)=1, (0,1)=1
    # W: (1,1)=1 (C11)
    add_rank1_term(col, [1*5+0, 0*5+0], [1, -1], [0*7+0, 0*7+1], [1, 1], [1*3+1], [1])
    col += 1
    
    # M7: (A01-A11)(B10+B11)
    # U: (0,1)=1, (1,1)=-1
    # V: (1,0)=1, (1,1)=1
    # W: (0,0)=1 (C00)
    add_rank1_term(col, [0*5+1, 1*5+1], [1, -1], [1*7+0, 1*7+1], [1, 1], [0*3+0], [1])
    col += 1
    
    # Now add schoolbook for all other (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip the 2x2x2 block
                if i in [0,1] and j in [0,1] and k in [0,1]:
                    continue
                add_rank1_term(col, [i*m+j], [1], [j*p+k], [1], [k*n+i], [1])
                col += 1
                
    return {"U": U, "V": V, "W": W}
