import numpy as np
import time
from itertools import product

# Matrix multiplication 2x3 x 3x9
# A: 2x3, B: 3x9, C: 2x9
# U: (n*m, rank) = (6, rank)
# V: (m*p, rank) = (27, rank)
# W: (p*n, rank) = (18, rank)

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the tensor decomposition of 2x3 * 3x9 matrix multiplication.
    """
    n, m, p = 2, 3, 9
    
    # Base rank-1 tensors for the full multiplication
    # We need a decomposition of the tensor T_{i,j,k} = delta_{i,j} delta_{j,k}
    # Wait, the tensor is T_{i, j, k} such that C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor T is a (2, 3, 9) tensor where T_{i, j, k} = 1 if (i,k) corresponds to j? No.
    # The multiplication tensor M_{n,m,p} has entries M_{i, j, k} = delta_{i, j} delta_{j, k}? No.
    # The tensor is defined such that <T, A \otimes B> = C flattened?
    # T is of shape (n*m, m*p, p*n) ? No, the factors are U (n*m, r), V (m*p, r), W (p*n, r).
    # The reconstruction is C = (W^T U) \circ ...? No.
    # The standard decomposition is T = sum_r U[:, r] \otimes V[:, r] \otimes W[:, r]
    # where T is the tensor of shape (n*m, m*p, p*n).
    # The entry T_{idx_u, idx_v, idx_w} is 1 if the term A[idx_u] B[idx_v] contributes to C[idx_w].
    # idx_u = i*m + j
    # idx_v = j*p + k
    # idx_w = k*n + i
    
    # Let's build the full rank-1 tensors for the "naive" multiplication first.
    # Naive rank = m * n * p = 3 * 2 * 9 = 54.
    # For each j in 0..2, i in 0..1, k in 0..8:
    # u = i*m + j
    # v = j*p + k
    # w = k*n + i
    
    # We can optimize this.
    # We can split the multiplication into blocks.
    
    # Let's try to find a lower rank decomposition using known algorithms or heuristics.
    # However, for small dimensions, we can try to solve for a lower rank using optimization or known bounds.
    # The border rank of 2x2x2 is 7.
    # Here we have 2x3x9.
    
    # Let's try to construct a decomposition with rank R.
    # We can use the fact that the tensor is linear in each mode.
    
    # A simple valid decomposition is the naive one with rank 54.
    # Can we do better?
    
    # Let's try to use Strassen-like tricks for the 2x2 parts.
    # Split A into A_left (2x2) and A_right (2x1).
    # Split B into B_top (2x9) and B_bottom (1x9).
    
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top is 2x2 * 2x9.
    # We can treat B_top as 9 columns of 2x1 vectors.
    # So we need 2x2 * 2x1 multiplication for 9 columns.
    # Rank of 2x2*2x1?
    # T_{i,j,k} for i in 0..1, j in 0..1, k in 0..1 (since 2x1 output is just 2x1? No, output is 2x1 per column).
    # Actually, 2x2 * 2x1 is a tensor of shape (4, 2, 2).
    # Naive rank is 2*2*1 = 4? No, m=2, n=2, p=1. Rank = n*m*p = 4.
    # Is there a better rank?
    # The tensor M_{2,2,1} has rank 4. It's equivalent to a 4x2 matrix multiplication?
    # Let's check.
    # C_i = sum_j A_{i,j} B_{j,0}.
    # C_0 = A_00 B_00 + A_01 B_10
    # C_1 = A_10 B_00 + A_11 B_10
    # This is just two dot products of length 2.
    # Can we compute two dot products of length 2 with rank < 4?
    # Rank of two dot products?
    # T = e_0 \otimes (e_0 \otimes e_0 + e_1 \otimes e_1) + e_1 \otimes (e_0 \otimes e_0 + e_1 \otimes e_1)? No.
    # The tensor is T_{i,j,k} = delta_{i,j} delta_{j,k}? No, k is the output index.
    # T_{i, j, k} = 1 if i=j and k is the output? No.
    # For 2x2*2x1, output C is 2x1.
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    # The tensor entries T_{u, v, w} where u is (i,j), v is j, w is i?
    # u = i*2+j. v = j. w = i.
    # T_{u,v,w} = 1 if u = w*2+v.
    # (0,0,0): u=0, v=0, w=0. 0=0*2+0. Yes.
    # (1,1,0): u=1, v=1, w=0. 1=0*2+1. Yes.
    # (2,0,1): u=2, v=0, w=1. 2=1*2+0. Yes.
    # (3,1,1): u=3, v=1, w=1. 3=1*2+1. Yes.
    # So the tensor is a permutation of 4 basis vectors.
    # The rank of a permutation tensor is its support size? No.
    # The rank of a diagonal tensor is its dimension.
    # This tensor is "diagonal" in a generalized sense.
    # It has 4 non-zero entries.
    # It is a direct sum of 4 rank-1 tensors?
    # T = e_{0,0,0} + e_{1,1,0} + e_{2,0,1} + e_{3,1,1}.
    # Each term is rank 1.
    # Can we combine them?
    # (e_0 + e_1) \otimes (e_0 + e_1) \otimes e_0 = e_{0,0,0} + e_{0,1,0} + e_{1,0,0} + e_{1,1,0}.
    # This introduces cross terms.
    # We need to cancel cross terms.
    # This is similar to Strassen.
    # However, for 2x2*2x1, the rank is known to be 4? Or can it be 3?
    # If we use 3 terms, we have 3*6=18 variables (if entries are generic).
    # The tensor has 12 entries (4x2x2), 4 of which are 1, 8 are 0.
    # Constraints: 12 equations.
    # It is possible rank is 4.
    
    # Let's assume rank 4 for 2x2*2x1.
    # Then for 9 columns, rank 9*4 = 36.
    # A_right * B_bottom is 2x1 * 1x9.
    # C_i = A_{i,2} B_{2,k}.
    # This is 2x1*1x1 for each k.
    # Tensor shape (2, 1, 2).
    # C_0 = A_02 B_20
    # C_1 = A_12 B_20
    # u = i*3+2. v = 2*9+k. w = k*2+i.
    # T_{u,v,w} = 1 if u = i*3+2, v = 2*9+k, w = k*2+i.
    # For a fixed k, we have two terms:
    # i=0: u=2, v=18+k, w=2k
    # i=1: u=5, v=18+k, w=2k+1
    # These are two separate rank-1 terms.
    # Can we combine them?
    # (e_2 + e_5) \otimes e_{18+k} \otimes (e_{2k} + e_{2k+1})?
    # This gives e_{2,18+k,2k} + e_{2,18+k,2k+1} + e_{5,18+k,2k} + e_{5,18+k,2k+1}.
    # We want e_{2,18+k,2k} + e_{5,18+k,2k+1}.
    # Cross terms: e_{2,18+k,2k+1} and e_{5,18+k,2k}.
    # We can't easily cancel these with a single term.
    # So rank is likely 2 per column.
    # Total rank for A_right * B_bottom is 9*2 = 18.
    
    # Total rank = 36 + 18 = 54.
    # This is the same as the naive rank.
    
    # Wait, can we do better on the 2x2*2x9 part?
    # We are computing 2x2 * 2x9.
    # This is equivalent to 9 instances of 2x2*2x1.
    # If the rank of 2x2*2x1 is 4, total is 36.
    # But maybe we can share computations across columns?
    # No, the columns are independent in B.
    # However, the A matrix is the same.
    # So we can precompute linear combinations of A that are used for multiple B columns?
    # In tensor decomposition, we sum over the rank index.
    # T = sum_r U_r \otimes V_r \otimes W_r.
    # If we group by A-combinations, we might save rank?
    # No, the rank is the number of terms in the sum.
    
    # Let's check if 2x2*2x1 can be done in rank 3.
    # If rank 3, then 2x2*2x9 might be done in less than 36?
    # Actually, if we treat it as 2x2 * 2x9, the tensor is M_{2,2,9}.
    # The rank of M_{2,2,9} is at least rank(M_{2,2,2}) = 7?
    # No, M_{2,2,9} contains M_{2,2,2} as a sub-tensor.
    # So rank(M_{2,2,9}) >= rank(M_{2,2,2}) = 7.
    # But naive is 4*9 = 36.
    # Is it possible to do much better?
    # The border rank of M_{2,2,9} is likely close to 36?
    # Actually, M_{n,m,p} rank is roughly n*m*p.
    # Strassen gives 2x2*2x2 in 7.
    # So M_{2,2,9} might be doable in 7 * 9 / 4 * something?
    # No, the columns are independent.
    # However, the A factors can be shared.
    # Let's try to construct a decomposition.
    
    # We will use the naive decomposition for now, as finding a significantly lower rank for 2x3*3x9
    # is a hard problem and likely requires advanced algorithms.
    # The naive rank is 54.
    # The current program score is 1.08, which implies rank 50?
    # Wait, the score is seed/value. If seed is 1.08 * value?
    # The prompt says "score is the rank".
    # "The evaluator measures the raw value (lower is better). The search score is seed/value".
    # If current score is 1.0800, and it is an improvement over 1.0, then value < seed.
    # If seed value is 54 (naive), and score is 1.08, then 54/50 = 1.08.
    # So the current program has rank 50.
    # This means a rank 50 decomposition exists and is implemented in the current program.
    # I should try to replicate or improve upon rank 50.
    
    # How was rank 50 achieved?
    # 2x2*2x9 part: Rank 32?
    # 2x1*1x9 part: Rank 18?
    # 32 + 18 = 50.
    # How to get rank 32 for 2x2*2x9?
    # 9 columns. 32/9 is not integer.
    # Maybe some columns use rank 4, some use rank 3?
    # Or maybe the whole 2x2*2x9 block is rank 32.
    # 32 = 7 + 25?
    # Let's look at the structure.
    # 2x2*2x9.
    # Maybe split 9 into 4+4+1.
    # 4 columns: 2x2*2x4. Rank 7*4/2? No.
    # If 2x2*2x2 is rank 7.
    # 2x2*2x4 might be rank 7*2 = 14? (since 4 cols = 2 blocks of 2).
    # So 2 blocks of 2x2*2x2 -> rank 14.
    # 2 more blocks -> rank 14.
    # Total 28.
    # Remaining 1 column: 2x2*2x1. Rank 4.
    # Total 28 + 4 = 32.
    # Plus 18 for the rest.
    # Total 50.
    
    # This matches!
    # So the strategy is:
    # Split B_top (2x9) into four 2x2 blocks and one 2x1 block.
    # Use Strassen (rank 7) for each 2x2 block.
    # Use naive (rank 4) for the 2x1 block.
    # Use naive (rank 2) for each column of B_bottom (1x9).
    
    # Let's implement this.
    
    R = 50
    U = np.zeros((6, R), dtype=float)
    V = np.zeros((27, R), dtype=float)
    W = np.zeros((18, R), dtype=float)
    
    col = 0
    
    # Strassen 2x2*2x2
    # Inputs: A (2x2), B (2x2). Output C (2x2).
    # U indices: A_{00}=0, A_{01}=1, A_{10}=3, A_{11}=4. (Assuming A is 2x3, left part 2x2).
    # Wait, A is 2x3. A_left is A[:, 0:2].
    # A_{00} is U[0], A_{01} is U[1], A_{10} is U[3], A_{11} is U[4].
    # B_top is B[0:2, :]. B is 3x9.
    # For a block of columns k0, k1.
    # B_{0,k0} is V[0*9+k0], B_{0,k1} is V[0*9+k1].
    # B_{1,k0} is V[1*9+k0], B_{1,k1} is V[1*9+k1].
    # W indices: C_{i,k}. C is 2x9.
    # W[k*n+i] = W[k*2+i].
    
    def strassen_block(k0, k1):
        nonlocal col
        # U indices
        a = 0 # A00
        b = 1 # A01
        c = 3 # A10
        d = 4 # A11
        
        # V indices
        e = k0 # B0, k0
        f = k1 # B0, k1
        g = 9 + k0 # B1, k0
        h = 9 + k1 # B1, k1
        
        # W indices
        w00 = k0 * 2 # C0, k0
        w01 = k1 * 2 # C0, k1
        w10 = k0 * 2 + 1 # C1, k0
        w11 = k1 * 2 + 1 # C1, k1
        
        # M1 = (A00 + A11) * (B00 + B11) -> C00 += M1, C11 += M1
        U[a, col] += 1
        U[d, col] += 1
        V[e, col] += 1
        V[h, col] += 1
        W[w00, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M2 = (A10 + A11) * B00 -> C10 += M2, C11 -= M2
        U[c, col] += 1
        U[d, col] += 1
        V[e, col] += 1
        W[w10, col] += 1
        W[w11, col] -= 1
        col += 1
        
        # M3 = A00 * (B01 - B11) -> C01 += M3, C11 += M3
        U[a, col] += 1
        V[f, col] += 1
        V[h, col] -= 1
        W[w01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M4 = A11 * (B10 - B00) -> C00 += M4, C10 += M4
        U[d, col] += 1
        V[g, col] += 1
        V[e, col] -= 1
        W[w00, col] += 1
        W[w10, col] += 1
        col += 1
        
        # M5 = (A00 + A01) * B11 -> C00 -= M5, C01 += M5
        U[a, col] += 1
        U[b, col] += 1
        V[h, col] += 1
        W[w00, col] -= 1
        W[w01, col] += 1
        col += 1
        
        # M6 = (A10 - A00) * (B00 + B01) -> C11 += M6
        U[c, col] += 1
        U[a, col] -= 1
        V[e, col] += 1
        V[f, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M7 = (A01 - A11) * (B10 + B11) -> C00 += M7
        U[b, col] += 1
        U[d, col] -= 1
        V[g, col] += 1
        V[h, col] += 1
        W[w00, col] += 1
        col += 1
        
    # Apply Strassen to 4 blocks of 2 columns
    for i in range(4):
        strassen_block(2*i, 2*i+1)
        
    # Last column (k=8) of B_top
    # 2x2 * 2x1. Naive rank 4.
    k = 8
    a, b, c, d = 0, 1, 3, 4
    e = k # B0, k
    g = 9 + k # B1, k
    w0 = k * 2 # C0, k
    w1 = k * 2 + 1 # C1, k
    
    # C0 = A00*B0 + A01*B1
    U[a, col] += 1
    V[e, col] += 1
    W[w0, col] += 1
    col += 1
    
    U[b, col] += 1
    V[g, col] += 1
    W[w0, col] += 1
    col += 1
    
    # C1 = A10*B0 + A11*B1
    U[c, col] += 1
    V[e, col] += 1
    W[w1, col] += 1
    col += 1
    
    U[d, col] += 1
    V[g, col] += 1
    W[w1, col] += 1
    col += 1
    
    # Now A_right (2x1) * B_bottom (1x9)
    # A_right indices: A02=2, A12=5.
    # B_bottom indices: B2,k = 18+k.
    # W indices: C_{i,k} = k*2+i.
    
    ar0 = 2 # A02
    ar1 = 5 # A12
    
    for k in range(9):
        v = 18 + k
        
        # C0,k += A02 * B2,k
        U[ar0, col] += 1
        V[v, col] += 1
        W[k*2, col] += 1
        col += 1
        
        # C1,k += A12 * B2,k
        U[ar1, col] += 1
        V[v, col] += 1
        W[k*2+1, col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U, V, W = res['U'], res['V'], res['W']
    print(U.shape, V.shape, W.shape)
    
