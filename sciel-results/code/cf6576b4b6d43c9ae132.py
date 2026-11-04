import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

def _build_mult_tensor(n, m, p):
    """
    Build the matrix M such that vec(C) = M * (vec(A) ⊗ vec(B))
    where vec uses row-major order.
    U shape: (n*m, rank) -> U[i*m+j, s]
    V shape: (m*p, rank) -> V[j*p+k, s]
    W shape: (p*n, rank) -> W[k*n+i, s]
    
    The tensor T[i,j,k,s] = U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    We want sum_s T[i,j,k,s] = 1 for all i,j,k.
    
    Alternatively, think of it as:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    Let's define the flattening:
    A_flat index: i*m + j
    B_flat index: j*p + k
    C_flat index: k*n + i  (per the task contract: W[k*n+i, r])
    
    The multiplication tensor M has dimensions (p*n) x (n*m * m*p)
    M[c_idx, a_idx * (m*p) + b_idx] = 1 if the term A[a_idx] * B[b_idx] contributes to C[c_idx]
    
    a_idx = i*m + j
    b_idx = j'*p + k
    c_idx = k'*n + i'
    
    Contribution is 1 if i == i', j == j', k == k'.
    """
    num_a = n * m
    num_b = m * p
    num_c = p * n
    
    M = np.zeros((num_c, num_a * num_b), dtype=int)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                M[c_idx, a_idx * num_b + b_idx] = 1
                
    return M

def _verify_solution(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] == 1 for all i,j,k.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    try:
        S = np.einsum('ijs,jks,ksi->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(S, 1.0, atol=tol)
    except Exception:
        return False

def _schoolbook_solution(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W, "rank": rank}

def _try_rank_decomposition(n, m, p, target_rank, time_limit=10):
    """
    Attempt to find a decomposition of rank `target_rank`.
    This is a placeholder for a more sophisticated search.
    For now, we will use a greedy exact algebraic approach if possible,
    but given the complexity, we will fall back to known bounds or heuristics.
    
    Actually, let's implement a specific known lower rank if we can find one.
    Current best known for 3x4x8?
    Schoolbook: 96.
    Strassen 2x2x2 blocks: 88 (as in current code).
    
    Can we do better?
    3x4x8 can be split into:
    3x2x8 + 3x2x8? No, that's 3x4x8.
    3x2x8 rank?
    If we use Strassen for 2x2x2 blocks inside 3x2x8:
    3x2x8 = 3 rows, 2 cols, 8 cols.
    Split k into 4 blocks of 2.
    Each 3x2x2 block.
    Rank of 3x2x2?
    Schoolbook 12.
    Can we do better?
    3x2x2 is small.
    Let's check if 3x2x2 has rank < 12.
    
    Actually, let's try to construct a solution using a different block structure.
    Split the 4 columns into 1+1+1+1?
    Split the 8 rows into 2+2+2+2?
    
    Let's try to use the fact that 3x4x8 contains 3x2x4 blocks.
    3x2x4 rank?
    Schoolbook 24.
    Laderman 2x3x4 is rank 14.
    By symmetry, 3x2x4 is rank 14.
    
    So if we split the 4 columns into two 2-column blocks:
    Block 1: 3x2x8.
    Block 2: 3x2x8.
    
    If we use Laderman on 3x2x4 sub-blocks of the 3x2x8?
    3x2x8 = two 3x2x4 blocks.
    Rank = 14 + 14 = 28 per 3x2x8 block?
    Total rank = 28 + 28 = 56?
    
    Let's verify this structure.
    A is 3x4. B is 4x8.
    Split A into A1 (3x2) and A2 (3x2).
    Split B into B1 (2x8) and B2 (2x8).
    C = A1*B1 + A2*B2.
    
    A1 is 3x2. B1 is 2x8.
    Multiply 3x2 by 2x8.
    This is a 3x2x8 multiplication.
    
    Can we decompose 3x2x8 efficiently?
    Split B1 into B1a (2x4) and B1b (2x4).
    C1 = A1*B1a + A1*B1b.
    
    A1 (3x2) * B1a (2x4).
    This is 3x2x4.
    Laderman says rank 14.
    
    So A1*B1a rank 14.
    A1*B1b rank 14.
    Total for A1*B1 is 28.
    
    Similarly for A2*B2 is 28.
    Total rank 56.
    
    Let's implement Laderman 3x2x4 (or 2x3x4 transposed).
    Laderman 2x3x4: A(2x3), B(3x4), C(2x4). Rank 14.
    We need 3x2x4: A(3x2), B(2x4), C(3x4).
    This is just a permutation of indices.
    Let's derive the factors for 3x2x4 from 2x3x4.
    
    2x3x4 factors (U 2x3, V 3x4, W 4x2):
    We need U' 3x2, V' 2x4, W' 4x3.
    This is not a simple transpose because the roles of A and B are different in the tensor product structure if we just swap.
    Actually, the tensor T_{ijk} for A(i,j)B(j,k)->C(i,k) is symmetric in the sense that if we swap A and B and transpose everything, the rank is the same.
    Rank(3x2x4) = Rank(2x3x4) = 14.
    
    Let's find the explicit factors for 3x2x4.
    Or better, let's use a known implementation of Laderman 2x3x4 and adapt it.
    
    The Laderman 14 factors for 2x3x4 (A 2x3, B 3x4, C 2x4) are often listed.
    Let's assume we can construct them.
    
    However, implementing Laderman correctly is error-prone.
    Let's try a different structural split.
    
    What if we use Strassen for 2x2x2 blocks, but pack them differently?
    Current: 88.
    
    Is there a rank 84 solution?
    3x4x8.
    
    Let's try to combine Strassen blocks.
    Split 3x4x8 into:
    2x4x8 (rank 56 using Strassen 2x2x2 blocks)
    1x4x8 (rank 32 using schoolbook)
    Total 88.
    
    Can we do better on 2x4x8?
    2x4x8 = 2x2x8 + 2x2x8.
    2x2x8 = 4 blocks of 2x2x2.
    Each 2x2x2 rank 7. Total 28.
    Two 2x2x8 blocks -> 56.
    
    What if we use a different algorithm for 2x2x8?
    Maybe 2x2x8 has rank < 28?
    2x2x8 schoolbook is 32.
    Strassen gives 28.
    Is 28 optimal for 2x2x8?
    Lower bound for 2x2xn?
    
    Let's try to search for a better 1x4x8 decomposition?
    1x4x8 is just dot products. Rank is 32. It's optimal because each output is independent and depends on 4 inputs, and the inputs are shared?
    No, C[i,k] = sum_j A[j] B[j,k].
    For fixed i, C[i,:] = A * B.
    This is a vector-matrix product.
    Rank of 1x4x8 tensor?
    It is the sum of 4 rank-1 tensors? No.
    C_k = sum_j A_j B_jk.
    This is bilinear.
    The tensor T_{j,k} = A_j B_k.
    This is rank 1 in the space of (j,k) pairs for fixed coefficients?
    No, the tensor is T_{j,k,s}.
    We need sum_s U_j,s V_{k,s} W_k,s? No.
    U is 1x4. V is 4x8. W is 8x1.
    C_k = sum_s U_j,s V_{j,k,s} ... wait.
    
    For 1x4x8:
    U shape (4, R).
    V shape (32, R).
    W shape (8, R).
    
    C_k = sum_j A_j B_jk.
    Decomposition: sum_s U_j,s V_{j,k,s} W_{k,s}?
    No, V is indexed by (j,k).
    C_k = sum_s (sum_j U_j,s V_{j,k,s}) W_{k,s}?
    No, the standard form is:
    C_{i,k} = sum_s U_{i,j,s} V_{j,k,s} W_{k,i,s}.
    For i=0 (only row):
    C_k = sum_s U_{0,j,s} V_{j,k,s} W_{k,0,s}.
    
    If we set W_{k,0,s} = 1 for all k,s?
    Then C_k = sum_s (sum_j U_{0,j,s} V_{j,k,s}).
    This looks like we are computing 4 matrix-vector products?
    
    Actually, the rank of 1xn x nxm is n*m?
    Yes, because the outputs are linearly independent combinations.
    So 1x4x8 is rank 32.
    
    So the bottleneck is the 2x4x8 part.
    Current rank 56.
    Schoolbook 64.
    Strassen 56.
    
    Can we do 2x4x8 in less than 56?
    2x4x8.
    Split 4 into 2+2.
    2x2x8 + 2x2x8.
    2x2x8 rank?
    
    Let's check if 2x2x8 can be done in < 28.
    2x2x8 = 2x2x2 + 2x2x2 + 2x2x2 + 2x2x2.
    Each 2x2x2 rank 7. Total 28.
    
    Is there a better way to handle 2x2x8?
    Maybe 2x2x4 blocks?
    2x2x4 rank?
    Schoolbook 16.
    Strassen? 2x2x4 is not square.
    
    Let's try to use the Laderman 2x3x4 idea on 2x2x4?
    2x2x4 is smaller.
    Rank of 2x2x4?
    Lower bound?
    It is known that rank(2x2xn) >= 2n?
    For n=2, rank 7.
    For n=4, rank?
    
    Actually, let's just stick to the 88 solution but try to verify it works, and perhaps try a randomized search for a rank 84 solution by perturbing the 88 solution?
    No, exact integer solutions are hard.
    
    Let's implement the 56+32=88 solution robustly.
    """
    return None

def _construct_88():
    n, m, p = 3, 4, 8
    rank = 88
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    def add_strassen_block(i_base, j_base, k_base, s_offset):
        factors = [
            (np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1]), np.array([1, 0, 0, 1])),
            (np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]), np.array([0, 0, 1, -1])),
            (np.array([1, 0, 0, 0]), np.array([0, 1, 0, -1]), np.array([0, 1, 0, 1])),
            (np.array([0, 0, 0, 1]), np.array([-1, 0, 1, 0]), np.array([1, 0, 1, 0])),
            (np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1]), np.array([-1, 1, 0, 0])),
            (np.array([-1, 0, 1, 0]), np.array([1, 1, 0, 0]), np.array([0, 0, 0, 1])),
            (np.array([0, 1, 0, -1]), np.array([0, 0, 1, 1]), np.array([1, 0, 0, 0]))
        ]
        
        for s_idx, (u, v, w) in enumerate(factors):
            s = s_offset + s_idx
            for i in range(2):
                for j in range(2):
                    u_idx = (i_base + i) * m + (j_base + j)
                    U[u_idx, s] += u[i * 2 + j]
            for j in range(2):
                for k in range(2):
                    v_idx = (j_base + j) * p + (k_base + k)
                    V[v_idx, s] += v[j * 2 + k]
            for k in range(2):
                for i in range(2):
                    w_idx = (k_base + k) * n + (i_base + i)
                    W[w_idx, s] += w[k * 2 + i]

    blocks = [
        (0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6),
        (0, 2, 0), (0, 2, 2), (0, 2, 4), (0, 2, 6)
    ]
    
    for idx, (ib, jb, kb) in enumerate(blocks):
        add_strassen_block(ib, jb, kb, idx * 7)
    
    s_offset = 56
    s_idx = 0
    for j in range(4):
        for k in range(8):
            s = s_offset + s_idx
            s_idx += 1
            U[2 * m + j, s] = 1.0
            V[j * p + k, s] = 1.0
            W[k * n + 2, s] = 1.0
            
    return {"U": U, "V": V, "W": W, "rank": rank}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Use the 88 rank solution
    sol = _construct_88()
    
    if _verify_solution(sol["U"], sol["V"], sol["W"], n, m, p):
        return {"U": sol["U"].tolist(), "V": sol["V"].tolist(), "W": sol["W"].tolist()}
    
    # Fallback
    sol = _schoolbook_solution(n, m, p)
    return {"U": sol["U"].tolist(), "V": sol["V"].tolist(), "W": sol["W"].tolist()}
