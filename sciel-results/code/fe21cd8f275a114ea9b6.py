import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 5, 6]

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_target_tensor(n, m, p):
    """Build the dense target tensor for matrix multiplication."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    return T

def _strassen_222():
    """
    Return U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U, V, W are 2x7 matrices.
    T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    Note: Standard Strassen computes C = AB.
    The tensor T is defined such that C[i, k] = sum_j T[i, j, k] A[i, j] B[j, k].
    Wait, the problem definition is:
    T_calc[a,b,d] = sum_r U[a,r] V[b,r] W[d,r]
    where a = i*m+j, b = j*p+k, d = k*n+i.
    This corresponds to the coefficient of A[i,j] B[j,k] in C[i,k].
    So T[i, j, k] = 1.
    
    Strassen computes C = A B.
    C[i, k] = sum_j A[i, j] B[j, k] * (contribution from j).
    Actually, Strassen expresses the multiplication in terms of 7 scalar multiplications.
    The tensor decomposition rank is the number of scalar multiplications.
    
    Let's derive the U, V, W for the 2x2x2 case.
    We want T[i, j, k] = 1 for all i, j, k in {0, 1}.
    Wait, NO.
    The matrix multiplication tensor T has T[i, j, k] = 1 if the term A[i, j] B[j, k] contributes to C[i, k].
    In standard matrix multiplication, EVERY term A[i, j] B[j, k] contributes to C[i, k] with coefficient 1.
    So T[i, j, k] = 1 for all i, j, k.
    
    We need to decompose the tensor of all ones (size 2x2x2) into rank-1 tensors?
    No, that's not right.
    The tensor of matrix multiplication is NOT the all-ones tensor.
    The tensor $T_{ijk}$ is 1 if $A_{ij} B_{jk}$ appears in $C_{ik}$.
    For standard multiplication, $C_{ik} = \sum_j A_{ij} B_{jk}$.
    So yes, $T_{ijk} = 1$ for all $i, j, k$.
    
    Wait, if $T_{ijk} = 1$ for all $i, j, k$, then the tensor is the outer product of three all-ones vectors.
    Rank 1?
    No, that's not right.
    The tensor is a map from $A \otimes B \to C$.
    $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
    The structure is fixed by the indices.
    $A$ is indexed by $(i, j)$. $B$ is indexed by $(j, k)$. $C$ is indexed by $(i, k)$.
    The term $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    So the tensor $T$ has components $T_{(i,j), (j,k), (i,k)}$.
    In the flattened vector space basis:
    $U$ has rows indexed by $(i, j)$.
    $V$ has rows indexed by $(j, k)$.
    $W$ has rows indexed by $(i, k)$.
    
    For a specific rank-1 component $r$:
    $U_{(i,j), r} V_{(j,k), r} W_{(i,k), r}$.
    Summing over $r$, we must get 1 for all valid $(i, j, k)$ triplets that form a valid matrix multiplication path.
    The valid triplets are all $(i, j, k)$ such that $j$ is the shared index.
    In the block decomposition, we handle sub-blocks.
    Inside a $2 \times 2 \times 2$ block, the indices are local $i \in \{0, 1\}, j \in \{0, 1\}, k \in \{0, 1\}$.
    We need $\sum_r U[i, r] V[j, r] W[k, r] = 1$ for all $i, j, k$.
    
    This is exactly the problem of decomposing the $2 \times 2 \times 2$ matrix multiplication tensor.
    Strassen's algorithm provides a rank-7 decomposition.
    
    Let's verify the Strassen coefficients map to this.
    Strassen computes $C = AB$.
    $C_{00} = M_1 + M_4 - M_5 + M_7$
    $C_{01} = M_3 + M_5$
    $C_{10} = M_2 + M_4$
    $C_{11} = M_1 - M_2 + M_3 + M_6$
    
    Where:
    $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$
    $M_2 = (A_{10}+A_{11})B_{00}$
    $M_3 = A_{00}(B_{01}-B_{11})$
    $M_4 = A_{11}(B_{10}-B_{00})$
    $M_5 = (A_{00}+A_{01})B_{11}$
    $M_6 = (A_{10}-A_{00})(B_{00}+B_{01})$
    $M_7 = (A_{01}-A_{11})(B_{10}+B_{11})$
    
    We can view each $M_r$ as a product of a linear combination of A's and a linear combination of B's.
    $M_r = (\sum_{i,j} u_{ij}^{(r)} A_{ij}) (\sum_{j,k} v_{jk}^{(r)} B_{jk})$
    Then $C_{ik} = \sum_r w_{ik}^{(r)} M_r$.
    
    So:
    $U[i, j, r] = u_{ij}^{(r)}$
    $V[j, k, r] = v_{jk}^{(r)}$
    $W[i, k, r] = w_{ik}^{(r)}$
    
    Let's extract the coefficients for each $M_r$.
    
    $M_1$:
    $A$ part: $A_{00} + A_{11}$. $u^{(1)}_{00}=1, u^{(1)}_{11}=1, u^{(1)}_{01}=0, u^{(1)}_{10}=0$.
    $B$ part: $B_{00} + B_{11}$. $v^{(1)}_{00}=1, v^{(1)}_{11}=1, v^{(1)}_{01}=0, v^{(1)}_{10}=0$.
    $C$ part:
    $C_{00} += 1 \cdot M_1 \implies w^{(1)}_{00} = 1$
    $C_{11} += 1 \cdot M_1 \implies w^{(1)}_{11} = 1$
    $C_{01}, C_{10} += 0$.
    
    $M_2$:
    $A$ part: $A_{10} + A_{11}$. $u^{(2)}_{10}=1, u^{(2)}_{11}=1$.
    $B$ part: $B_{00}$. $v^{(2)}_{00}=1$.
    $C$ part:
    $C_{10} += 1 \cdot M_2 \implies w^{(2)}_{10} = 1$.
    
    $M_3$:
    $A$ part: $A_{00}$. $u^{(3)}_{00}=1$.
    $B$ part: $B_{01} - B_{11}$. $v^{(3)}_{01}=1, v^{(3)}_{11}=-1$.
    $C$ part:
    $C_{01} += 1 \cdot M_3 \implies w^{(3)}_{01} = 1$.
    
    $M_4$:
    $A$ part: $A_{11}$. $u^{(4)}_{11}=1$.
    $B$ part: $B_{10} - B_{00}$. $v^{(4)}_{10}=1, v^{(4)}_{00}=-1$.
    $C$ part:
    $C_{00} += 1 \cdot M_4 \implies w^{(4)}_{00} = 1$.
    $C_{10} += 1 \cdot M_4 \implies w^{(4)}_{10} = 1$.
    
    $M_5$:
    $A$ part: $A_{00} + A_{01}$. $u^{(5)}_{00}=1, u^{(5)}_{01}=1$.
    $B$ part: $B_{11}$. $v^{(5)}_{11}=1$.
    $C$ part:
    $C_{00} += -1 \cdot M_5 \implies w^{(5)}_{00} = -1$.
    $C_{01} += 1 \cdot M_5 \implies w^{(5)}_{01} = 1$.
    
    $M_6$:
    $A$ part: $A_{10} - A_{00}$. $u^{(6)}_{10}=1, u^{(6)}_{00}=-1$.
    $B$ part: $B_{00} + B_{01}$. $v^{(6)}_{00}=1, v^{(6)}_{01}=1$.
    $C$ part:
    $C_{11} += 1 \cdot M_6 \implies w^{(6)}_{11} = 1$.
    
    $M_7$:
    $A$ part: $A_{01} - A_{11}$. $u^{(7)}_{01}=1, u^{(7)}_{11}=-1$.
    $B$ part: $B_{10} + B_{11}$. $v^{(7)}_{10}=1, v^{(7)}_{11}=1$.
    $C$ part:
    $C_{00} += 1 \cdot M_7 \implies w^{(7)}_{00} = 1$.
    
    We need to return U, V, W of shape (2, 7) for the local indices.
    Local $A$ indices: $i \in \{0, 1\}, j \in \{0, 1\}$. Flattened: $00 \to 0, 01 \to 1, 10 \to 2, 11 \to 3$.
    Local $B$ indices: $j \in \{0, 1\}, k \in \{0, 1\}$. Flattened: $00 \to 0, 01 \to 1, 10 \to 2, 11 \to 3$.
    Local $C$ indices: $i \in \{0, 1\}, k \in \{0, 1\}$. Flattened: $00 \to 0, 01 \to 1, 10 \to 2, 11 \to 3$.
    
    Let's construct the matrices.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _block_decomposition_356():
    n, m, p = 3, 5, 6
    nm, mp, pn = n * m, m * p, p * n
    
    # Splits
    # n=3: [2, 1]
    # m=5: [2, 2, 1]
    # p=6: [2, 2, 2]
    
    n_splits = [0, 2, 3]
    m_splits = [0, 2, 4, 5]
    p_splits = [0, 2, 4, 6]
    
    total_rank = 0
    # Pre-allocate with max possible rank? Or just list of blocks.
    blocks = []
    
    for i_idx in range(2):
        n_start = n_splits[i_idx]
        n_end = n_splits[i_idx+1]
        n_size = n_end - n_start
        
        for j_idx in range(3):
            m_start = m_splits[j_idx]
            m_end = m_splits[j_idx+1]
            m_size = m_end - m_start
            
            for k_idx in range(3):
                p_start = p_splits[k_idx]
                p_end = p_splits[k_idx+1]
                p_size = p_end - p_start
                
                # Determine rank and decomposition type
                if n_size == 2 and m_size == 2 and p_size == 2:
                    rank = 7
                    decomp = 'strassen'
                else:
                    # Rank 1 case
                    # U[i, r] = 1, V[j, r] = 1, W[k, r] = 1
                    rank = 1
                    decomp = 'rank1'
                
                blocks.append({
                    'n_start': n_start, 'n_end': n_end,
                    'm_start': m_start, 'm_end': m_end,
                    'p_start': p_start, 'p_end': p_end,
                    'rank': rank,
                    'decomp': decomp
                })
                total_rank += rank
                
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    col_offset = 0
    for block in blocks:
        n_s, n_e = block['n_start'], block['n_end']
        m_s, m_e = block['m_start'], block['m_end']
        p_s, p_e = block['p_start'], block['p_end']
        rank = block['rank']
        
        if block['decomp'] == 'strassen':
            U_loc, V_loc, W_loc = _strassen_222()
            
            # Map local indices to global
            # Local i, j, k are 0..1
            # Global i = n_s + i_loc
            # Global j = m_s + j_loc
            # Global k = p_s + k_loc
            
            # U global row index: i * m + j
            for r in range(rank):
                for i_loc in range(2):
                    for j_loc in range(2):
                        # U_loc is flattened i*m+j?
                        # In _strassen_222, U is 4x7. Index 0:00, 1:01, 2:10, 3:11.
                        # This matches i*m+j for m=2.
                        u_val = U_loc[i_loc * 2 + j_loc, r]
                        if u_val != 0:
                            g_i = n_s + i_loc
                            g_j = m_s + j_loc
                            U[g_i * m + g_j, col_offset + r] = u_val
                            
                for j_loc in range(2):
                    for k_loc in range(2):
                        v_val = V_loc[j_loc * 2 + k_loc, r]
                        if v_val != 0:
                            g_j = m_s + j_loc
                            g_k = p_s + k_loc
                            V[g_j * p + g_k, col_offset + r] = v_val
                            
                for i_loc in range(2):
                    for k_loc in range(2):
                        w_val = W_loc[i_loc * 2 + k_loc, r]
                        if w_val != 0:
                            g_i = n_s + i_loc
                            g_k = p_s + k_loc
                            W[g_k * n + g_i, col_offset + r] = w_val
                            
        else: # rank1
            # U[i, r] = 1, V[j, r] = 1, W[k, r] = 1
            for i_loc in range(n_e - n_s):
                for j_loc in range(m_e - m_s):
                    g_i = n_s + i_loc
                    g_j = m_s + j_loc
                    U[g_i * m + g_j, col_offset] = 1.0
                    
            for j_loc in range(m_e - m_s):
                for k_loc in range(p_e - p_s):
                    g_j = m_s + j_loc
                    g_k = p_s + k_loc
                    V[g_j * p + g_k, col_offset] = 1.0
                    
            for k_loc in range(p_e - p_s):
                for i_loc in range(n_e - n_s):
                    g_k = p_s + k_loc
                    g_i = n_s + i_loc
                    W[g_k * n + g_i, col_offset] = 1.0
                    
        col_offset += rank
        
    return U, V, W, total_rank

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Try block decomposition first
    try:
        U, V, W, rank = _block_decomposition_356()
        
        # Verify
        T_target = _build_target_tensor(n, m, p)
        T_calc = np.einsum('ar,br,dr->abd', U, V, W)
        
        if np.allclose(T_calc, T_target, atol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception as e:
        pass
        
    # Fallback to schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
