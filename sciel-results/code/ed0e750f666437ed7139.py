import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 28 Strassen decomposition
    rank_28 = 28
    U28 = np.zeros((n * m, rank_28), dtype=int)
    V28 = np.zeros((m * p, rank_28), dtype=int)
    W28 = np.zeros((p * n, rank_28), dtype=int)
    
    def add_term(u_vec, v_vec, w_vec, col_idx):
        for idx, coeff in u_vec:
            U28[idx, col_idx] += coeff
        for idx, coeff in v_vec:
            V28[idx, col_idx] += coeff
        for idx, coeff in w_vec:
            W28[idx, col_idx] += coeff
            
    def strassen_terms():
        terms = []
        u = {0: 1, 3: 1}; v = {0: 1, 3: 1}; w = {0: 1, 3: 1}
        terms.append((u, v, w))
        u = {2: 1, 3: 1}; v = {0: 1}; w = {2: 1, 3: -1}
        terms.append((u, v, w))
        u = {0: 1}; v = {1: 1, 3: -1}; w = {1: 1, 3: 1}
        terms.append((u, v, w))
        u = {3: 1}; v = {2: 1, 0: -1}; w = {0: 1, 2: 1}
        terms.append((u, v, w))
        u = {0: 1, 1: 1}; v = {3: 1}; w = {0: -1, 1: 1}
        terms.append((u, v, w))
        u = {2: 1, 0: -1}; v = {0: 1, 1: 1}; w = {3: 1}
        terms.append((u, v, w))
        u = {1: 1, 3: -1}; v = {2: 1, 3: 1}; w = {0: 1}
        terms.append((u, v, w))
        return terms

    base_terms = strassen_terms()
    
    def apply_strassen(a_offset, b_row_offset, b_col_offset, c_col_offset, start_col):
        for t_idx, (u_loc, v_loc, w_loc) in enumerate(base_terms):
            col = start_col + t_idx
            u_global = {}
            for loc_idx, coeff in u_loc.items():
                r = loc_idx // 2; c = loc_idx % 2
                g_r = r; g_c = a_offset * 2 + c
                g_idx = g_r * 4 + g_c
                u_global[g_idx] = u_global.get(g_idx, 0) + coeff
            v_global = {}
            for loc_idx, coeff in v_loc.items():
                r = loc_idx // 2; c = loc_idx % 2
                g_r = b_row_offset * 2 + r; g_c = b_col_offset * 2 + c
                g_idx = g_r * 4 + g_c
                v_global[g_idx] = v_global.get(g_idx, 0) + coeff
            w_global = {}
            for loc_idx, coeff in w_loc.items():
                r = loc_idx // 2; c = loc_idx % 2
                g_r = r; g_c = c_col_offset * 2 + c
                g_idx = g_c * 2 + g_r
                w_global[g_idx] = w_global.get(g_idx, 0) + coeff
            add_term(list(u_global.items()), list(v_global.items()), list(w_global.items()), col)

    apply_strassen(0, 0, 0, 0, 0)
    apply_strassen(1, 1, 0, 0, 7)
    apply_strassen(0, 0, 1, 1, 14)
    apply_strassen(1, 1, 1, 1, 21)
    
    def verify(U, V, W):
        np.random.seed(42)
        for _ in range(5):
            A = np.random.randint(-10, 10, (n, m))
            B = np.random.randint(-10, 10, (m, p))
            C = A @ B
            Cf = np.zeros((n, p))
            for ii in range(n):
                for kk in range(p):
                    acc = 0.0
                    for cc in range(U.shape[1]):
                        for jj in range(m):
                            acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                    Cf[ii, kk] = acc
            if not np.allclose(C, Cf, atol=1e-9):
                return False
        return True

    start_time = time.time()
    deadline = 140.0
    
    target_rank = 27
    
    # Search for rank 27
    if time.time() < deadline - 5:
        np.random.seed(123)
        
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # Reshape for gradient computation
            # U: (n, m, R), V: (m, p, R), W: (p, n, R)
            U_mat = U.reshape(n, m, target_rank)
            V_mat = V.reshape(m, p, target_rank)
            W_mat = W.reshape(p, n, target_rank)
            
            lr = 0.01
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute P_r = U_r V_r
                # P: (n, p, R)
                P = np.einsum('ijr,jkr->ikr', U_mat, V_mat)
                
                # Compute C = sum_r P_r * W_r^T (Hadamard)
                # W_r^T has shape (n, p) with element W_ki at (i, k)
                # W_mat is (p, n, R). W_mat[k, i, r] is W_ki.
                # We need W_transposed_mat[i, k, r] = W_mat[k, i, r]
                W_T = np.transpose(W_mat, (1, 0, 2)) # (n, p, R)
                
                C = np.sum(P * W_T, axis=2) # (n, p)
                
                # Target C is not fixed, it depends on A, B.
                # We need to minimize the error over ALL A, B.
                # This is equivalent to minimizing the tensor norm.
                # The tensor T is the identity map for matrix mult.
                # T[i, j, k] = 1 if ...?
                # Actually, we can just pick random A, B and minimize error for them?
                # No, we need to decompose the *operator*.
                # The loss function should be the Frobenius norm of the tensor difference.
                # The tensor of matrix multiplication is $\sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}$?
                # No.
                # The tensor is $T \in \mathbb{R}^{n \times m \times m \times p \times p \times n}$? No.
                # The decomposition is $C_{ik} = \sum_r \sum_j U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}$.
                # This must hold for ALL A, B.
                # This implies that the tensor $\mathcal{M}$ defined by $\mathcal{M}_{ij, jk, ki} = 1$ (if indices match) must be decomposed.
                # Actually, the condition is:
                # $\sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = \delta_{jk}$? No.
                # $C_{ik} = \sum_j A_{ij} B_{jk}$.
                # So $\sum_r \sum_j U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} A_{ij} B_{jk} = \sum_j A_{ij} B_{jk}$.
                # This must hold for all A, B.
                # This implies $\sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = \delta_{jj}$? No.
                # The coefficient of $A_{ij} B_{jk}$ in the LHS is $\sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}$.
                # The coefficient in the RHS is 1 if $j$ is the summation index?
                # Wait. $C_{ik} = \sum_{j=1}^m A_{ij} B_{jk}$.
                # So for a fixed $i, k$, the term $A_{ij} B_{jk}$ appears with coefficient 1.
                # So we require $\sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = 1$ for all $i, j, k$.
                
                # So the target tensor $T$ has $T[i, j, k] = 1$ for all $i, j, k$.
                # Wait, is that right?
                # $C_{ik} = \sum_j A_{ij} B_{jk}$.
                # LHS: $\sum_r \sum_j U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} A_{ij} B_{jk}$.
                # We want this to equal $\sum_j A_{ij} B_{jk}$.
                # So we need $\sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = 1$ for all $i, j, k$.
                
                # So the target tensor is just the all-ones tensor?
                # Let's check.
                # If $U, V, W$ are constant 1, then sum is $R$.
                # We need sum to be 1.
                # So we are decomposing the all-ones tensor of size $n \times m \times p$?
                # No, the indices are $i$ (row of A), $j$ (col of A / row of B), $k$ (col of B).
                # And $W$ has indices $k, i$.
                # So the tensor is $T[i, j, k] = 1$.
                
                # If this is true, then the problem is to decompose the all-ones tensor $\mathbf{1}_{n \times m \times p}$ into rank $R$ terms $U^{(r)} \otimes V^{(r)} \otimes W^{(r)}$?
                # No, the structure is fixed: $U$ is $n \times m$, $V$ is $m \times p$, $W$ is $p \times n$.
                # And the contraction is $U_{ij} V_{jk} W_{ki}$.
                # This is a specific type of tensor decomposition.
                
                # Let's verify this hypothesis.
                # If $T[i, j, k] = 1$, then we need $\sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)} = 1$.
                
                # Let's use this for gradient descent.
                # Target: $T_{ijk} = 1$.
                # Current: $S_{ijk} = \sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}$.
                # Error: $E_{ijk} = S_{ijk} - 1$.
                # Loss: $\sum_{i,j,k} E_{ijk}^2$.
                
                # Gradients:
                # $\frac{\partial L}{\partial U_{ij}^{(r)}} = 2 \sum_{k} E_{ijk} V_{jk}^{(r)} W_{ki}^{(r)}$.
                # $\frac{\partial L}{\partial V_{jk}^{(r)}} = 2 \sum_{i} E_{ijk} U_{ij}^{(r)} W_{ki}^{(r)}$.
                # $\frac{\partial L}{\partial W_{ki}^{(r)}} = 2 \sum_{j} E_{ijk} U_{ij}^{(r)} V_{jk}^{(r)}$.
                
                # Compute E
                # S = einsum('ijr,jkr,ki r -> ijk', U_mat, V_mat, W_mat)
                # Note: W_mat is (p, n, R). W_mat[k, i, r].
                S = np.einsum('ijr,jkr,ki r->ijk', U_mat, V_mat, W_mat)
                E = S - 1.0
                
                # Gradients
                dU = 2 * np.einsum('ijk,jkr,ki r->ijr', E, V_mat, W_mat)
                dV = 2 * np.einsum('ijk,ijr,ki r->jkr', E, U_mat, W_mat)
                dW = 2 * np.einsum('ijk,ijr,jkr->kir', E, U_mat, V_mat)
                
                # Update
                U_mat -= lr * dU
                V_mat -= lr * dV
                W_mat -= lr * dW
                
                # Flatten back
                U = U_mat.reshape(n * m, target_rank)
                V = V_mat.reshape(m * p, target_rank)
                W = W_mat.reshape(p * n, target_rank)
                
                # Optional: Snap to half-integers occasionally?
                # Or just at the end.
            
            # After GD, snap to half-integers
            U_snapped = np.round(U * 2) / 2
            V_snapped = np.round(V * 2) / 2
            W_snapped = np.round(W * 2) / 2
            
            if verify(U_snapped, V_snapped, W_snapped):
                return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
            
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
# EVOLVE-BLOCK-END
