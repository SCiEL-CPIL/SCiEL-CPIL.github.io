import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 3, 3, 4
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, c] = 1
                V_school[j * p + k, c] = 1
                W_school[k * n + i, c] = 1
                c += 1
    
    start_time = time.time()
    deadline = 110 # 10s margin for 120s limit
    
    # Strategy 1: Find a low-rank decomposition for 3x3x3, then pad to 3x3x4
    # The 4th column of the middle matrix (V) corresponds to k=3.
    # The terms for k=3 are A[i,j] * B[j,3] * W[3,n+i].
    # We can add 3 terms:
    # Term r: U[:,r] = e_i (for i in 0..2), V[:,r] = e_j (for j in 0..2)?? No.
    # The tensor for the 4th column is T[i, j, 3] = delta_{i, i'}? No.
    # T[i, j, k] = 1 if (i,j,k) is valid.
    # For the 4th column (k=3), the tensor is T[i, j, 3] = 1 for all i,j?
    # No, T[i, j, k] is the coefficient of A[i,j] B[j,k] in C[k,i].
    # So for fixed k=3, the tensor is a 3x3 matrix of ones?
    # C[3, i] = sum_j A[i,j] B[j,3].
    # This is a matrix-vector product.
    # The tensor for this part is $T_{sub}[i, j, 3] = \delta_{i, i'}$? No.
    # The decomposition term for a specific (i, j, k) is U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1.
    # So for the 4th column (k=3), we need to decompose the tensor $T_{col3}$ where $T_{col3}[i, j, 3] = 1$?
    # Wait, the tensor $T$ has indices $i \in [0,3), j \in [0,3), k \in [0,4)$.
    # $T[i,j,k] = 1$ if the term $A[i,j]B[j,k]$ contributes to $C[k,i]$.
    # Yes, it always contributes.
    # So for $k=3$, $T[i,j,3] = 1$ for all $i,j$.
    # This is a $3 \times 3 \times 1$ tensor.
    # Its rank is 3 (since it's a rank-1 matrix of ones? No, it's a vector of 1s in the k-dimension? No, k is fixed).
    # The tensor is $1_{3 \times 3}$ in the i-j plane, for k=3.
    # The rank of a $3 \times 3 \times 1$ tensor which is a matrix of ones is 3?
    # $M_{ij} = 1$. $M = u v^T$ where $u=[1,1,1]^T, v=[1,1,1]^T$.
    # So it can be decomposed into 3 terms?
    # Term 1: i=0, j=0? No.
    # If we use the standard basis:
    # $1_{3 \times 3} = \sum_{r=0}^2 e_r e_r^T$? No, that's identity.
    # $1_{3 \times 3} = \sum_{r=0}^2 u_r v_r^T$?
    # We can use 3 terms:
    # $r=0: u=[1,0,0], v=[1,0,0]$ -> $e_0 e_0^T$
    # $r=1: u=[0,1,0], v=[0,1,0]$ -> $e_1 e_1^T$
    # $r=2: u=[0,0,1], v=[0,0,1]$ -> $e_2 e_2^T$
    # Sum is Identity. Not ones.
    # We need $1_{3 \times 3}$.
    # Rank of all-ones matrix is 1.
    # So we can do it in 1 term?
    # $U[i, r] = 1$ for all i? $V[j, r] = 1$ for all j?
    # Then $U[i,r] V[j,r] = 1$.
    # And $W[k,r]$ must be 1 for $k=3$?
    # But $W$ has index $k \in [0,4)$.
    # If we set $W[3, r] = 1$ and $W[k', r] = 0$ for $k' \neq 3$.
    # Then the term is $1 \cdot 1 \cdot \delta_{k,3} = \delta_{k,3}$.
    # This matches $T[i,j,k] = \delta_{k,3}$ for the 4th column?
    # Yes, because $T[i,j,3] = 1$ and $T[i,j,k \neq 3] = 0$ for this part.
    # So the 4th column can be handled with **1 term** if we allow $U, V$ to have all 1s?
    # But $U$ and $V$ are shared with the other terms.
    # If we add this term to the $3 \times 3 \times 3$ decomposition:
    # The $3 \times 3 \times 3$ decomposition handles $k \in \{0,1,2\}$.
    # The new term handles $k=3$.
    # Does the new term interfere with $k \in \{0,1,2\}$?
    # $W[k, r_{new}] = 0$ for $k \neq 3$. So no interference.
    # So we just add 1 term!
    # Total rank = Rank($3 \times 3 \times 3$) + 1.
    
    # If Rank($3 \times 3 \times 3$) = 23, Total = 24.
    # If Rank($3 \times 3 \times 3$) = 24, Total = 25.
    # If Rank($3 \times 3 \times 3$) = 27, Total = 28.
    
    # Let's try to find a $3 \times 3 \times 3$ decomposition.
    
    best_decomp = None
    best_rank = rank_school
    
    # Try 3x3x3 ranks
    for target_rank_333 in [23, 24, 25, 26, 27]:
        if time.time() > deadline - 5: # Leave 5s for construction
            break
        decomp_333 = _find_decomp(3, 3, 3, target_rank_333, deadline - 5)
        if decomp_333 is not None:
            # Construct 3x3x4 decomp
            U_333 = np.array(decomp_333['U'])
            V_333 = np.array(decomp_333['V'])
            W_333 = np.array(decomp_333['W'])
            
            # Add 1 term for k=3
            # U_new: (9, 1) all 1s
            # V_new: (12, 1) -> indices 0..8 for k=0,1,2 are 0? No.
            # V is (m*p, rank) = (3*4, rank) = (12, rank).
            # The term for k=3 should have V[j, 3] = 1 for all j?
            # Wait, V index is $j \cdot p + k$.
            # For $k=3$, indices are $j \cdot 4 + 3$.
            # We want $V[j \cdot 4 + 3, r_{new}] = 1$ for all j?
            # And $U[i \cdot 3 + j, r_{new}] = 1$ for all i,j?
            # And $W[3 \cdot 3 + i, r_{new}] = 1$ for all i?
            
            # Let's check the product:
            # $U[i,j] V[j,3] W[3,i] = 1 \cdot 1 \cdot 1 = 1$.
            # This matches $T[i,j,3] = 1$.
            
            U_new = np.zeros((9, target_rank_333 + 1))
            V_new = np.zeros((12, target_rank_333 + 1))
            W_new = np.zeros((12, target_rank_333 + 1)) # p*n = 4*3 = 12
            
            U_new[:, :target_rank_333] = U_333
            V_new[:9, :target_rank_333] = V_333 # V_333 is (9, rank)
            # V_333 corresponds to k=0,1,2. Indices j*3+k.
            # In V_new, indices are j*4+k.
            # So we need to map V_333[j*3+k] to V_new[j*4+k].
            # V_333 is (9, R). Rows 0..8 correspond to (j,k) in 3x3.
            # V_new rows 0..11 correspond to (j,k) in 3x4.
            # Rows 0,1,2 (j=0, k=0,1,2) -> V_333 rows 0,1,2.
            # Row 3 (j=0, k=3) -> 0.
            # Rows 4,5,6 (j=1, k=0,1,2) -> V_333 rows 3,4,5.
            # Row 7 (j=1, k=3) -> 0.
            # Rows 8,9,10 (j=2, k=0,1,2) -> V_333 rows 6,7,8.
            # Row 11 (j=2, k=3) -> 0.
            
            for j in range(3):
                for k in range(3):
                    V_new[j*4+k, :target_rank_333] = V_333[j*3+k, :target_rank_333]
            
            W_new[:9, :target_rank_333] = W_333 # W_333 is (9, rank) for k=0,1,2?
            # W index is k*n+i.
            # W_333 rows 0..8 correspond to k=0,1,2.
            # W_new rows 0..11 correspond to k=0,1,2,3.
            # Rows 0..2 (k=0) -> W_333 0..2.
            # Rows 3..5 (k=1) -> W_333 3..5.
            # Rows 6..8 (k=2) -> W_333 6..8.
            # Rows 9..11 (k=3) -> 0.
            
            for k in range(3):
                for i in range(3):
                    W_new[k*3+i, :target_rank_333] = W_333[k*3+i, :target_rank_333]
            
            # Add the new term
            r_new = target_rank_333
            U_new[:, r_new] = 1.0
            V_new[3, r_new] = 1.0 # j=0, k=3
            V_new[7, r_new] = 1.0 # j=1, k=3
            V_new[11, r_new] = 1.0 # j=2, k=3
            W_new[9, r_new] = 1.0 # k=3, i=0
            W_new[10, r_new] = 1.0 # k=3, i=1
            W_new[11, r_new] = 1.0 # k=3, i=2
            
            # Verify
            T_recon = np.einsum('ir,jr,kr->ijk', U_new, V_new, W_new)
            T_target = np.zeros((3, 3, 4))
            for i in range(3):
                for j in range(3):
                    for k in range(4):
                        T_target[i,j,k] = 1.0
            
            if np.allclose(T_recon, T_target, atol=1e-5):
                return {
                    "U": U_new.tolist(),
                    "V": V_new.tolist(),
                    "W": W_new.tolist()
                }
                
    # Fallback: Direct search for 3x3x4
    for target_rank in [24, 25, 26]:
        if time.time() > deadline:
            break
        decomp = _find_decomp(3, 3, 4, target_rank, deadline)
        if decomp is not None:
            return decomp
            
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

def _find_decomp(n, m, p, rank, deadline):
    # Tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
        
    T_flat = T.flatten()
    
    def objective(x):
        U = x[:rank * n * m].reshape(rank, n * m)
        V = x[rank * n * m : rank * n * m + rank * m * p].reshape(rank, m * p)
        W = x[rank * n * m + rank * m * p :].reshape(rank, p * n)
        
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        error = np.linalg.norm(T_recon.flatten() - T_flat)
        return error ** 2
        
    # Initial guess
    np.random.seed(42)
    best_error = float('inf')
    best_x = None
    
    # Try a few restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        x0 = np.random.randn(rank * (n * m + m * p + p * n))
        
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        
        if res.fun < best_error:
            best_error = res.fun
            best_x = res.x
            
        if best_error < 1e-8:
            break
            
    if best_error < 1e-6:
        x = best_x
        U = x[:rank * n * m].reshape(rank, n * m)
        V = x[rank * n * m : rank * n * m + rank * m * p].reshape(rank, m * p)
        W = x[rank * n * m + rank * m * p :].reshape(rank, p * n)
        
        # Round to nearest 0.5
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Verify
        T_recon = np.einsum('ir,jr,kr->ijk', U_int, V_int, W_int)
        if np.allclose(T_recon, T, atol=1e-5):
            return {
                "U": U_int.tolist(),
                "V": V_int.tolist(),
                "W": W_int.tolist()
            }
            
    return None
