"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from functools import reduce
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _build_3x3x8_mps_search(allow_half_int=True):
    """
    Search for a rank-56 decomposition of the 3x3x8 tensor using MPS bond structure
    and greedy tensor train truncation to integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 45.0
    
    n, m, p = 3, 3, 8
    rank = 56
    
    def tensor_product(arrs, out_dim=1, ax=0):
        """Compute the outer product of a list of arrays into a flat vector of shape (prod(out_dims), rank)."""
        res = arrs[0].astype(float)
        for a in arrs[1:]:
            res = np.multiply.outer(res, a).reshape(-1, rank)
        return res

    def truncate_to_rational(T, tol=1e-5):
        """
        Project entries of a 2D numpy array to nearest integer or half-integer,
        then verify the projection error is within tolerance.
        """
        T = np.asarray(T, dtype=float)
        if allow_half_int:
            candidates = np.arange(0.5, np.ceil(np.max(np.abs(T))) + 0.5, 0.5)
            # We also include negative values if T has them
            min_T = np.min(T)
            max_T = np.max(T)
            if min_T < 0:
                 candidates = np.concatenate([np.arange(min_T - 0.5, min_T + 0.5, 0.5), candidates])
            else:
                 candidates = np.concatenate([np.array([0.0]), candidates])
            
            dists = np.abs(T[:, None] - candidates[None, :])
            closest_indices = np.argmin(dists, axis=1)
            proj = candidates[closest_indices]
        else:
            proj = np.round(T)
        
        # Check if projection is exact enough
        if np.max(np.abs(T - proj)) <= tol:
            return proj
        return None

    def random_init(seed_val):
        # Initialize cores using structured random generation to enforce MPS rank structure
        # T0: 1 x 1 x rank
        # T1: 3 x 1 x rank
        # T2: 3 x 1 x rank
        # T3: 8 x 1 x rank
        # We define A_i (1,3), B_j (1,3), C_k (1,8), D (8,1) matrices such that
        # T = A ⊗ B ⊗ C ⊗ D with a shared rank dimension.
        # To keep it simple, we generate random integer 8x8 matrix M.
        
        g = np.random.default_rng(seed_val)
        
        # M is 8x8. Rank 8. 
        # U: 9x8, V: 3x8, W: 8x8
        # Then total U,V,W are 56.
        # We need to find U, V, W such that they form an MPS.
        # We assume U = M^T (8x8), V = N (3x8), W = O (8x8)
        # And T_{ijk, alpha} = U_{i, alpha} * V_{j, alpha} * W_{k, alpha}
        # Since the MPS is not strictly sequential in 1D (it's a 3D grid flattened to 1D sequence of length 3), the "bond" is the full rank vector for each site.
        # This is just the definition of rank 56.
        
        # So we generate M, N, O randomly with small integers.
        M = g.integers(-2, 3, size=(8, 8))
        N = g.integers(-2, 3, size=(3, 8))
        O = g.integers(-2, 3, size=(8, 8))
        
        return M, N, O

    def refine(M, N, O):
        # Refine M, N, O to minimize error.
        # We will use gradient descent on a smoothed objective with projection.
        M_c = M.astype(float)
        N_c = N.astype(float)
        O_c = O.astype(float)
        
        # Objective: minimize norm of (M^T N^T O^T) - T
        # We will project to lattice every step.
        
        for _ in range(1000):
            if time.time() > deadline:
                return M, N, O
                
            # Compute current T
            T_approx = tensor_product([M_c.T, N_c.T, O_c.T])
            
            # Error
            E = T_approx - T_target
            
            # Compute gradients
            # E_{(3*i + j)*8 + k, a}
            # d/da M_{a, i} = sum_{j,k} E_{..., a} * N_{j, a} * O_{k, a}
            # This is a tensor contraction.
            # We reshape E to (3, 3, 8, 56)
            E_r = E.reshape(3, 3, 8, 56)
            
            # M_grad[i, a] = sum_{j,k} E_r[i,j,k,a] * N[j,a] * O[k,a]
            # This can be done with np.einsum or loops. For speed, we do a simple loop.
            M_grad = np.zeros((8, 8))
            N_grad = np.zeros((3, 8))
            O_grad = np.zeros((8, 8))
            
            for a in range(56):
                # Slice E_r[:, :, :, a]
                e_a = E_r[:, :, :, a]
                n_a = N_c[:, a]
                o_a = O_c[:, a]
                
                M_grad[:, a] = np.einsum('ijk,k,ji->i', e_a, o_a, n_a) # wait.
                # Let's check indices carefully.
                # e_a is (i,j,k). n_a is (j), o_a is (k).
                # dE/dM_{ia} = sum_{jk} E_{ijk} * V_{ja} * W_{ka}
                # V_{ja} = N_c[j, a]. W_{ka} = O_c[k, a].
                # So sum_{jk} e_a[i,j,k] * n_a[j] * o_a[k]
                # np.einsum('ijk,j,k->i', e_a, n_a, o_a)
                
                M_grad[:, a] = np.einsum('ijk,j,k->i', e_a, n_a, o_a)
                N_grad[:, a] = np.einsum('ijk,i,k->j', e_a, M_c[:, a], o_a)
                O_grad[:, a] = np.einsum('ijk,i,j->k', e_a, M_c[:, a], n_a)
            
            # Update step
            lr = 0.01
            M_c -= lr * M_grad
            N_c -= lr * N_grad
            O_c -= lr * O_grad
            
            # Project to lattice
            M_p = truncate_to_rational(M_c)
            if M_p is not None: M_c = M_p
            N_p = truncate_to_rational(N_c)
            if N_p is not None: N_c = N_p
            O_p = truncate_to_rational(O_c)
            if O_p is not None: O_c = O_p
            
        return M_c.astype(int) if not allow_half_int else M_c, N_c, O_c

    # 1. Construct Target Tensor
    U_t = np.zeros((9, 8))
    V_t = np.zeros((3, 8))
    W_t = np.zeros((8, 8))
    
    for i in range(3):
        for j in range(3):
            for k in range(8):
                c = (i*m + j)*p + k
                U_t[i*m+j, k] = 1.0
                V_t[j, k] = 1.0
                W_t[k*3+i, k] = 1.0
    T_target = tensor_product([U_t.T, V_t.T, W_t.T])
    
    best_M, best_N, best_O = None, None, None
    best_err = float('inf')
    
    # 2. Search Loop
    for seed in range(1, 50):
        if time.time() > deadline: break
        M_init, N_init, O_init = random_init(seed)
        
        M_f, N_f, O_f = refine(M_init, N_init, O_init)
        
        # Convert to python lists and verify
        if isinstance(M_f, np.ndarray) and not isinstance(M_f, float):
             # If it's a float array from refine, truncate it again for safety
             M_v = truncate_to_rational(M_f)
             N_v = truncate_to_rational(N_f)
             O_v = truncate_to_rational(O_f)
             if M_v is None or N_v is None or O_v is None:
                 continue
             
             # Build U,V,W
             U_list = []
             # U has 9 rows. Indices 0..8.
             # Rows 0..8 come from M.T (8x8).
             # Row 8 comes from 0.
             U_mat = np.vstack([M_v.T, np.zeros((1, 8))])
             
             V_mat = N_v
             W_mat = O_v
             
             # Check exactness
             # T_check = U_mat^T V_mat^T W_mat^T
             # U_mat is (9, 56). Wait. 
             # U in the problem is (9, 56). U_mat.T is (56, 9). 
             # My M_v is (8, 8). M_v.T is (8, 8). This is the part where alpha < 48.
             # My U_t.T is (8, 9).
             # My decomposition T_{ijk, a} = U_{ia} V_{ja} W_{ka}.
             # This matches my definition of M_v, N_v, O_v.
             
             # Let's re-verify indices.
             # U_mat (9, 56). 
             # V_mat (3, 56).
             # W_mat (8, 56).
             # T_check_{ijk, a} = U_mat[i, a] V_mat[j, a] W_mat[k, a].
             # This is the standard outer product.
             T_check = tensor_product([U_mat, V_mat, W_mat])
             
             err = np.max(np.abs(T_check - T_target))
             if err < best_err:
                 best_err = err
                 best_M, best_N, best_O = M_v, N_v, O_v
                 
             if best_err < 1e-6:
                 break

    if best_err < 1e-6:
        # Construct result
        M_v, N_v, O_v = best_M, best_N, best_O
        # Convert to python lists, ensuring int/half-int types
        U_res = []
        for i in range(9):
            if i < 8:
                # M_v.T is (8, 56)
                row = M_v.T[i, :].tolist()
            else:
                row = [0] * 56
            # Ensure int or float
            row = [int(x) if x == int(x) else x for x in row]
            U_res.append(row)
            
        V_res = N_v.tolist()
        V_res = [[int(x) if x == int(x) else x for x in row] for row in V_res]
        
        W_res = O_v.tolist()
        W_res = [[int(x) if x == int(x) else x for x in row] for row in W_res]
        
        return {"U": U_res, "V": V_res, "W": W_res}
        
    return None

# EVOLVE-BLOCK-END

def construct():
    start_time = time.time()
    result = _build_3x3x8_mps_search(allow_half_int=True)
    
    if result and time.time() - start_time < 110:
        # Optional: verify once more
        return result
        
    # Fallback: Schoolbook
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
