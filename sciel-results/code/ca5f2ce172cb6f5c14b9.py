import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form a valid exact tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Condition: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    And for all invalid (a,b,c) combinations in the tensor support, the contribution must be 0.
    Actually, the tensor T is defined such that T[i,j,k] = 1 if the indices align, else 0.
    The condition is: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    But since we are decomposing the multiplication tensor, the support is exactly the set of (i,j,k) triples.
    The condition is:
    For all i in [0,n), j in [0,m), k in [0,p):
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    This is the only condition needed because the tensor is zero elsewhere in the "tensor space" 
    but the decomposition must reconstruct the specific tensor.
    Wait, the tensor is a 3-way tensor T_{i,j,k}.
    The decomposition is T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}.
    For the matrix multiplication tensor, T_{i,j,k} = 1 if the indices correspond to the standard multiplication, else 0?
    No, the tensor is defined by the multiplication.
    T_{i,j,k} = 1 if we are computing C_{i,k} += A_{i,j} B_{j,k}.
    So T_{i,j,k} is 1 for all valid i,j,k.
    Wait, the tensor is T = sum_{i,j,k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i} (with appropriate indices).
    Specifically, T_{i,j,k} = 1 for all i,j,k in their ranges.
    So the condition is:
    For all i, j, k:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    Is that correct?
    Let's check the schoolbook.
    U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1.
    For a fixed (i,j,k), only the column col corresponding to (i,j,k) has all three non-zero.
    So the sum is 1.
    For any other combination of indices?
    The tensor is only defined on the grid [0,n) x [0,m) x [0,p).
    The decomposition must satisfy the equation for all i,j,k in the grid.
    So yes, the condition is simply:
    For all i, j, k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
    """
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # Efficient check using einsum
    # Compute T_reconstructed[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Note: W index is k*n+i.
    # Let's construct the full tensor and check.
    # But that's expensive for large ranks.
    # We can check a subset or use the structure.
    # For small dimensions, full check is fine.
    # n=2, m=3, p=7. 42 triples.
    # rank <= 42.
    # 42 * 42 operations. Very fast.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook():
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

def _construct_ladder():
    """
    Attempt to construct a lower rank decomposition using the "ladder" or "peeling" idea.
    2x3x7 = 2x3x1 + 2x3x6
    2x3x1 rank is 2 (Winograd for 2x1x1 is 2? No, 2x1x1 is just 2 mults. Rank 2).
    Actually, 2x1x1 multiplication: C0 = A0*B0 + A1*B1? No.
    2x1 matrix A, 1x1 matrix B. C = A*B.
    C0 = A0*B0, C1 = A1*B1.
    This is rank 2.
    
    2x3x6:
    Can we decompose 2x3x6 into 2x3x1 + 2x3x5?
    2x3x1 rank 2.
    2x3x5 rank?
    Winograd for 2x3x5?
    Known ranks:
    2x2x2: 7
    2x2x3: 11?
    2x3x3: 16?
    2x3x5: ?
    
    If we can find a rank for 2x3x5 that is less than 2*3*5=30, we can use it.
    Let's try to find a decomposition for 2x3x5 with rank < 30.
    If we find rank 5 for 2x3x5 (as suggested in the prompt "known to have rank 5 via Winograd" - this is likely incorrect, 2x3x5 rank is likely higher, maybe 15-20?), then total rank would be 2+2+5=9.
    However, the prompt says "known to have rank 5 via Winograd". This is a strong hint to test this specific hypothesis.
    But 2x3x5 rank 5 is extremely low. The trivial rank is 30.
    The lower bound is nm+mp+pn-m^2 = 6+15+10-9 = 22.
    So rank 5 is impossible. The prompt's hint might be misleading or referring to a different tensor.
    Wait, "2x3x5 tensor (known to have rank 5 via Winograd)" is physically impossible given the lower bound.
    So I should ignore the specific "rank 5" claim and instead try to find *any* improvement over schoolbook (42) for 2x3x7.
    
    Strategy:
    1. Try to find a decomposition for 2x3x7 with rank < 42 using random search + local optimization.
    2. If that fails, return schoolbook.
    
    Since the current program already does random search, let's improve it.
    The current program tries to find U, V such that the slices are in the span of vec(u v^T).
    This is a valid approach.
    
    Let's refine the search:
    - Use integer arithmetic as much as possible.
    - Try to solve for W exactly.
    - If W is not integer/half-integer, try to adjust U, V or scale.
    
    Actually, a more robust approach for small ranks is to use the structure of the tensor.
    The tensor T has 42 non-zero entries (all 1s).
    We want to represent T as sum of r rank-1 tensors.
    
    Let's stick to the random search but make it more efficient and robust.
    We will try ranks from 41 down to 30 (a reasonable range).
    For each rank, we will try to find U, V, W.
    
    To speed up, we can fix W to be simple (e.g., 1s and 0s) and solve for U, V?
    Or fix U, V and solve for W?
    
    The current approach fixes U, V randomly and solves for W.
    If W is not "nice", it fails.
    
    Alternative:
    Use a linear solver to find a low-rank approximation, then round?
    No, it must be exact.
    
    Let's try a different random search:
    Generate random integer U, V, W and check if they work.
    This is what inspiration 3 does.
    Probability of hitting a valid decomposition with random integers is very low.
    
    The current program's approach (fix U, V, solve W) is better because it guarantees W is in the span, but W might not be integer.
    However, if U, V are integers, W will be rational.
    If the denominator of W is small (1 or 2), it's a valid solution.
    
    Let's enhance the current program's search:
    1. Increase the number of attempts.
    2. Try different distributions for U, V (e.g., -1, 0, 1).
    3. Check if W is close to integer or half-integer.
    
    Also, consider the "ladder" construction explicitly.
    If we can find a valid decomposition for 2x3x1 (rank 2) and 2x3x6 (rank < 40), we can combine them.
    But finding a valid decomposition for 2x3x6 is the hard part.
    
    Let's just run the improved random search for the full 2x3x7 tensor.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = rank_schoolbook
    
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Try to find a lower rank decomposition
    # We will try ranks from best_rank - 1 down to 30
    # For each rank, we perform a random search
    
    random.seed(42)
    np.random.seed(42)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute slices of the tensor T
    # T_c is a matrix of size (n*m) x (m*p)
    # T_c[a, b] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # c is the output index (flattened k*n+i)
    slices = []
    for c in range(pn):
        k = c // n
        i = c % n
        T_c = np.zeros((nm, mp))
        for j in range(m):
            a = i * m + j
            b = j * p + k
            T_c[a, b] = 1
        slices.append(T_c.flatten())
    
    slices_np = np.array(slices)  # shape (pn, nm*mp)
    
    # Try ranks
    for r in range(best_rank - 1, 29, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations
        max_attempts = 100
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random U and V with small integers
            # Using -1, 0, 1 increases the chance of simple structures
            U_rand = np.random.choice([-1, 0, 1], size=(nm, r))
            V_rand = np.random.choice([-1, 0, 1], size=(mp, r))
            
            # Construct S: columns are vec(u_l v_l^T)
            S = np.zeros((nm * mp, r))
            for l in range(r):
                outer = np.outer(U_rand[:, l], V_rand[:, l])
                S[:, l] = outer.flatten()
            
            # Check if all slices are in Col(S)
            # Project slices onto Col(S)
            try:
                Q, R = np.linalg.qr(S)
                # Q has orthonormal columns spanning Col(S)
                proj = slices_np @ Q @ Q.T
                residuals = slices_np - proj
                if np.max(np.abs(residuals)) < 1e-8:
                    # Found a valid U, V
                    # Solve for W
                    # slices_np = W @ S.T
                    # W = slices_np @ S.T @ (S @ S.T)^-1
                    # Since Q is orthonormal basis for Col(S), W = slices_np @ Q.T
                    W_sol = slices_np @ Q.T
                    
                    # Check if W_sol has integer/half-integer entries
                    W_check = W_sol * 2
                    if np.allclose(W_check, np.round(W_check), atol=1e-6):
                        # Valid solution found
                        # Convert to lists
                        U_list = U_rand.tolist()
                        V_list = V_rand.tolist()
                        W_list = W_sol.tolist()
                        
                        # Final verification
                        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                            best = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = r
                            break
            except np.linalg.LinAlgError:
                continue
    
    return best
