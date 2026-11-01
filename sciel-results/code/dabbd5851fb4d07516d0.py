"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8

    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_np(U, V, W):
        """Fast numpy-based verification using einsum."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        # Reshape: U[4,R], V[8,R], W[8,R]
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Build the tensor via broadcasting
        R = Ua.shape[1]
        # For each r, the contribution is outer product of U[:,r], V[:,r], W[:,r]
        # But we need to map indices correctly
        # U index: i*m+j -> (i, j)
        # V index: j*p+k -> (j, k)
        # W index: k*n+i -> (k, i)
        # So T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for r in range(R):
            u = Ua[:, r].reshape(n, m)
            v = Va[:, r].reshape(m, p)
            w = Wa[:, r].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # This is: for each i,j,k: u[i,j]*v[j,k]*w[k,i]
            # = np.einsum('ij,jk,ki->ijk', u, v, w)
            T += np.einsum('ij,jk,ki->ijk', u, v, w)
        # Check T[i,j,k] == 1 for all i,j,k
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True

    def get_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        Returns (U, V, W) as numpy arrays of shape (4,14), (8,14), (8,14).
        """
        R = 14
        U = np.zeros((4, R), dtype=int)
        V = np.zeros((8, R), dtype=int)
        W = np.zeros((8, R), dtype=int)

        # Left block (cols 0,1 of B): Strassen on A * B_left where B_left is 2x2
        # B_left[0,0]=b00, B_left[0,1]=b01, B_left[1,0]=b10, B_left[1,1]=b11
        # V indices: j*p+k -> j*4+k
        #   b00 -> V[0], b01 -> V[1], b10 -> V[4], b11 -> V[5]
        # W indices: k*n+i -> k*2+i
        #   C[0,0] -> W[0], C[0,1] -> W[1], C[1,0] -> W[2], C[1,1] -> W[3]

        # L1: (a00+a11)(b00+b11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[5,0] = 1
        W[0,0] = 1; W[3,0] = 1

        # L2: (a10+a11)b00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[1,1] = 1; W[3,1] = -1

        # L3: a00(b01-b11)
        U[0,2] = 1
        V[1,2] = 1; V[5,2] = -1
        W[2,2] = 1; W[3,2] = 1

        # L4: a11(b10-b00)
        U[3,3] = 1
        V[4,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[1,3] = 1

        # L5: (a00+a01)b11
        U[0,4] = 1; U[1,4] = 1
        V[5,4] = 1
        W[0,4] = -1; W[2,4] = 1

        # L6: (a10-a00)(b00+b01)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1

        # L7: (a01-a11)(b10+b11)
        U[1,6] = 1; U[3,6] = -1
        V[4,6] = 1; V[5,6] = 1
        W[0,6] = 1

        # Right block (cols 2,3 of B): Strassen on A * B_right where B_right is 2x2
        # B_right[0,0]=b02, B_right[0,1]=b03, B_right[1,0]=b12, B_right[1,1]=b13
        # V indices:
        #   b02 -> V[2], b03 -> V[3], b12 -> V[6], b13 -> V[7]
        # W indices:
        #   C[0,2] -> W[4], C[0,3] -> W[5], C[1,2] -> W[6], C[1,3] -> W[7]

        # R1: (a00+a11)(b02+b13)
        U[0,7] = 1; U[3,7] = 1
        V[2,7] = 1; V[7,7] = 1
        W[4,7] = 1; W[7,7] = 1

        # R2: (a10+a11)b02
        U[2,8] = 1; U[3,8] = 1
        V[2,8] = 1
        W[5,8] = 1; W[7,8] = -1

        # R3: a00(b03-b13)
        U[0,9] = 1
        V[3,9] = 1; V[7,9] = -1
        W[6,9] = 1; W[7,9] = 1

        # R4: a11(b12-b02)
        U[3,10] = 1
        V[6,10] = 1; V[2,10] = -1
        W[4,10] = 1; W[5,10] = 1

        # R5: (a00+a01)b13
        U[0,11] = 1; U[1,11] = 1
        V[7,11] = 1
        W[4,11] = -1; W[6,11] = 1

        # R6: (a10-a00)(b02+b03)
        U[2,12] = 1; U[0,12] = -1
        V[2,12] = 1; V[3,12] = 1
        W[7,12] = 1

        # R7: (a01-a11)(b12+b13)
        U[1,13] = 1; U[3,13] = -1
        V[6,13] = 1; V[7,13] = 1
        W[4,13] = 1

        return U, V, W

    def try_strassen_rank14():
        """Try the structured rank-14 decomposition."""
        U, V, W = get_rank14()
        if verify_np(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    def _is_rank1_tensor(target):
        """
        Check if a 2x2x4 tensor can be written as u[i,j]*v[j,k]*w[k,i]
        with u in {-1,0,1}^4, v in {-1,0,1}^8, w in {-1,0,1}^8.
        Returns (u, v, w) as lists if found, else None.
        Uses algebraic constraints to prune the search space dramatically.
        """
        # target[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1,2,3}
        # Must equal u[i,j] * v[j,k] * w[k,i]
        
        # Key insight: if target[i,j,k] != 0, then u[i,j] != 0, v[j,k] != 0, w[k,i] != 0
        # And the sign must match.
        
        # Collect sign constraints
        u_sign = [None]*4  # u[i*2+j]
        v_sign = [None]*8  # v[j*4+k]
        w_sign = [None]*8  # w[k*2+i]
        
        # First pass: determine which entries must be non-zero
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    t = target[i,j,k]
                    ui = i*2+j
                    vj = j*4+k
                    wk = k*2+i
                    if t != 0:
                        # u[ui], v[vj], w[wk] must all be non-zero
                        # Sign: u[ui]*v[vj]*w[wk] = sign(t)
                        if u_sign[ui] is None:
                            u_sign[ui] = 1  # fix sign, will adjust later
                        if v_sign[vj] is None:
                            v_sign[vj] = 1
                        if w_sign[wk] is None:
                            w_sign[wk] = 1
                    else:
                        # At least one of u[ui], v[vj], w[wk] must be zero
                        pass
        
        # For entries that are zero in target, at least one factor must be zero.
        # Let's determine the support pattern.
        
        # Determine which u entries can be zero
        u_zero = [False]*4
        v_zero = [False]*8
        w_zero = [False]*8
        
        # An entry u[i*2+j] can be zero only if for ALL k, target[i,j,k] == 0
        for i in range(2):
            for j in range(2):
                ui = i*2+j
                if all(target[i,j,k] == 0 for k in range(4)):
                    u_zero[ui] = True
        
        # An entry v[j*4+k] can be zero only if for ALL i, target[i,j,k] == 0
        for j in range(2):
            for k in range(4):
                vj = j*4+k
                if all(target[i,j,k] == 0 for i in range(2)):
                    v_zero[vj] = True
        
        # An entry w[k*2+i] can be zero only if for ALL j, target[i,j,k] == 0
        for k in range(4):
            for i in range(2):
                wk = k*2+i
                if all(target[i,j,k] == 0 for j in range(2)):
                    w_zero[wk] = True
        
        # Now, for zero entries in target: target[i,j,k]=0 means
        # u[i,j]*v[j,k]*w[k,i] = 0
        # So at least one of u[i,j]=0, v[j,k]=0, w[k,i]=0
        # This gives constraints on which of these CAN be non-zero.
        
        # Let's build the list of "free" variables (those not forced to specific values)
        # and search over them.
        
        # Determine which variables are forced to be non-zero
        u_forced_nz = [False]*4
        v_forced_nz = [False]*8
        w_forced_nz = [False]*8
        
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    if target[i,j,k] != 0:
                        u_forced_nz[i*2+j] = True
                        v_forced_nz[j*4+k] = True
                        w_forced_nz[k*2+i] = True
        
        # Variables that are NOT forced non-zero and NOT forced zero are free
        # Free u: not forced_nz and not forced_zero
        free_u = [i for i in range(4) if not u_forced_nz[i] and not u_zero[i]]
        free_v = [i for i in range(8) if not v_forced_nz[i] and not v_zero[i]]
        free_w = [i for i in range(8) if not w_forced_nz[i] and not w_zero[i]]
        
        # For forced non-zero variables, their sign is determined by the equations.
        # Let's fix u[0] = 1 (if forced non-zero) to break scaling symmetry.
        
        # Build initial assignment
        u_vals = [0]*4
        v_vals = [0]*8
        w_vals = [0]*8
        
        for i in range(4):
            if u_forced_nz[i]:
                u_vals[i] = 1  # tentative
            elif u_zero[i]:
                u_vals[i] = 0
        for i in range(8):
            if v_forced_nz[i]:
                v_vals[i] = 1
            elif v_zero[i]:
                v_vals[i] = 0
        for i in range(8):
            if w_forced_nz[i]:
                w_vals[i] = 1
            elif w_zero[i]:
                w_vals[i] = 0
        
        # Now we need to determine signs and values of free variables.
        # The total number of free variables determines feasibility.
        total_free = len(free_u) + len(free_v) + len(free_w)
        
        if total_free > 20:
            return None  # Too many free variables
        
        # Search over free variable assignments: each can be -1, 0, or 1
        # But for free variables, they're not forced non-zero, so they can be 0, 1, or -1
        # However, if a free variable is set to non-zero, it must satisfy all equations.
        
        # To keep search small, let's try: free variables are either 0 or ±1
        # Enumerate all combinations
        from itertools import product as iproduct
        
        best = None
        for u_free_vals in iproduct([-1, 0, 1], repeat=len(free_u)):
            for v_free_vals in iproduct([-1, 0, 1], repeat=len(free_v)):
                for w_free_vals in iproduct([-1, 0, 1], repeat=len(free_w)):
                    # Build full assignment
                    u_test = u_vals[:]
                    v_test = v_vals[:]
                    w_test = w_vals[:]
                    
                    for idx, fi in enumerate(free_u):
                        u_test[fi] = u_free_vals[idx]
                    for idx, fi in enumerate(free_v):
                        v_test[fi] = v_free_vals[idx]
                    for idx, fi in enumerate(free_w):
                        w_test[fi] = w_free_vals[idx]
                    
                    # Check all 16 equations
                    ok = True
                    for i in range(2):
                        for j in range(2):
                            for k in range(4):
                                val = u_test[i*2+j] * v_test[j*4+k] * w_test[k*2+i]
                                if val != target[i,j,k]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    
                    if ok:
                        return (u_test, v_test, w_test)
        
        return None

    def try_random_rank13():
        """
        More targeted random search for rank 13, starting near the rank-14 solution.
        Uses a smarter approach: try removing one term and solving for a replacement.
        """
        if time.time() > deadline - 10:
            return None

        U14, V14, W14 = get_rank14()
        R14 = 14

        np.random.seed(123)
        for attempt in range(200):
            if time.time() > deadline - 3:
                break

            # Strategy: pick a term to remove, then try to find a replacement
            # that covers its contribution plus adjusts others.
            # Simpler: just do random sparse perturbation with verification.
            R = 13
            U = U14[:, :R].copy()
            V = V14[:, :R].copy()
            W = W14[:, :R].copy()

            # Perturb: flip signs or zero out a few entries
            for _ in range(4):
                mat = np.random.choice([U, V, W])
                row = np.random.randint(mat.shape[0])
                col = np.random.randint(R)
                if mat[row, col] != 0:
                    if np.random.random() < 0.5:
                        mat[row, col] = -mat[row, col]
                    else:
                        mat[row, col] = 0
                else:
                    mat[row, col] = np.random.choice([-1, 1])

            if verify_np(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

        return None

    def try_structured_rank13():
        """
        Try specific structured rank-13 decompositions.
        Key idea: share one term between the two Strassen blocks.
        """
        if time.time() > deadline - 10:
            return None

        U14, V14, W14 = get_rank14()
        if not verify_np(U14.tolist(), V14.tolist(), W14.tolist()):
            return None

        # Try all pairs (r1, r2) and check if their sum is rank-1 using the
        # efficient _is_rank1_tensor function.
        R14 = 14
        for r1 in range(R14):
            for r2 in range(r1 + 1, R14):
                if time.time() > deadline - 5:
                    return None

                target = np.zeros((2, 2, 4))
                for i in range(2):
                    for j in range(2):
                        for k in range(4):
                            target[i,j,k] = (U14[i*2+j, r1] * V14[j*4+k, r1] * W14[k*2+i, r1]
                                          + U14[i*2+j, r2] * V14[j*4+k, r2] * W14[k*2+i, r2])

                result = _is_rank1_tensor(target)
                if result is not None:
                    u_test, v_test, w_test = result
                    U_new = np.delete(U14, [r1, r2], axis=1)
                    V_new = np.delete(V14, [r1, r2], axis=1)
                    W_new = np.delete(W14, [r1, r2], axis=1)
                    U_new = np.column_stack([U_new, np.array(u_test, dtype=int)])
                    V_new = np.column_stack([V_new, np.array(v_test, dtype=int)])
                    W_new = np.column_stack([W_new, np.array(w_test, dtype=int)])
                    if verify_np(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                        return {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}

        return None

    # Try rank-13 approaches
    result = try_structured_rank13()
    if result is not None:
        return result

    result = try_random_rank13()
    if result is not None:
        return result

    # Try the structured rank-14 decomposition
    result = try_strassen_rank14()
    if result is not None:
        return result

    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
