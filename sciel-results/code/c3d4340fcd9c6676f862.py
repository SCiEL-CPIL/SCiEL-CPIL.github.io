import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 10
    deadline = time.time() + 145  # Safety margin before 150s limit

    # Schoolbook rank
    R_school = n * m * p  # 60

    def build_schoolbook():
        U = np.zeros((n * m, R_school))
        V = np.zeros((m * p, R_school))
        W = np.zeros((p * n, R_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-7):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        # Check that all entries of U, V, W are integer or half-integer
        for mat in [U, V, W]:
            for val in mat.flatten():
                if abs(val * 2 - round(val * 2)) > tol:
                    return False

        # Reconstruct tensor
        R = U.shape[1]
        T_rec = np.zeros((n * m, m * p, p * n))
        for r in range(R):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])

        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    if abs(T_rec[a, b, c] - 1.0) > tol:
                        return False

        # Check zero entries (sample)
        nonzero = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    nonzero.add((i * m + j, j * p + k, k * n + i))

        # Check all zero entries (6*30*20 = 3600 total, 60 nonzero, 3540 zero)
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    if (a, b, c) not in nonzero:
                        if abs(T_rec[a, b, c]) > tol:
                            return False
        return True

    # Start with schoolbook
    U, V, W = build_schoolbook()
    best_rank = R_school
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    if verify_decomposition(U, V, W):
        pass  # Schoolbook is valid

    # Try to reduce rank by eliminating redundant columns
    # Strategy: try to find a lower rank decomposition by searching
    # over random linear combinations of columns

    def try_reduce_rank(U_init, V_init, W_init, target_rank, time_limit=30):
        """Try to find a rank-`target_rank` decomposition starting from a higher rank one."""
        start_time = time.time()
        n, m, p = 2, 3, 10
        R = U_init.shape[1]

        # We want to find U, V, W with `target_rank` columns such that
        # sum_r U[:,r] V[:,r] W[:,r] == T
        #
        # Approach: random search over the space of rank-`target_rank` decompositions.
        # We parameterize by random matrices and solve.

        # For a given target rank R_t, we need to find U (6 x R_t), V (30 x R_t), W (20 x R_t)
        # such that the tensor equation holds.
        #
        # The tensor has 6*30*20 = 3600 entries, of which 60 are 1 and 3540 are 0.
        # Unknowns: 6*R_t + 30*R_t + 20*R_t = 56*R_t
        # Constraints: 3600 (but many are redundant due to structure)
        #
        # For R_t = 55: 56*55 = 3080 unknowns, 3600 constraints -> overdetermined
        # For R_t = 60: 56*60 = 3360 unknowns, 3600 constraints -> slightly overdetermined
        #
        # The actual rank is likely 55 or so based on literature.
        # Let's try a direct approach: fix U and V, solve for W, then iterate.

        # Alternative: use the fact that the tensor is structured.
        # Let's try to find a decomposition by solving a system of equations.

        # For a rank-R decomposition, we have R rank-1 tensors.
        # Let's fix U and V randomly and solve for W.

        # Actually, let's try a more systematic approach.
        # The matrix multiplication tensor for n x m * m x p has a known structure.
        # Let's use the "decomposition by elimination" method.

        # Fix U[:, 0] to be the standard basis vector for A[0,0] (i.e., U[0,0]=1, rest 0)
        # Then T[0, j*p+k, k*n+0] = V[j*p+k, 0] * W[k*n+0, 0] for all j, k
        # This gives us constraints on V[:, 0] and W[:, 0].

        # This is getting complex. Let's try a simpler random search.

        # Random search: try random U, V and solve for W
        for attempt in range(200):
            if time.time() - start_time > time_limit:
                break

            # Generate random U and V with half-integer entries
            U_rand = np.zeros((n * m, target_rank))
            V_rand = np.zeros((m * p, target_rank))

            for r in range(target_rank):
                # Random U column
                for a in range(n * m):
                    if random.random() < 0.3:
                        U_rand[a, r] = random.choice([-2, -1, 0, 1, 2]) / 2
                # Random V column
                for b in range(m * p):
                    if random.random() < 0.15:
                        V_rand[b, r] = random.choice([-2, -1, 0, 1, 2]) / 2

            # Solve for W: for each c, W[c, r] = T[:, :, c] / (U[:, r] * V[:, r])
            # Actually, we need: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            # This is a linear system in W for each (a,b) pair, but W is shared.
            #
            # Let's vectorize: T[a,b,c] = sum_r (U[a,r] * V[b,r]) * W[c,r]
            # For fixed c, this is: T[:, :, c] = sum_r M_r * W[c,r] where M_r = U[:,r] V[:,r]^T
            #
            # This is a linear system: for each c, we have 6*30 = 180 equations in target_rank unknowns.
            # Let's set up the system.

            # M_r[a,b] = U[a,r] * V[b,r]
            # T[a,b,c] = sum_r M_r[a,b] * W[c,r]

            # For each c, solve: sum_r M_r[:, :] * W[c, r] = T[:, :, c]
            # This is: A_c * W[c, :] = T[:, :, c].flatten()
            # where A_c is a (180 x target_rank) matrix with A_c[a*30+b, r] = U[a,r]*V[b,r]

            # But A_c is the same for all c! Let A[a*30+b, r] = U[a,r]*V[b,r]
            # Then for each c: A * W[c, :] = T[:, :, c].flatten()

            A_mat = np.zeros((n * m * m * p, target_rank))
            for r in range(target_rank):
                A_mat[:, r] = (U_rand * V_rand[:, np.newaxis]).flatten()

            # Solve A * W[c, :] = T[:, :, c] for each c
            W_sol = np.zeros((p * n, target_rank))
            valid = True
            for c in range(p * n):
                # Get T[:, :, c]
                T_c = np.zeros(n * m * m * p)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c_idx = k * n + i
                            if c_idx == c:
                                T_c[a * m * p + b] = 1.0

                # Solve least squares
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(A_mat, T_c, rcond=None)
                    W_sol[c, :] = sol
                except:
                    valid = False
                    break

            if not valid:
                continue

            # Check if solution is exact
            # Verify: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]
            T_rec = np.zeros((n * m, m * p, p * n))
            for r in range(target_rank):
                T_rec += np.einsum('a,b,c->abc', U_rand[:, r], V_rand[:, r], W_sol[:, r])

            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        if abs(T_rec[a, b, c] - 1.0) > 1e-5:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue

            # Check zero entries
            nonzero = set()
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        nonzero.add((i * m + j, j * p + k, k * n + i))

            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        if (a, b, c) not in nonzero:
                            if abs(T_rec[a, b, c]) > 1e-5:
                                ok = False
                                break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue

            # Check half-integer constraint
            for mat in [U_rand, V_rand, W_sol]:
                for val in mat.flatten():
                    if abs(val * 2 - round(val * 2)) > 1e-5:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            return U_rand, V_rand, W_sol

        return None

    # Try to find a rank-55 decomposition
    for target_rank in [55, 56, 57, 58, 59]:
        if time.time() > deadline - 10:
            break
        result = try_reduce_rank(U, V, W, target_rank, time_limit=20)
        if result is not None:
            U_try, V_try, W_try = result
            if verify_decomposition(U_try, V_try, W_try):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break

    # Return the best result
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
