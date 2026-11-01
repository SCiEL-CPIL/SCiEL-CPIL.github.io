"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145  # safety margin before 150s eval limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    # Build rank-42 from rank-14 M_{2,3,3} decomposition
    def build_rank_42_from_233(U233, V233, W233):
        R1 = U233.shape[1]
        R2 = 3
        R = R1 * R2
        
        U2 = np.ones((1, 3))
        V2 = np.eye(3)
        W2 = np.eye(3)
        
        U_comb = np.zeros((n * m, R))
        V_comb = np.zeros((m * p, R))
        W_comb = np.zeros((p * n, R))
        
        for r in range(R1):
            for s in range(R2):
                rank_idx = r * R2 + s
                for i in range(n):
                    for j in range(m):
                        U_comb[i * m + j, rank_idx] = U233[i * m + j, r] * U2[0, s]
                for j in range(m):
                    for k in range(p):
                        k1 = k // 3
                        k2 = k % 3
                        V_comb[j * p + k, rank_idx] = V233[j * 3 + k1, r] * V2[k2, s]
                for k in range(p):
                    for i in range(n):
                        k1 = k // 3
                        k2 = k % 3
                        W_comb[k * n + i, rank_idx] = W233[k1 * 2 + i, r] * W2[k2, s]
        
        def to_list(mat):
            rounded = np.round(mat * 2) / 2
            if np.allclose(mat, rounded, atol=1e-8):
                result = []
                for row in rounded:
                    row_list = []
                    for val in row:
                        if abs(val * 2 - round(val * 2)) < 1e-8:
                            row_list.append(int(round(val)))
                        else:
                            row_list.append(float(val))
                    result.append(row_list)
                return result
            else:
                return [[float(x) for x in row] for row in mat]
        
        return {"U": to_list(U_comb), "V": to_list(V_comb), "W": to_list(W_comb)}
    
    # Find rank-14 decomposition of M_{2,3,3}
    def find_rank_14_233(max_time=None):
        if max_time is None:
            max_time = deadline - time.time() - 5
        
        n1, m1, p1 = 2, 3, 3
        R = 14
        
        # Build target tensor
        target = {}
        for i in range(n1):
            for j in range(m1):
                for k in range(p1):
                    target[(i * m1 + j, j * p1 + k, k * n1 + i)] = 1
        
        def compute_error(U, V, W):
            error = 0.0
            for (a, b, c), val in target.items():
                computed = np.sum(U[a, :] * V[b, :] * W[c, :])
                error += (computed - val) ** 2
            return error
        
        def try_round_and_verify(U, V, W):
            for scale in [1, 2]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                error_r = compute_error(U_r, V_r, W_r)
                if error_r < 1e-8:
                    return U_r, V_r, W_r
            return None
        
        best_result = None
        best_error = float('inf')
        
        num_restarts = min(50, int(max_time / 0.5))
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(restart * 1000 + 42)
            
            # Initialize with some structure
            U = np.random.randn(n1 * m1, R) * 0.5
            V = np.random.randn(m1 * p1, R) * 0.5
            W = np.random.randn(p1 * n1, R) * 0.5
            
            # Normalize columns
            for r in range(R):
                norm = np.sqrt(np.sum(U[:, r]**2) * np.sum(V[:, r]**2) * np.sum(W[:, r]**2))
                if norm > 1e-10:
                    scale = norm ** (1/3)
                    U[:, r] /= scale
                    V[:, r] /= scale
                    W[:, r] /= scale
            
            max_iters = 300
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # ALS step for U
                U_new = U.copy()
                for a in range(n1 * m1):
                    equations = []
                    for b in range(m1 * p1):
                        for c in range(p1 * n1):
                            if (a, b, c) in target:
                                coeff = np.array([V[b, r] * W[c, r] for r in range(R)])
                                equations.append((coeff, target[(a, b, c)]))
                    
                    if len(equations) >= R:
                        A_mat = np.array([eq[0] for eq in equations[:R * 3]])
                        b_vec = np.array([eq[1] for eq in equations[:R * 3]])
                        U_new[a, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                U = U_new
                
                # ALS step for V
                V_new = V.copy()
                for b in range(m1 * p1):
                    equations = []
                    for a in range(n1 * m1):
                        for c in range(p1 * n1):
                            if (a, b, c) in target:
                                coeff = np.array([U[a, r] * W[c, r] for r in range(R)])
                                equations.append((coeff, target[(a, b, c)]))
                    
                    if len(equations) >= R:
                        A_mat = np.array([eq[0] for eq in equations[:R * 3]])
                        b_vec = np.array([eq[1] for eq in equations[:R * 3]])
                        V_new[b, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                V = V_new
                
                # ALS step for W
                W_new = W.copy()
                for c in range(p1 * n1):
                    equations = []
                    for a in range(n1 * m1):
                        for b in range(m1 * p1):
                            if (a, b, c) in target:
                                coeff = np.array([U[a, r] * V[b, r] for r in range(R)])
                                equations.append((coeff, target[(a, b, c)]))
                    
                    if len(equations) >= R:
                        A_mat = np.array([eq[0] for eq in equations[:R * 3]])
                        b_vec = np.array([eq[1] for eq in equations[:R * 3]])
                        W_new[c, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                W = W_new
                
                # Scale to prevent blowup
                for r in range(R):
                    norm = np.sqrt(np.sum(U[:, r]**2) * np.sum(V[:, r]**2) * np.sum(W[:, r]**2))
                    if norm > 1e-10 and abs(norm - 1.0) > 0.1:
                        scale = norm ** (1/3)
                        U[:, r] /= scale
                        V[:, r] /= scale
                        W[:, r] /= scale
                
                error = compute_error(U, V, W)
                
                if error < best_error:
                    best_error = error
                    rounded = try_round_and_verify(U, V, W)
                    if rounded is not None:
                        return rounded
                    best_result = (U.copy(), V.copy(), W.copy())
                
                if error < 1e-10:
                    rounded = try_round_and_verify(U, V, W)
                    if rounded is not None:
                        return rounded
                    break
            
            if best_error < 1e-6 and best_result is not None:
                rounded = try_round_and_verify(best_result[0], best_result[1], best_result[2])
                if rounded is not None:
                    return rounded
        
        return best_result
    
    # Try to find the decomposition
    if time.time() < deadline - 10:
        result = find_rank_14_233()
        if result is not None:
            U233, V233, W233 = result
            if time.time() < deadline - 3:
                return build_rank_42_from_233(U233, V233, W233)
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
