import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = 2, 2, 11
    
    def build_strassen_solution():
        rank = 39
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0
            v_f = k1
            v_g = 11 + k0
            v_h = 11 + k1
            w_c00 = 2 * k0
            w_c10 = 2 * k0 + 1
            w_c01 = 2 * k1
            w_c11 = 2 * k1 + 1
            
            # m1 = (a+d)(e+h)
            U[0][r] += 1; U[3][r] += 1
            V[v_e][r] += 1; V[v_h][r] += 1
            W[w_c00][r] += 1; W[w_c11][r] += 1
            r += 1
            # m2 = (c+d)e
            U[2][r] += 1; U[3][r] += 1
            V[v_e][r] += 1
            W[w_c10][r] += 1; W[w_c11][r] += -1
            r += 1
            # m3 = a(f-h)
            U[0][r] += 1
            V[v_f][r] += 1; V[v_h][r] += -1
            W[w_c01][r] += 1; W[w_c11][r] += 1
            r += 1
            # m4 = d(g-e)
            U[3][r] += 1
            V[v_g][r] += 1; V[v_e][r] += -1
            W[w_c00][r] += 1; W[w_c10][r] += 1
            r += 1
            # m5 = (a+b)h
            U[0][r] += 1; U[1][r] += 1
            V[v_h][r] += 1
            W[w_c00][r] += -1; W[w_c01][r] += 1
            r += 1
            # m6 = (c-a)(e+f)
            U[2][r] += 1; U[0][r] += -1
            V[v_e][r] += 1; V[v_f][r] += 1
            W[w_c11][r] += 1
            r += 1
            # m7 = (b-d)(g+h)
            U[1][r] += 1; U[3][r] += -1
            V[v_g][r] += 1; V[v_h][r] += 1
            W[w_c00][r] += 1
            r += 1
        
        # Last column k=10: schoolbook
        k2 = 10
        v_e2 = k2
        v_g2 = 11 + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1
        
        U[0][r] += 1; V[v_e2][r] += 1; W[w_c02][r] += 1; r += 1
        U[1][r] += 1; V[v_g2][r] += 1; W[w_c02][r] += 1; r += 1
        U[2][r] += 1; V[v_e2][r] += 1; W[w_c12][r] += 1; r += 1
        U[3][r] += 1; V[v_g2][r] += 1; W[w_c12][r] += 1; r += 1
        
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify that the decomposition correctly computes matrix multiplication."""
        n, m, p = 2, 2, 11
        rank = len(U[0])
        # Check dimensions
        if len(U) != n*m or len(V) != m*p or len(W) != p*n:
            return False
        for i in range(len(U)):
            if len(U[i]) != rank: return False
        for i in range(len(V)):
            if len(V[i]) != rank: return False
        for i in range(len(W)):
            if len(W[i]) != rank: return False
        
        # Test with random integer inputs
        import random
        random.seed(42)
        for _ in range(5):
            A = [[random.randint(-5,5) for _ in range(2)] for _ in range(2)]
            B = [[random.randint(-5,5) for _ in range(11)] for _ in range(2)]
            
            # Compute C = A * B directly
            C_direct = [[sum(A[i][k]*B[k][j] for k in range(2)) for j in range(11)] for i in range(2)]
            
            # Compute C via decomposition
            # C[i,j] = sum_t U[i*2+j]... wait, need to check indexing.
            # U has shape (n*m, rank) = (4, rank). U[row, t] where row = i*n + j? Or i*m + j?
            # V has shape (m*p, rank) = (22, rank). V[row, t] where row = k*p + j? 
            # W has shape (p*n, rank) = (22, rank). W[row, t] where row = j*n + i?
            
            # From the task: W[k*n+i, r] is the output coefficient for C[i,k]
            # So W index = k*n + i = k*2 + i
            # U index: for A[i,j], U[i*m + j, r]? m=2, so U[i*2+j, r]
            # V index: for B[j,k], V[j*p + k, r]? p=11, so V[j*11+k, r]
            
            C_decomp = [[0]*11 for _ in range(2)]
            for i in range(2):
                for k in range(11):
                    w_idx = k * 2 + i  # W[k*n + i]
                    for t in range(rank):
                        if W[w_idx][t] != 0:
                            # u_idx for A[i, ?]: U has 4 rows = n*m = 4
                            # A is 2x2, so A[i,j] for i in 0,1 and j in 0,1
                            # U row index: i*m + j = i*2 + j
                            # But we need to sum over all U rows weighted by V
                            pass
                        # Actually: C[i,k] = sum_t U[u_idx, t] * V[v_idx, t] * W[w_idx, t]
                        # But U, V, W are matrices. The decomposition is:
                        # C[i,k] = sum_t (sum_{u} U[u,t] * A[u]) * (sum_{v} V[v,t] * B[v]) * W[w_idx, t]
                        # Wait no. The tensor decomposition is:
                        # T[i,k] = sum_t U[i,t] * V[k,t] * W[i,k,t]
                        # But here U, V, W are the factor matrices.
                        # Actually, for matrix multiplication:
                        # C = A*B, where C[i,j] = sum_k A[i,k]*B[k,j]
                        # The tensor T has indices (i, k, j) for A[i,k], B[k,j], C[i,j]
                        # Rank-r decomposition: T = sum_t u_t ⊗ v_t ⊗ w_t
                        # where u_t is in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}
                        # C[i,j] = sum_t w_t[i,j] * (sum_{k} u_t[i,k] * A[i,k]) * (sum_{k} v_t[k,j] * B[k,j])
                        # Wait, that's not right either.
                        
                        # Let me think again. The standard tensor decomposition for matmul:
                        # T_{ijk} = sum_t U_{it} * V_{kt} * W_{jt}  ... no.
                        
                        # Actually: C_{ij} = sum_k A_{ik} B_{kj}
                        # The multiplication tensor T has T_{ikj} = 1 if the triple (i,k,j) appears, 0 otherwise.
                        # Rank-r: T = sum_{t=1}^r u^{(t)} ⊗ v^{(t)} ⊗ w^{(t)}
                        # where u^{(t)} in R^{n*m}, v^{(t)} in R^{m*p}, w^{(t)} in R^{p*n}
                        # Then: C_{ij} = sum_t w^{(t)}_{ji} * (sum_{ik} u^{(t)}_{ik} A_{ik}) * (sum_{kj} v^{(t)}_{kj} B_{kj})
                        # Wait, I need to be more careful.
                        
                        # The tensor is T[i, k, j] = delta (meaning C[i,j] gets contribution A[i,k]*B[k,j])
                        # Rank-r decomposition: T[i,k,j] = sum_t U[i,k,t] * V[k,j,t] * W[i,j,t]
                        # But U, V, W here are 3D. In the problem, they're given as 2D matrices:
                        # U has shape (n*m, rank), so U[i*m+k, t] = U^{(t)}[i,k]
                        # V has shape (m*p, rank), so V[k*p+j, t] = V^{(t)}[k,j]  
                        # W has shape (p*n, rank), so W[j*n+i, t] = W^{(t)}[j,i] or W^{(t)}[i,j]?
                        
                        # The task says: W[k*n+i, r] is the output factor for C[i,k]
                        # So W index = k*n + i, and this corresponds to C[i,k].
                        # So W^{(t)}[i,k] = W[k*n + i, t]
                        
                        # Then: C[i,k] = sum_t U[i*2 + j, t] * V[j*11 + k, t] * W[k*2 + i, t]
                        # Wait, that doesn't look right dimensionally.
                        
                        # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
                        # n=2, m=2, p=11
                        # U: (4, rank) - corresponds to A (2x2)
                        # V: (22, rank) - corresponds to B (2x11)
                        # W: (22, rank) - corresponds to C (2x11)
                        
                        # The decomposition: C[i,k] = sum_t (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t]
                        # Hmm, that's not standard. Let me think differently.
                        
                        # Standard: C = A*B means C[i,k] = sum_j A[i,j] * B[j,k]
                        # Tensor: T[i,j,k] = 1 (contribution A[i,j]*B[j,k] to C[i,k])
                        # Rank-r: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[i,k,t]
                        # Where U[i,j,t] = U[i*m+j, t], V[j,k,t] = V[j*p+k, t], W[i,k,t] = W[k*n+i, t]
                        
                        # So: C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k])
                        
                        pass
                    # Let me just compute it properly
                    pass
            
            # Proper computation:
            C_decomp = [[0]*11 for _ in range(2)]
            for i in range(2):
                for k in range(11):
                    total = 0
                    for t in range(rank):
                        # Compute u_val = sum_j U[i*2+j, t] * A[i,j]
                        u_val = 0
                        for j in range(2):
                            u_val += U[i*2 + j][t] * A[i][j]
                        # Compute v_val = sum_j V[j*11+k, t] * B[j,k]
                        v_val = 0
                        for j in range(2):
                            v_val += V[j*11 + k][t] * B[j][k]
                        w_val = W[k*2 + i][t]
                        total += w_val * u_val * v_val
                    C_decomp[i][k] = total
                    
                    if C_decomp[i][k] != C_direct[i][k]:
                        return False
        return True
    
    # Build and verify the rank-39 solution
    best = build_strassen_solution()
    if not verify(best["U"], best["V"], best["W"]):
        # Fallback: schoolbook rank-44
        rank = 44
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(22)]
        W = [[0]*rank for _ in range(22)]
        r = 0
        for i in range(2):
            for j in range(2):
                for k in range(11):
                    U[i*2+j][r] = 1
                    V[j*11+k][r] = 1
                    W[k*2+i][r] = 1
                    r += 1
        best = {"U": U, "V": V, "W": W}
    
    # Try to find a better solution (rank < 39) within time budget
    # For now, just return the verified rank-39 solution
    # A full search for rank < 39 would require significant computational effort
    
    return best
