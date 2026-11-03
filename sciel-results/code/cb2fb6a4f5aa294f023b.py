import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = 3, 4, 6
    t_start = time.time()
    deadline = 135  # safety margin
    
    def build_rank66():
        R = 66
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        
        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00, A01 = a_r*m+a_c, a_r*m+a_c+1
            A10, A11 = (a_r+1)*m+a_c, (a_r+1)*m+a_c+1
            B00, B01 = b_r*p+b_c, b_r*p+b_c+1
            B10, B11 = (b_r+1)*p+b_c, (b_r+1)*p+b_c+1
            C00, C01 = c_c*n+c_r, (c_c+1)*n+c_r
            C10, C11 = c_c*n+(c_r+1), (c_c+1)*n+(c_r+1)
            
            U[A00][col]+=1; U[A11][col]+=1; V[B00][col]+=1; V[B11][col]+=1; W[C00][col]+=1; W[C11][col]+=1; col+=1
            U[A10][col]+=1; U[A11][col]+=1; V[B00][col]+=1; W[C10][col]+=1; W[C11][col]-=1; col+=1
            U[A00][col]+=1; V[B01][col]+=1; V[B11][col]-=1; W[C01][col]+=1; W[C11][col]+=1; col+=1
            U[A11][col]+=1; V[B10][col]+=1; V[B00][col]-=1; W[C00][col]+=1; W[C10][col]+=1; col+=1
            U[A00][col]+=1; U[A01][col]+=1; V[B11][col]+=1; W[C00][col]-=1; W[C01][col]+=1; col+=1
            U[A10][col]+=1; U[A00][col]-=1; V[B00][col]+=1; V[B01][col]+=1; W[C11][col]+=1; col+=1
            U[A01][col]+=1; U[A11][col]-=1; V[B10][col]+=1; V[B11][col]+=1; W[C00][col]+=1; col+=1
        
        def add_1x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0, A1 = a_r*m+a_c, a_r*m+a_c+1
            B00, B01 = b_r*p+b_c, b_r*p+b_c+1
            B10, B11 = (b_r+1)*p+b_c, (b_r+1)*p+b_c+1
            C0, C1 = c_c*n+c_r, (c_c+1)*n+c_r
            U[A0][col]+=1; V[B00][col]+=1; W[C0][col]+=1; col+=1
            U[A1][col]+=1; V[B10][col]+=1; W[C0][col]+=1; col+=1
            U[A0][col]+=1; V[B01][col]+=1; W[C1][col]+=1; col+=1
            U[A1][col]+=1; V[B11][col]+=1; W[C1][col]+=1; col+=1
        
        add_1x2(0,0,0,0,0,0); add_1x2(0,2,2,0,0,0)
        add_1x2(0,0,0,2,0,2); add_1x2(0,2,2,2,0,2)
        add_1x2(0,0,0,4,0,4); add_1x2(0,2,2,4,0,4)
        add_strassen(1,0,0,0,1,0); add_strassen(1,2,2,0,1,0)
        add_strassen(1,0,0,2,1,2); add_strassen(1,2,2,2,1,2)
        add_strassen(1,0,0,4,1,4); add_strassen(1,2,2,4,1,4)
        
        return U, V, W, col
    
    U, V, W, R = build_rank66()
    
    # Verify
    U_np, V_np, W_np = np.array(U,float), np.array(V,float), np.array(W,float)
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    T_recon = np.einsum('ac,bc,dc->abd', U_np, V_np, W_np, optimize=True)
    if not np.allclose(T_recon, T_target, atol=1e-9):
        R_fb = n*m*p
        U_fb = [[0]*R_fb for _ in range(n*m)]
        V_fb = [[0]*R_fb for _ in range(m*p)]
        W_fb = [[0]*R_fb for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c]=1; V_fb[j*p+k][c]=1; W_fb[k*n+i][c]=1; c+=1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    best_R = R
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to reduce by finding sparse equivalents
    # Use a greedy approach: try to replace two columns with one
    while time.time() - t_start < deadline - 5:
        improved = False
        for r1 in range(R):
            if time.time() - t_start > deadline - 10:
                break
            for r2 in range(r1+1, R):
                if time.time() - t_start > deadline - 10:
                    break
                # Try to find U', V', W' such that
                # U[:,r1]*V[:,r1]*W[:,r1] + U[:,r2]*V[:,r2]*W[:,r2] = U'*V'*W'
                # This is generally not possible, but try simple cases
                # If one of the terms is a scalar multiple, we can merge
                u1 = np.array([U[i][r1] for i in range(n*m)], float)
                v1 = np.array([V[i][r1] for i in range(m*p)], float)
                w1 = np.array([W[i][r1] for i in range(p*n)], float)
                u2 = np.array([U[i][r2] for i in range(n*m)], float)
                v2 = np.array([V[i][r2] for i in range(m*p)], float)
                w2 = np.array([W[i][r2] for i in range(p*n)], float)
                
                # Check if term2 = alpha * term1
                if np.allclose(u2, 0) and np.allclose(v2, 0) and np.allclose(w2, 0):
                    # Remove r2
                    U = [row[:r2] + row[r2+1:] for row in U]
                    V = [row[:r2] + row[r2+1:] for row in V]
                    W = [row[:r2] + row[r2+1:] for row in W]
                    R -= 1
                    if r1 >= R: r1 = R-1
                    improved = True
                    break
                elif np.allclose(u1, 0) and np.allclose(v1, 0) and np.allclose(w1, 0):
                    U = [row[:r1] + row[r1+1:] for row in U]
                    V = [row[:r1] + row[r1+1:] for row in V]
                    W = [row[:r1] + row[r1+1:] for row in W]
                    R -= 1
                    improved = True
                    break
                else:
                    # Check if they are proportional
                    try:
                        # Find alpha such that u2 ≈ alpha*u1
                        mask = np.abs(u1) > 1e-10
                        if mask.sum() > 0:
                            alpha_u = np.median(u2[mask] / u1[mask])
                            if np.allclose(u2, alpha_u * u1, atol=1e-9):
                                mask_v = np.abs(v1) > 1e-10
                                if mask_v.sum() > 0:
                                    alpha_v = np.median(v2[mask_v] / v1[mask_v])
                                    if np.allclose(v2, alpha_v * v1, atol=1e-9):
                                        mask_w = np.abs(w1) > 1e-10
                                        if mask_w.sum() > 0:
                                            alpha_w = np.median(w2[mask_w] / w1[mask_w])
                                            if np.allclose(w2, alpha_w * w1, atol=1e-9):
                                                if abs(alpha_u - alpha_v) < 1e-9 and abs(alpha_v - alpha_w) < 1e-9:
                                                    # Merge: add the two terms
                                                    for i in range(n*m):
                                                        U[i][r1] += U[i][r2]
                                                    for i in range(m*p):
                                                        V[i][r1] += V[i][r2]
                                                    for i in range(p*n):
                                                        W[i][r1] += W[i][r2]
                                                    U = [row[:r2] + row[r2+1:] for row in U]
                                                    V = [row[:r2] + row[r2+1:] for row in V]
                                                    W = [row[:r2] + row[r2+1:] for row in W]
                                                    R -= 1
                                                    if r1 >= R: r1 = R-1
                                                    improved = True
                                                    break
                    except:
                        pass
            if improved:
                break
        
        if not improved:
            break
        
        # Verify
        U_np, V_np, W_np = np.array(U,float), np.array(V,float), np.array(W,float)
        T_recon = np.einsum('ac,bc,dc->abd', U_np, V_np, W_np, optimize=True)
        if not np.allclose(T_recon, T_target, atol=1e-9):
            U, V, W, R = best_U, best_V, best_W, best_R
            break
        else:
            best_R = R
            best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Round
    def round_mat(M):
        return [[round(x) if abs(x-round(x))<1e-9 else x for x in row] for row in M]
    
    return {"U": round_mat(best_U), "V": round_mat(best_V), "W": round_mat(best_W)}
