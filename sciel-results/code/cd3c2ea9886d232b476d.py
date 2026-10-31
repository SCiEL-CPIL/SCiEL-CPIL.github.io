import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Hermite polynomial values at 0
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 10080
    # H_12(0) = 665280
    h0 = 1.0
    h4 = 12.0
    h8 = 10080.0
    h12 = 665280.0
    
    # Weights w_k = 2^k k! (ignoring sqrt(pi))
    # k=0: 1
    # k=4: 16 * 24 = 384
    # k=8: 256 * 40320 = 10321920
    # k=12: 4096 * 479001600 = 1962022963200
    w0 = 1.0
    w4 = 384.0
    w8 = 10321920.0
    w12 = 1962022963200.0
    
    # u_k = (k + 0.5) w_k
    u0 = 0.5 * w0
    u4 = 4.5 * w4
    u8 = 8.5 * w8
    u12 = 12.5 * w12
    
    # Matrix P such that c = P z, z = [c0, c4, c8]
    # c12 = -(c0*h0 + c4*h4 + c8*h8) / h12
    alpha = -h0 / h12
    beta = -h4 / h12
    gamma = -h8 / h12
    
    # P is 4x3
    # P = [[1, 0, 0],
    #      [0, 1, 0],
    #      [0, 0, 1],
    #      [alpha, beta, gamma]]
    
    # V_diag = [w0, w4, w8, w12]
    # U_diag = [u0, u4, u8, u12]
    
    # V_tilde = P^T V P
    # V_tilde[i,j] = sum_k P[k,i] V[k,k] P[k,j]
    # Since V is diagonal, V_tilde[i,j] = P[0,i]w0 P[0,j] + P[1,i]w4 P[1,j] + P[2,i]w8 P[2,j] + P[3,i]w12 P[3,j]
    
    # P[0] = [1, 0, 0]
    # P[1] = [0, 1, 0]
    # P[2] = [0, 0, 1]
    # P[3] = [alpha, beta, gamma]
    
    V_tilde = np.zeros((3, 3))
    U_tilde = np.zeros((3, 3))
    
    # Diagonal parts from first 3 rows
    V_tilde[0,0] += w0
    V_tilde[1,1] += w4
    V_tilde[2,2] += w8
    
    U_tilde[0,0] += u0
    U_tilde[1,1] += u4
    U_tilde[2,2] += u8
    
    # Off-diagonal and extra diagonal from 4th row
    p3 = np.array([alpha, beta, gamma])
    V_tilde += w12 * np.outer(p3, p3)
    U_tilde += u12 * np.outer(p3, p3)
    
    # Solve generalized eigenvalue problem: U_tilde z = lambda V_tilde z
    # This is equivalent to V_tilde^{-1} U_tilde z = lambda z
    # Since V_tilde is symmetric positive definite (it's a Gram matrix), we can use np.linalg.eig
    # Or better, use scipy.linalg.eigh for symmetric definite problems
    try:
        from scipy.linalg import eigh
        eigenvalues, eigenvectors = eigh(U_tilde, V_tilde)
        # The eigenvalues are the possible values of the Rayleigh quotient
        # We want the minimum
        min_val = eigenvalues[0]
        
        # The corresponding eigenvector gives the optimal z
        z_opt = eigenvectors[:, 0]
        
        # Recover c
        c0 = z_opt[0]
        c4 = z_opt[1]
        c8 = z_opt[2]
        
        return np.array([c0, c4, c8])
        
    except Exception as e:
        # Fallback to numerical optimization if eigh fails
        from scipy.optimize import minimize
        
        def obj(c):
            c0, c4, c8 = c
            c12 = -(c0*h0 + c4*h4 + c8*h8) / h12
            cs = [c0, c4, c8, c12]
            ws = [w0, w4, w8, w12]
            us = [u0, u4, u8, u12]
            
            num = sum(c*c*u for c, u in zip(cs, us))
            den = sum(c*c*w for c, w in zip(cs, ws))
            if den < 1e-15:
                return 1e10
            return num / den
        
        # Start from a reasonable guess
        # The unconstrained minimum is c0 != 0, others 0, but c12 must be 0, so c0=0.
        # So we need to balance.
        best_c = np.array([1.0, 1.0, 1.0])
        best_val = obj(best_c)
        
        for _ in range(10):
            res = minimize(obj, best_c, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x
        
        return best_c
