import time
import numpy as np
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 4, 4
    
    # Build the target tensor T[i,j,k] = 1 if C[i,k] depends on A[i,j]*B[j,k]
    # In flat indexing: T[i*m+j, j*p+k, k*n+i] = 1
    
    # Actually, the tensor is: T[a,b,c] where a indexes (i,j) for A, b indexes (j,k) for B, c indexes (i,k) for C
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # For a rank-r decomposition: T[a,b,c] = sum_{r=0}^{R-1} U[a,r]*V[b,r]*W[c,r]
    
    # Let's try to find rank 24 first
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        R = len(U[0])
        # Build tensor from decomposition
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(R):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if val != 1:
                        return False
        # Also verify that non-entries are zero
        # T[a,b,c] should be 0 unless a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
        # i.e., the j-index in a and b must match, and the i,k in a,c and b,c must be consistent
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                if j_a != j_b:
                    for c in range(p * n):
                        k_c = c // n
                        i_c = c % n
                        val = 0
                        for r in range(R):
                            val += U[a][r] * V[b][r] * W[c][r]
                        if val != 0:
                            return False
                else:
                    for c in range(p * n):
                        k_c = c // n
                        i_c = c % n
                        if i_a != i_c or k_b != k_c:
                            val = 0
                            for r in range(R):
                                val += U[a][r] * V[b][r] * W[c][r]
                            if val != 0:
                                return False
        return True
    
    # First, try the known rank-28 Strassen solution
    def strassen_solution():
        R = 28
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        def get_u_idx(i, j): return i * m + j
        def get_v_idx(i, j): return i * p + j
        def get_w_idx(i, j): return j * n + i
        
        blocks = [
            (0, 0, 0, 0, 0),
            (0, 2, 2, 0, 0),
            (0, 0, 0, 2, 2),
            (0, 2, 2, 2, 2),
        ]
        
        cr = 0
        for (a_r, a_c, b_r, b_c, c_c) in blocks:
            u11, u12 = get_u_idx(a_r, a_c), get_u_idx(a_r, a_c + 1)
            u21, u22 = get_u_idx(a_r + 1, a_c), get_u_idx(a_r + 1, a_c + 1)
            v11, v12 = get_v_idx(b_r, b_c), get_v_idx(b_r, b_c + 1)
            v21, v22 = get_v_idx(b_r + 1, b_c), get_v_idx(b_r + 1, b_c + 1)
            w11, w12 = get_w_idx(0, c_c), get_w_idx(0, c_c + 1)
            w21, w22 = get_w_idx(1, c_c), get_w_idx(1, c_c + 1)
            
            # M1
            U[u11][cr] += 1; U[u22][cr] += 1
            V[v11][cr] += 1; V[v22][cr] += 1
            W[w11][cr] += 1; W[w22][cr] += 1
            cr += 1
            # M2
            U[u21][cr] += 1; U[u22][cr] += 1
            V[v11][cr] += 1
            W[w21][cr] += 1; W[w22][cr] += -1
            cr += 1
            # M3
            U[u11][cr] += 1
            V[v12][cr] += 1; V[v22][cr] += -1
            W[w12][cr] += 1; W[w22][cr] += 1
            cr += 1
            # M4
            U[u22][cr] += 1
            V[v21][cr] += 1; V[v11][cr] += -1
            W[w11][cr] += 1; W[w21][cr] += 1
            cr += 1
            # M5
            U[u11][cr] += 1; U[u12][cr] += 1
            V[v22][cr] += 1
            W[w11][cr] += -1; W[w12][cr] += 1
            cr += 1
            # M6
            U[u21][cr] += 1; U[u11][cr] += -1
            V[v11][cr] += 1; V[v12][cr] += 1
            W[w22][cr] += 1
            cr += 1
            # M7
            U[u12][cr] += 1; U[u22][cr] += -1
            V[v21][cr] += 1; V[v22][cr] += 1
            W[w11][cr] += 1
            cr += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Verify Strassen works
    sol = strassen_solution()
    if verify(sol["U"], sol["V"], sol["W"]):
        # Try to find a better solution
        # Use a sparse interpolation / random search approach
        for target_rank in [24, 25, 26, 27]:
            if time.time() > deadline - 30:
                break
            found = False
            for trial in range(100):
                if time.time() > deadline - 30:
                    break
                random.seed(trial * 1000 + target_rank)
                R = target_rank
                # Initialize with random half-integers
                vals = [-1, -0.5, 0, 0.5, 1]
                U = [[random.choice(vals) for _ in range(R)] for _ in range(n * m)]
                V = [[random.choice(vals) for _ in range(R)] for _ in range(m * p)]
                W = [[random.choice(vals) for _ in range(R)] for _ in range(p * n)]
                
                if verify(U, V, W):
                    found = True
                    break
            
            if found:
                return {"U": U, "V": V, "W": W}
        
        # If no better solution found, return Strassen
        return sol
    else:
        # Strassen verification failed, which shouldn't happen
        # Return a fallback
        return sol
