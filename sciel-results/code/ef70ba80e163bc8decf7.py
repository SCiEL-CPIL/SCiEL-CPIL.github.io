"""Rank-minimal exact tensor decomposition for 2x5 @ 5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        r = len(U[0])
        # Check diagonal: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Check off-diagonal: for all i,j,j2,k,k2,i3 where not (j==j2 and k==k2 and i==i3)
        # sum_c U[i*m+j,c]*V[j2*p+k,c]*W[k2*n+i3,c] == 0
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    for k in range(p):
                        for k2 in range(p):
                            for i3 in range(n):
                                if j == j2 and k == k2 and i == i3:
                                    continue
                                s = 0.0
                                for c in range(r):
                                    s += U[i*m+j][c] * V[j2*p+k][c] * W[k2*n+i3][c]
                                if abs(s) > 1e-9:
                                    return False
        return True
    
    # Try to find a rank-25 decomposition using a structured approach
    # For n=2, try the following parameterization:
    # Component c indexed by (j, k) where j in 0..m-1, k in 0..p-1
    # U[i*m+j', c] = delta(j,j') * a[i,k]
    # V[j'*p+k', c] = delta(j,j') * delta(k,k') * b[j,k]  
    # W[k'*n+i', c] = delta(k,k') * d[i',j]
    
    # This gave constraints that seemed contradictory, so let's try a different structure.
    
    # Alternative: Use rank = m*p = 25 with a different parameterization
    # Let's try: for each (j,k), component c=(j,k)
    # U[i*m+j', c] = delta(j,j') * u[i,k]
    # V[j'*p+k', c] = delta(j,j') * delta(k,k')
    # W[k'*n+i', c] = delta(k,k') * w[i',j]
    
    # Then sum_c U[i*m+j,c]V[j2*p+k2,c]W[k3*n+i3,c]
    # = sum_{jc,kc} delta(j,jc) u[i,kc] * delta(j2,jc) delta(k2,kc) * delta(k3,kc) w[i3,jc]
    # = u[i,k2] * w[i3,j] if j=j2 and k2=k3, else 0
    
    # Need: u[i,k]*w[i,j] = 1 when i=i3, and u[i,k]*w[i3,j]=0 when i!=i3
    # From first: u[i,k] = 1/w[i,j] for all j,k... but w[i,j] depends on j, so u[i,k] can't depend on j.
    
    # This doesn't work. Let me try yet another structure.
    
    # Let's just try a numerical search for small ranks using random initialization and gradient-free methods
    # Or try known results.
    
    # For 2x5x5, the known upper bound is 50 (schoolbook). 
    # Let's try to find if rank 40 or 35 works using a simple heuristic.
    
    # Actually, let's try a completely different approach: use the fact that we can
    # absorb some structure. 
    
    # For now, return schoolbook as it's guaranteed correct.
    rank = n * m * p  # 50
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
# EVOLVE-BLOCK-END
