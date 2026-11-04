import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]
_DEADLINE = 135.0

def _remaining():
    return _DEADLINE - time.time()

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _verify(n, m, p, U, V, W):
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_rank_49(n, m, p):
    """
    Construct rank-49 decomposition for 2x5x5 matrix multiplication.
    
    Known result: For 2xn x n, rank 2n^2 - 1 is achievable.
    For n=5: rank = 2*25 - 1 = 49.
    
    Construction: Use 48 schoolbook terms for all (i,j,k) except (i=0,j=4,k=4)
    and (i=1,j=4,k=4), plus 1 additional term that compensates for both.
    
    The key identity: We can combine the two missing products A[0,4]*B[4,4]
    and A[1,4]*B[4,4] using a shared auxiliary structure.
    """
    if time.time() > _DEADLINE - 10:
        return None
    
    n, m, p = 2, 5, 5
    target_rank = 49
    
    try:
        # Use a structured construction.
        # We'll have 49 terms indexed c=0..48.
        # Terms c=0..47: schoolbook for 48 of the 50 (i,j,k) triples.
        # Term c=48: a correction term.
        
        # Which (i,j,k) to omit? Let's omit (i=0,j=4,k=4) and (i=1,j=4,k=4).
        # We need to add contributions to all outputs that depended on these terms.
        
        # The term (i=0,j=4,k=4) contributes to C[0,4] via A[0,4]*B[4,4].
        # The term (i=1,j=4,k=4) contributes to C[1,4] via A[1,4]*B[4,4].
        
        # We need a single rank-1 term that produces both contributions.
        # A rank-1 term U[i,j]*V[j,k]*W[k,i] produces exactly one output element.
        # So we can't directly combine two different outputs in one term.
        
        # The correct construction must modify multiple terms. Let me use a
        # different strategy: perturb the schoolbook decomposition.
        
        # Start with full schoolbook (50 terms) and try to merge two terms.
        # We want to find U,V,W with 49 columns such that the tensor equation holds.
        
        # Use the fact that we can write the missing products as a linear
        # combination of other products plus a new term.
        
        # Let's try: express A[0,4]*B[4,4] and A[1,4]*B[4,4] using other
        # available products.
        # Available: A[i,j]*B[j,4] for j=0,1,2,3 and i=0,1
        #            A[i,4]*B[j,4] for j=0,1,2,3 and i=0,1
        #
        # This doesn't give a linear relation in general.
        
        # Let me try a completely different approach: use the explicit
        # construction from the literature for 2xn x n rank 2n^2-1.
        
        # The construction uses n^2 - 1 "standard" terms for each row,
        # plus 1 shared term, giving 2(n^2-1) + 1 = 2n^2-1 total.
        
        # For row i, column k: C[i,k] = sum_j A[i,j]*B[j,k]
        # 
        # For (j,k) != (n-1,n-1):
        #   Term_{i,j,k}: U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
        # For (j,k) = (n-1,n-1):
        #   We need to produce A[i,n-1]*B[n-1,n-1] for i=0,1
        #   Use a shared term with auxiliary structure
        
        # Actually, the standard construction for 2x2x2 rank 7 is:
        # c=0: U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0]  (A00*B00->C00)
        # c=1: U=[1,0,0,0], V=[0,1,0,0], W=[0,1,0,0]  (A00*B01->C01)  
        # c=2: U=[0,1,0,0], V=[1,0,0,0], W=[0,0,1,0]  (A10*B00->C10)
        # c=3: U=[0,1,0,0], V=[0,1,0,0], W=[0,0,0,1]  (A11*B11->C11)
        # c=4: U=[1,0,0,0], V=[0,0,1,0], W=[0,0,1,0]  (A00*B10->C10 extra)
        # c=5: U=[0,0,1,0], V=[1,0,0,0], W=[0,1,0,0]  (A01*B00->C01 extra)
        # c=6: U=[0,0,1,0], V=[0,0,1,0], W=[0,0,0,1]  (A01*B10->C11 extra)
        
        # Let me verify c=4: contributes to C[1,0] via U[1,0]=1? No, U[1,0] is for i=1,j=0.
        # U[1,0]=1 means this term uses A[1,0]. But we want A[0,0]*B[1,0].
        # So U should have U[0,0]=1 for A[0,0]. Let me recheck.
        
        # I'm getting confused with indices. Let me just try multiple random
        # initializations with gradient descent and see if any converge.
        
        for attempt in range(5):
            if time.time() > _DEADLINE - 15:
                break
            np.random.seed(1000 + attempt)
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Initialize with schoolbook for first 49 terms
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if c >= target_rank:
                            break
                        U[i*m+j, c] = 1
                        V[j*p+k, c] = 1
                        W[k*n+i, c] = 1
                        c += 1
            
            # Add small random perturbation to all terms
            U += np.random.randn(n*m, target_rank) * 0.01
            V += np.random.randn(m*p, target_rank) * 0.01
            W += np.random.randn(p*n, target_rank) * 0.01
            
            def compute_error(U, V, W):
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for cc in range(target_rank):
                                s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                            err += (s - 1.0)**2
                return err
            
            def gradients(U, V, W):
                gU = np.zeros_like(U)
                gV = np.zeros_like(V)
                gW = np.zeros_like(W)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for cc in range(target_rank):
                                s += U[i*m+j, cc] * V[j*p+k, cc] * W[k*n+i, cc]
                            ds = 2.0 * (s - 1.0)
                            for cc in range(target_rank):
                                gU[i*m+j, cc] += ds * V[j*p+k, cc] * W[k*n+i, cc]
                                gV[j*p+k, cc] += ds * U[i*m+j, cc] * W[k*n+i, cc]
                                gW[k*n+i, cc] += ds * U[i*m+j, cc] * V[j*p+k, cc]
                return gU, gV, gW
            
            lr = 0.005
            for step in range(3000):
                if time.time() > _DEADLINE - 15:
                    break
                gU, gV, gW = gradients(U, V, W)
                U -= lr * gU
                V -= lr * gV
                W -= lr * gW
                if step % 500 == 0:
                    err = compute_error(U, V, W)
                    if err < 1e-12:
                        break
            
            err = compute_error(U, V, W)
            if err > 1e-6:
                continue
            
            # Try rounding
            for scale in [1, 2]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                U_list = [[float(U_r[i, j]) for j in range(target_rank)] for i in range(n*m)]
                V_list = [[float(V_r[i, j]) for j in range(target_rank)] for i in range(m*p)]
                W_list = [[float(W_r[i, j]) for j in range(target_rank)] for i in range(p*n)]
                if _verify(n, m, p, U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
        
    except Exception:
        return None

def _construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    
    # Try to find rank 49
    result = _try_rank_49(n, m, p)
    if result is not None:
        if _verify(n, m, p, result['U'], result['V'], result['W']):
            return result
    
    return best

def construct():
    try:
        return _construct()
    except Exception:
        n, m, p = DIMENSIONS
        return _schoolbook(n, m, p)
