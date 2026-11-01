"""Matrix multiplication 2x2 by 2x6 rank-minimal decomposition."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _verify(U, V, W, n=2, m=2, p=6, tol=1e-9):
    """Verify that U,V,W exactly decompose the multiplication tensor.
    
    Convention: W[k*n+i, r], U[i*m+j, r], V[j*p+k, r]
    The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 for all i,j,k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k]).
    """
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n=2, m=2, p=6):
    """Standard schoolbook decomposition of rank n*m*p = 24."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2(n=2, m=2, p=2):
    """Strassen decomposition for 2x2 * 2x2, rank 7."""
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # 
    # M1 = (a+b)(e+f)
    # M2 = (c+d)(e)
    # M3 = (a)(f-g)
    # M4 = (d)(h-g)
    # M5 = (a+d)(h)
    # M6 = (b+c)(g)
    # M7 = (a-c)(e+f)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # U: rows indexed by i*m+j (i=row of A, j=col of A), 4 rows
    # V: rows indexed by j*p+k (j=row of B, k=col of B), 4 rows
    # W: rows indexed by k*n+i (k=col of result, i=row of result), 4 rows
    #
    # A indices: a=0, b=1, c=2, d=3
    # B indices: e=0, f=1, g=2, h=3
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3 (k*n+i: k=col, i=row)
    
    rank = 7
    U = [[0.0]*rank for _ in range(4)]
    V = [[0.0]*rank for _ in range(4)]
    W = [[0.0]*rank for _ in range(4)]
    
    # M1 = (a+b)(e+f): U[0]=1, U[1]=1; V[0]=1, V[1]=1
    U[0][0] = 1.0; U[1][0] = 1.0
    V[0][0] = 1.0; V[1][0] = 1.0
    # M1 contributes to C[0,0] and C[1,1]
    W[0][0] = 1.0  # C[0,0]
    W[3][0] = 1.0  # C[1,1]
    
    # M2 = (c+d)(e): U[2]=1, U[3]=1; V[0]=1
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    # M2 contributes to C[1,0] with +1, C[1,1] with -1
    W[2][1] = 1.0  # C[1,0]
    W[3][1] = -1.0 # C[1,1]
    
    # M3 = (a)(f-g): U[0]=1; V[1]=1, V[2]=-1
    U[0][2] = 1.0
    V[1][2] = 1.0; V[2][2] = -1.0
    # M3 contributes to C[0,1] with +1, C[1,1] with +1
    W[1][2] = 1.0  # C[0,1]
    W[3][2] = 1.0  # C[1,1]
    
    # M4 = (d)(h-g): U[3]=1; V[2]=-1, V[3]=1
    U[3][3] = 1.0
    V[2][3] = -1.0; V[3][3] = 1.0
    # M4 contributes to C[0,0] with +1, C[1,0] with +1
    W[0][3] = 1.0  # C[0,0]
    W[2][3] = 1.0  # C[1,0]
    
    # M5 = (a+d)(h): U[0]=1, U[3]=1; V[3]=1
    U[0][4] = 1.0; U[3][4] = 1.0
    V[3][4] = 1.0
    # M5 contributes to C[0,0] with -1, C[0,1] with +1
    W[0][4] = -1.0 # C[0,0]
    W[1][4] = 1.0  # C[0,1]
    
    # M6 = (b+c)(g): U[1]=1, U[2]=1; V[2]=1
    U[1][5] = 1.0; U[2][5] = 1.0
    V[2][5] = 1.0
    # M6 contributes to C[1,1] with +1
    W[3][5] = 1.0  # C[1,1]
    
    # M7 = (a-c)(e+f): U[0]=1, U[2]=-1; V[0]=1, V[1]=1
    U[0][6] = 1.0; U[2][6] = -1.0
    V[0][6] = 1.0; V[1][6] = 1.0
    # M7 contributes to C[0,0] with +1
    W[0][6] = 1.0  # C[0,0]
    
    return {"U": U, "V": V, "W": W}

def _try_rank_r(r, n=2, m=2, p=6, max_iter=5000, seed=42):
    """Try to find a rank-r decomposition using random initialization + Newton-like refinement.
    
    Uses the structure: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rng = np.random.RandomState(seed)
    
    # Number of variables: 4*r + 4*r + 12*r = 20*r
    # Number of constraints: 2*2*6 = 24
    # For r >= 24/20 = 1.2, so r >= 2 is potentially feasible (but likely not)
    # Actually we need at least enough to satisfy 24 constraints.
    
    # Initialize with small random values
    U = rng.randn(n*m, r) * 0.1
    V = rng.randn(m*p, r) * 0.1
    W = rng.randn(p*n, r) * 0.1
    
    # Normalize so that the sum of products gives approximately 1
    # For each (i,j,k), we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Use a simple iterative approach: for each constraint, adjust variables
    # This is a system of 24 polynomial equations in 20*r unknowns
    
    # Better approach: use the fact that for small r, we can try to find solutions
    # by treating it as an optimization problem
    
    # Define the objective: sum of squares of (T[i,j,k] - 1)
    def objective(U, V, W):
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for rr in range(r):
                        val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    total += (val - 1.0) ** 2
        return total
    
    # Try multiple random initializations
    best_obj = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(100):
        U = rng.randn(n*m, r) * 0.5
        V = rng.randn(m*p, r) * 0.5
        W = rng.randn(p*n, r) * 0.5
        
        # Gradient-free local search (Nelder-Mead style on flattened params)
        # Flatten
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Simple coordinate descent / random perturbation
        obj = objective(U, V, W)
        
        for it in range(max_iter // 100):
            # Random perturbation
            delta = rng.randn(len(params)) * 0.01
            new_params = params + delta
            new_U = new_params[:n*m*r].reshape(n*m, r)
            new_V = new_params[n*m*r:(n*m+r)*r].reshape(m*p, r)
            new_W = new_params[(n*m+r)*r:].reshape(p*n, r)
            
            new_obj = objective(new_U, new_V, new_W)
            if new_obj < obj:
                obj = new_obj
                params = new_params
                U = new_U
                V = new_V
                W = new_W
        
        if obj < best_obj:
            best_obj = obj
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_obj < 1e-12:
            break
    
    if best_obj < 1e-9:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None

def construct():
    """Find the rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Start with the best known: schoolbook rank 24
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p  # 24
    
    # Try Strassen-based constructions to reduce rank
    # For 2x2 * 2x6, we can use Strassen for the 2x2 part and extend to 2x6
    
    # Strategy: decompose the 2x6 matrix B into three 2x2 blocks
    # B = [B1, B2, B3] where each Bi is 2x2
    # Then A*B = A*B1 + A*B2 + A*B3
    # Using Strassen for each: 3 * 7 = 21 multiplications
    # But we can share some terms since A is the same
    
    # Better: use the structure more carefully
    # A = [[a,b],[c,d]], B = [[e,f,g,h,i,j],[k,l,m,n,o,q]]
    # Result C = [[ae+bg+ci+dk+ei+gi+ki, af+bh+cj+dl+fj+hj+lj, ...], ...]
    
    # Actually, let's think about this differently.
    # The tensor for 2x2 * 2x6 has dimension 2*2*6 = 24.
    # We want to find the minimum rank.
    
    # Known results: For matrix multiplication <n,m,p>, the rank is at most n*m*p (schoolbook)
    # and can be reduced using Strassen-type methods.
    
    # For 2x2 * 2x6, one approach:
    # Split the 6 columns into three groups of 2: B = [B1, B2, B3]
    # A*B = [A*B1, A*B2, A*B3]
    # Each A*Bi uses 7 Strassen products, total 21.
    # But since A is shared, we might do better.
    
    # Let's construct a rank-21 decomposition:
    # For each block Bi (i=0,1,2), use Strassen's 7 products.
    # Total rank = 3*7 = 21.
    
    # Strassen for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1=(a+b)(e+f), M2=(c+d)e, M3=a(f-g), M4=d(h-g), M5=(a+d)h, M6=(b+c)g, M7=(a-c)(e+f)
    
    # For B with 6 columns: B = [[e0,f0,g0,h0,i0,j0],[k0,l0,m0,n0,o0,q0]]
    # Block 0: B0 = [[e0,f0],[k0,l0]]
    # Block 1: B1 = [[g0,h0],[m0,n0]]  
    # Block 2: B2 = [[i0,j0],[o0,q0]]
    
    rank_21 = 21
    U = [[0.0]*rank_21 for _ in range(n*m)]  # 4 rows
    V = [[0.0]*rank_21 for _ in range(m*p)]  # 12 rows
    W = [[0.0]*rank_21 for _ in range(p*n)]  # 12 rows
    
    # A indices: a=0, b=1, c=2, d=3
    # B indices: row0: e0=0,f0=1,g0=2,h0=3,i0=4,j0=5; row1: k0=6,l0=7,m0=8,n0=9,o0=10,q0=11
    # C indices: k*n+i where k=0..5, i=0..1
    # C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3, C[2,0]=4, C[2,1]=5, C[3,0]=6, C[3,1]=7, C[4,0]=8, C[4,1]=9, C[5,0]=10, C[5,1]=11
    
    # For each block, apply Strassen
    for block in range(3):
        base = block * 7
        # B block columns: 2*block, 2*block+1 (for row 0) and 2*block+6, 2*block+7 (for row 1)
        b0_col = 2*block      # e
        b1_col = 2*block+1    # f
        b2_col = 6+2*block    # g
        b3_col = 7+2*block    # h
        
        # C output columns for this block: 2*block, 2*block+1 (for rows 0 and 1)
        # C[k*n+i]: for k=2*block, i=0 -> 2*block; k=2*block, i=1 -> 2*block+1
        # k=2*block+1, i=0 -> 2*block+2; k=2*block+1, i=1 -> 2*block+3
        
        c00 = 2*block      # C[2*block, 0]
        c01 = 2*block+1    # C[2*block, 1]
        c10 = 2*block+2    # C[2*block+1, 0]
        c11 = 2*block+3    # C[2*block+1, 1]
        
        # M1 = (a+b)(e+f)
        U[0][base+0] = 1.0; U[1][base+0] = 1.0
        V[b0_col][base+0] = 1.0; V[b1_col][base+0] = 1.0
        W[c00][base+0] = 1.0; W[c11][base+0] = 1.0
        
        # M2 = (c+d)(e)
        U[2][base+1] = 1.0; U[3][base+1] = 1.0
        V[b0_col][base+1] = 1.0
        W[c10][base+1] = 1.0; W[c11][base+1] = -1.0
        
        # M3 = a(f-g)
        U[0][base+2] = 1.0
        V[b1_col][base+2] = 1.0; V[b2_col][base+2] = -1.0
        W[c01][base+2] = 1.0; W[c11][base+2] = 1.0
        
        # M4 = d(h-g)
        U[3][base+3] = 1.0
        V[b2_col][base+3] = -1.0; V[b3_col][base+3] = 1.0
        W[c00][base+3] = 1.0; W[c10][base+3] = 1.0
        
        # M5 = (a+d)(h)
        U[0][base+4] = 1.0; U[3][base+4] = 1.0
        V[b3_col][base+4] = 1.0
        W[c00][base+4] = -1.0; W[c01][base+4] = 1.0
        
        # M6 = (b+c)(g)
        U[1][base+5] = 1.0; U[2][base+5] = 1.0
        V[b2_col][base+5] = 1.0
        W[c11][base+5] = 1.0
        
        # M7 = (a-c)(e+f)
        U[0][base+6] = 1.0; U[2][base+6] = -1.0
        V[b0_col][base+6] = 1.0; V[b1_col][base+6] = 1.0
        W[c00][base+6] = 1.0
    
    # Verify the rank-21 decomposition
    if _verify(U, V, W, n, m, p):
        best_result = {"U": U, "V": V, "W": W}
        best_rank = 21
    else:
        # Fallback to schoolbook
        best_result = _schoolbook(n, m, p)
        best_rank = 24
    
    # Try to reduce further using optimization
    if time.time() < deadline - 20:
        # Try rank 20
        result = _try_rank_r(20, n, m, p, max_iter=2000, seed=123)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = 20
    
    if time.time() < deadline - 10:
        # Try rank 19
        result = _try_rank_r(19, n, m, p, max_iter=2000, seed=456)
        if result is not None:
            U_r, V_r, W_r = result
            if _verify(U_r, V_r, W_r, n, m, p):
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = 19
    
    return best_result
# EVOLVE-BLOCK-END
