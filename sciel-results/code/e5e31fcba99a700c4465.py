import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    def schoolbook_decomposition(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return (U, V, W), rank
    
    # Start with schoolbook as fallback
    best_result, best_rank = schoolbook_decomposition(n, m, p)
    
    # === TENSOR PRODUCT + STRASSEN APPROACH ===
    # M_{3,4,8} = M_{3,2,4} ⊗ M_{1,2,2}
    # M_{3,2,4} = M_{3,2,2} ⊗ M_{1,1,2}
    # M_{1,1,2}: rank 2
    # M_{3,2,2}: try to find low rank
    
    def tensor_product(U1, V1, W1, U2, V2, W2):
        """Compute tensor product of two decompositions.
        If T1 = U1⊗V1⊗W1 for M_{a,b,c} and T2 = U2⊗V2⊗W2 for M_{d,e,f},
        then T1⊗T2 is a decomposition of M_{ad,be,cf} with rank r1*r2.
        U = U1 ⊗ U2 (Kronecker), etc.
        """
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        U = np.kron(U1, U2)  # (a*d, r1*r2)
        V = np.kron(V1, V2)  # (b*e, r1*r2)
        W = np.kron(W1, W2)  # (c*f, r1*r2)
        return U, V, W
    
    def strassen_2x2():
        """Return rank-7 Strassen decomposition for M_{2,2,2}.
        U: (4,7), V: (4,7), W: (4,7)
        Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        # Strassen's 7 products for 2x2*2x2:
        # m1 = (a+b)(e+f)
        # m2 = (c+d)(e+f)  
        # m3 = a(h-f)
        # m4 = d(g-e)
        # m5 = (a+e)(h+f)
        # m6 = (c+a)(f-h)
        # m7 = (b+d)(g-c)
        # 
        # Result:
        # p00 = m1+m4-m5+m7
        # p01 = m5-m1+m2+m6
        # p10 = m1-m3+m4+m6
        # p11 = -m3+m2+m5+m7
        #
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p00,p01],[p10,p11]]
        #
        # U[i*2+j, c] for a=U[0,c], b=U[1,c], c=U[2,c], d=U[3,c]
        # V[j*2+k, c] for e=V[0,c], f=V[1,c], g=V[2,c], h=V[3,c]
        # W[k*2+i, c] for p00=W[0,c], p01=W[1,c], p10=W[2,c], p11=W[3,c]
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1 = (a+b)(e+f): U: a=1,b=1; V: e=1,f=1
        U[0,0]=1; U[1,0]=1; V[0,0]=1; V[1,0]=1
        W[0,0]=1  # p00 += m1
        W[1,0]=-1 # p01 -= m1
        W[2,0]=1  # p10 += m1
        # p11: no m1
        
        # m2 = (c+d)(e+f): U: c=1,d=1; V: e=1,f=1
        U[2,1]=1; U[3,1]=1; V[0,1]=1; V[1,1]=1
        W[1,1]=1  # p01 += m2
        W[3,1]=1  # p11 += m2
        
        # m3 = a(h-f): U: a=1; V: h=1,f=-1
        U[0,2]=1; V[3,2]=1; V[1,2]=-1
        W[2,2]=-1 # p10 -= m3
        W[3,2]=-1 # p11 -= m3
        
        # m4 = d(g-e): U: d=1; V: g=1,e=-1
        U[3,3]=1; V[2,3]=1; V[0,3]=-1
        W[0,3]=1  # p00 += m4
        W[2,3]=1  # p10 += m4
        
        # m5 = (a+e)(h+f): U: a=1; V: e=1,h=1,f=1
        U[0,4]=1; V[0,4]=1; V[3,4]=1; V[1,4]=1
        W[0,4]=-1 # p00 -= m5
        W[1,4]=1  # p01 += m5
        W[3,4]=1  # p11 += m5
        
        # m6 = (c+a)(f-h): U: c=1,a=1; V: f=1,h=-1
        U[2,5]=1; U[0,5]=1; V[1,5]=1; V[3,5]=-1
        W[1,5]=1  # p01 += m6
        W[2,5]=1  # p10 += m6
        
        # m7 = (b+d)(g-c): U: b=1,d=1; V: g=1,c=-1
        U[1,6]=1; U[3,6]=1; V[2,6]=1; V[2,6]=1  # wait, c is index 2 in U
        # U: b=U[1], d=U[3]; V: g=V[2], c=... 
        # Actually for m7=(b+d)(g-c): U has b=1,d=1; V has g=1,c=-1
        # But c in V context is the second row first col of B? No.
        # Let me reclarify: B = [[e,f],[g,h]], so V[0]=e, V[1]=f, V[2]=g, V[3]=h
        # m7 = (b+d)(g-c) where c is from A: A=[[a,b],[c,d]]
        # So U: b=1 (U[1]), d=1 (U[3]); V: g=1 (V[2]), and -c... but c is in U not V!
        # Wait, I think I made an error. Let me redo Strassen properly.
        
        # Standard Strassen:
        # m1 = (a+b)(e+f)
        # m2 = (c+d)(e+f)
        # m3 = a(h-f)
        # m4 = d(g-e)
        # m5 = (a+e)(h+f)
        # m6 = (c+a)(f-h)
        # m7 = (b+d)(g-c)
        
        # For m7: (b+d) from A, (g-c) from B... but c is from A!
        # Actually g and c: g is B[1,0], c is A[1,0]. So (g-c) mixes A and B?
        # No, that's not right either. Let me look this up properly.
        
        # Correct Strassen:
        # m1 = (a+b)(e+f)
        # m2 = (c+d)(e+f)
        # m3 = a(h-f)
        # m4 = d(g-e)
        # m5 = (a+e)(h+f)
        # m6 = (c+a)(f-h)
        # m7 = (b+d)(g-c)
        
        # Here all of a,b,c,d are from A and e,f,g,h are from B.
        # m7 = (b+d)(g-c): b,d from A (row 0 col 1, row 1 col 1), g,c from B (row 1 col 0, row 0 col 0)
        # Wait, that means m7 uses elements from both A and B in a mixed way? No...
        # (b+d) is a linear combo of A elements, (g-c) is... g is B[1,0] and c is A[1,0]?
        # That can't be right for a bilinear form.
        
        # I think the standard Strassen is:
        # m1 = (a+b)(e+f)
        # m2 = (c+d)(e+f)
        # m3 = a(h-f)
        # m4 = d(g-e)
        # m5 = (a+e)(h+f)
        # m6 = (c+a)(f-h)
        # m7 = (b+d)(g-c)
        
        # And the result is:
        # p00 = m1 + m4 - m5 + m7
        # p01 = m5 - m1 + m2 + m6
        # p10 = m1 - m3 + m4 + m6
        # p11 = -m3 + m2 + m5 + m7
        
        # Let me verify m7 = (b+d)(g-c):
        # b = A[0,1], d = A[1,1], g = B[1,0], c = A[1,0]
        # This mixes A and B! That's not a valid bilinear decomposition.
        
        # I think I'm confusing myself. The correct Strassen uses:
        # m7 = (b+d)(g-c) where c here refers to... let me just use a known correct version.
        
        # Actually, I recall now: the issue is that in Strassen, the "c" in m7=(b+d)(g-c) 
        # refers to the same c as in A=[[a,b],[c,d]]. So it IS mixing. But that's fine
        # because the product (b+d)(g-c) = bg - bc + dg - dc, and when summed with other terms,
        # the cross terms cancel appropriately.
        
        # For our tensor decomposition, U[i*2+j,c] picks out A elements, V[j*2+k,c] picks out B elements.
        # So for m7: U needs b and d (A[0,1] and A[1,1]), V needs g and c... but c is an A element!
        # This means m7 as stated doesn't fit the U⊗V⊗W form directly.
        
        # I think the correct Strassen formulas that work for tensor decomposition are:
        # m1 = (a+b)(e+f)
        # m2 = (c+d)(e+f)
        # m3 = a(h-f)
        # m4 = d(g-e)
        # m5 = (a+e)(h+f)
        # m6 = (c+a)(f-h)
        # m7 = (b+d)(g-c)
        
        # And these DO work because each mi is a product of a linear form in A and a linear form in B.
        # m7: (b+d) is linear in A, (g-c) is linear in B... wait, c is in A not B.
        
        # I'm going in circles. Let me just use a verified rank-7 decomposition.
        # The standard one has:
        # U (for A elements a,b,c,d):
        # col0: [1,1,0,0]  (a+b)
        # col1: [0,0,1,1]  (c+d)
        # col2: [1,0,0,0]  (a)
        # col3: [0,0,0,1]  (d)
        # col4: [1,0,0,0]  (a)
        # col5: [1,0,1,0]  (a+c)
        # col6: [0,1,0,1]  (b+d)
        #
        # V (for B elements e,f,g,h):
        # col0: [1,1,0,0]  (e+f)
        # col1: [1,1,0,0]  (e+f)
        # col2: [0,-1,0,1] (h-f)
        # col3: [-1,0,1,0] (g-e)
        # col4: [0,1,0,1]  (h+f)
        # col5: [0,1,-1,0] (f-h)
        # col6: [0,-1,1,0] (g-c)... NO, this should be (g-c) but c is not in B!
        
        # I see the issue. Let me just abandon the Strassen approach for now and use a 
        # more systematic search with the tensor product structure.
        
        return None
    
    # Let me try a different approach: direct search with structured initialization
    
    def try_rank_decomposition(n, m, p, target_rank, max_attempts=100, seed_base=42):
        """Try to find a rank-target_rank decomposition."""
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline - 5:
                return None
            
            seed = seed_base + attempt * 7919 + target_rank
            np.random.seed(seed)
            
            # Generate U and V with small integer/half-integer entries
            U = np.random.randint(-3, 4, size=(n*m, target_rank)).astype(float)
            V = np.random.randint(-3, 4, size=(m*p, target_rank)).astype(float)
            
            # Compute M[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            M = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for c in range(target_rank):
                        M[i, k, c] = sum(U[i*m+j, c] * V[j*p+k, c] for j in range(m))
            
            # Solve for W: W[k*n+i, c] such that sum_c M[i,k,c] * W[k*n+i, c] = 1
            W = np.zeros((p*n, target_rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    mv = M[i, k, :]
                    norm_sq = np.dot(mv, mv)
                    if norm_sq < 1e-10:
                        valid = False
                        break
                    W[k*n+i, :] = mv / norm_sq
            if not valid:
                continue
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_decomposition(U_r, V_r, W_r, target_rank):
                return (U_r, V_r, W_r)
        
        return None
    
    # Try structured tensor product approaches first
    
    # Approach 1: M_{3,4,8} = M_{3,2,4} ⊗ M_{1,2,2}
    # M_{1,2,2}: 1x2 * 2x2 -> 1x2, rank 4 (schoolbook)
    # M_{3,2,4}: 3x2 * 2x4 -> 3x4, schoolbook rank 24
    # If we can get M_{3,2,4} to rank 12, total = 12*4 = 48
    
    # Approach 2: M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # M_{1,1,2}: rank 2
    # M_{3,4,4}: 3x4 * 4x4 -> 3x4, schoolbook rank 48
    # If M_{3,4,4} has rank 24, total = 24*2 = 48
    
    # Approach 3: M_{3,4,8} = M_{3,4,2} ⊗ M_{1,1,4}
    # M_{1,1,4}: rank 4
    # M_{3,4,2}: 3x4 * 4x2 -> 3x2, schoolbook rank 24
    # Total = 24*4 = 96 (no savings)
    
    # Let's try to find good decompositions for sub-problems
    
    def find_sub_decomposition(n, m, p, target_rank, max_attempts=50, seed_base=0):
        """Find a decomposition for M_{n,m,p} with given rank."""
        return try_rank_decomposition(n, m, p, target_rank, max_attempts, seed_base)
    
    # Try M_{3,4,4} with various ranks
    # Schoolbook for M_{3,4,4} is 48
    
    best_344 = None
    best_344_rank = 48
    for r in range(47, 20, -1):
        if time.time() - start_time > deadline - 20:
            break
        result = find_sub_decomposition(3, 4, 4, r, max_attempts=30, seed_base=1000)
        if result is not None:
            best_344 = result
            best_344_rank = r
            break
    
    if best_344 is not None:
        # Combine with M_{1,1,2} (rank 2)
        U_344, V_344, W_344 = best_344
        # M_{1,1,2}: U=[[1,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]
        U_112 = np.array([[1.0, 1.0]])
        V_112 = np.array([[1.0, 0.0], [0.0, 1.0]])
        W_112 = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        U_total, V_total, W_total = tensor_product(U_344, V_344, W_344, U_112, V_112, W_112)
        
        # Verify shapes
        # U_344: (12, r), U_112: (1, 2) -> U_total: (12, 2r)
        # V_344: (16, r), V_112: (2, 2) -> V_total: (32, 2r)
        # W_344: (12, r), W_112: (2, 2) -> W_total: (24, 2r)
        # Expected: U: (12, rank), V: (32, rank), W: (24, rank)
        
        if U_total.shape == (12, 2*best_344_rank) and V_total.shape == (32, 2*best_344_rank) and W_total.shape == (24, 2*best_344_rank):
            if verify_decomposition(U_total, V_total, W_total, 2*best_344_rank):
                total_rank = 2 * best_344_rank
                if total_rank < best_rank:
                    best_result = (U_total, V_total, W_total)
                    best_rank = total_rank
    
    # Try M_{3,2,4} with various ranks, combine with M_{1,2,2}
    best_324 = None
    best_324_rank = 24
    for r in range(23, 10, -1):
        if time.time() - start_time > deadline - 15:
            break
        result = find_sub_decomposition(3, 2, 4, r, max_attempts=30, seed_base=2000)
        if result is not None:
            best_324 = result
            best_324_rank = r
            break
    
    if best_324 is not None:
        U_324, V_324, W_324 = best_324
        # M_{1,2,2}: 1x2 * 2x2 -> 1x2, schoolbook rank 4
        # U: (2,4), V: (4,4), W: (2,4)
        U_122 = np.zeros((2, 4))
        V_122 = np.zeros((4, 4))
        W_122 = np.zeros((2, 4))
        # T[i,j,k] = 1 for (0,0,0) and (0,1,1)
        U_122[0,0] = 1; V_122[0,0] = 1; W_122[0,0] = 1
        U_122[0,1] = 1; V_122[2,1] = 1; W_122[1,1] = 1
        U_122[1,2] = 1; V_122[1,2] = 1; W_122[0,2] = 1
        U_122[1,3] = 1; V_122[3,3] = 1; W_122[1,3] = 1
        
        U_total, V_total, W_total = tensor_product(U_324, V_324, W_324, U_122, V_122, W_122)
        
        # U_324: (6, r), U_122: (2, 4) -> U_total: (12, 4r)
        # V_324: (8, r), V_122: (4, 4) -> V_total: (32, 4r)
        # W_324: (12, r), W_122: (2, 4) -> W_total: (24, 4r)
        
        if U_total.shape == (12, 4*best_324_rank) and V_total.shape == (32, 4*best_324_rank) and W_total.shape == (24, 4*best_324_rank):
            if verify_decomposition(U_total, V_total, W_total, 4*best_324_rank):
                total_rank = 4 * best_324_rank
                if total_rank < best_rank:
                    best_result = (U_total, V_total, W_total)
                    best_rank = total_rank
    
    # Direct search on M_{3,4,8}
    search_ranks = list(range(best_rank - 1, max(30, best_rank - 40), -1))
    
    for target_rank in search_ranks:
        if time.time() - start_time > deadline - 5:
            break
        
        result = try_rank_decomposition(n, m, p, target_rank, max_attempts=50, seed_base=5000)
        if result is not None:
            best_result = result
            best_rank = target_rank
            search_ranks = list(range(target_rank - 1, max(30, target_rank - 30), -1))
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
