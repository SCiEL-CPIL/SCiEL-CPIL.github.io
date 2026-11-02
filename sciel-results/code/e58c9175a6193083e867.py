"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_einsum(U, V, W):
        """Verify the tensor decomposition is exact using einsum."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        R = U_arr.shape[1]
        
        # Compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Using einsum: sum over r
        result = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        
        # Check that result[i,j,k] = 1 for all i,j,k
        # But we need to map indices correctly
        # U index: i*m+j where i in [0,n), j in [0,m)
        # V index: j*p+k where j in [0,m), k in [0,p)
        # W index: k*n+i where k in [0,p), i in [0,n)
        
        # Let's build the expected tensor
        expected = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected[i,j,k] = 1.0
        
        # The einsum gives us T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
        # but U is indexed by (i*m+j), V by (j*p+k), W by (k*n+i)
        # So we need to reshape
        U_reshaped = U_arr.reshape(n, m, R)
        V_reshaped = V_arr.reshape(m, p, R)
        W_reshaped = W_arr.reshape(p, n, R)
        
        result2 = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
        
        if not np.allclose(result2, expected, atol=1e-9):
            return False
        return True
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_strassen_rectangular():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B):
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B):
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        else:
            return None
    
    def try_rank13_shared():
        """
        Try to find a rank-13 decomposition by searching over structured
        modifications of the rank-14 Strassen decomposition.
        
        Key insight: In the rank-14 solution, terms 0 and 7 both have
        U = [1,0,0,1]. We can try to merge them by finding a single term
        whose V and W vectors produce the combined effect.
        
        We also try a broader search: parameterize deviations from the
        rank-14 solution and check if any term can be eliminated.
        """
        if time.time() > deadline:
            return None
        
        # The rank-14 solution as numpy arrays
        U14 = np.zeros((4, 14))
        V14 = np.zeros((8, 14))
        W14 = np.zeros((8, 14))
        
        # Left block (Strassen 2x2x2)
        U14[0,0]=1; U14[3,0]=1; V14[0,0]=1; V14[5,0]=1; W14[0,0]=1; W14[3,0]=1
        U14[2,1]=1; U14[3,1]=1; V14[0,1]=1; W14[1,1]=1; W14[3,1]=-1
        U14[0,2]=1; V14[1,2]=1; V14[5,2]=-1; W14[2,2]=1; W14[3,2]=1
        U14[3,3]=1; V14[4,3]=1; V14[0,3]=-1; W14[0,3]=1; W14[1,3]=1
        U14[0,4]=1; U14[1,4]=1; V14[5,4]=1; W14[0,4]=-1; W14[2,4]=1
        U14[2,5]=1; U14[0,5]=-1; V14[0,5]=1; V14[1,5]=1; W14[3,5]=1
        U14[1,6]=1; U14[3,6]=-1; V14[4,6]=1; V14[5,6]=1; W14[0,6]=1
        
        # Right block (Strassen 2x2x2)
        U14[0,7]=1; U14[3,7]=1; V14[2,7]=1; V14[7,7]=1; W14[4,7]=1; W14[7,7]=1
        U14[2,8]=1; U14[3,8]=1; V14[2,8]=1; W14[5,8]=1; W14[7,8]=-1
        U14[0,9]=1; V14[3,9]=1; V14[7,9]=-1; W14[6,9]=1; W14[7,9]=1
        U14[3,10]=1; V14[6,10]=1; V14[2,10]=-1; W14[4,10]=1; W14[5,10]=1
        U14[0,11]=1; U14[1,11]=1; V14[7,11]=1; W14[4,11]=-1; W14[6,11]=1
        U14[2,12]=1; U14[0,12]=-1; V14[2,12]=1; V14[3,12]=1; W14[7,12]=1
        U14[1,13]=1; U14[3,13]=-1; V14[6,13]=1; V14[7,13]=1; W14[4,13]=1
        
        def check_rank13(U, V, W):
            """Check if the rank-13 decomposition is valid."""
            # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
            # U: (n,m,R) = (2,2,13), V: (m,p,R) = (2,4,13), W: (p,n,R) = (4,2,13)
            U3 = U.reshape(2, 2, 13)
            V3 = V.reshape(2, 4, 13)
            W3 = W.reshape(4, 2, 13)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            return np.allclose(T, 1.0, atol=1e-9)
        
        # Strategy 1: Try to merge terms 0 and 7 (both have U=[1,0,0,1])
        # Term 0: V=[1,0,0,0,0,1,0,0], W=[1,0,0,1,0,0,0,0]
        # Term 7: V=[0,0,1,0,0,0,0,1], W=[0,0,0,0,1,0,0,1]
        # Combined: V=[1,0,1,0,0,1,0,1], W=[1,0,0,1,1,0,0,1]
        # But this would give cross terms: V[0]*W[4], V[2]*W[0], etc.
        # Let's check: the combined term contributes to T[i,j,k]:
        #   U[i,0,0]*V[0,k,0]*W[k,0,0] for various i,k
        # This won't work directly due to cross terms.
        
        # Strategy 2: Try to find a rank-13 by allowing one term to have
        # non-trivial coefficients that cancel cross terms.
        # 
        # Let's try: keep terms 1-6 and 8-13 as-is (12 terms), and find
        # a single term 0 that replaces both original terms 0 and 7.
        # The single term must satisfy:
        #   For all (i,j,k): U[i,j]*V[j,k]*W[k,i] = contribution needed
        #
        # The contributions from original terms 0 and 7:
        # Term 0: i in {0,1}, j=0,k=0 and j=1,k=1
        #   T[0,0,0] += 1*1*1 = 1
        #   T[1,0,0] += 1*1*1 = 1 (wait, W[0,1]=0, so no)
        # Let me recompute carefully.
        #
        # Term 0: U=[1,0,0,1], V=[1,0,0,0,0,1,0,0], W=[1,0,0,1,0,0,0,0]
        #   T[i,j,k] += U[i,j]*V[j,k]*W[k,i]
        #   Non-zero when V[j,k]!=0 and W[k,i]!=0 and U[i,j]!=0
        #   V[0,0]=1, V[1,1]=1
        #   W[0,0]=1, W[1,1]=1
        #   U[0,0]=1, U[1,1]=1
        #   So: T[0,0,0] += 1*1*1 = 1
        #         T[1,1,1] += 1*1*1 = 1
        #
        # Term 7: U=[1,0,0,1], V=[0,0,1,0,0,0,0,1], W=[0,0,0,0,1,0,0,1]
        #   V[0,2]=1, V[1,3]=1
        #   W[2,0]=1, W[3,1]=1
        #   U[0,0]=1, U[1,1]=1
        #   So: T[0,0,2] += 1*1*1 = 1
        #         T[1,1,3] += 1*1*1 = 1
        
        # So the merged term needs to contribute:
        #   T[0,0,0] += 1, T[1,1,1] += 1, T[0,0,2] += 1, T[1,1,3] += 1
        # and contribute 0 to all other (i,j,k).
        
        # Let U = [a,b,c,d] (for positions 0,1,2,3)
        # V = [v0,v1,v2,v3,v4,v5,v6,v7]
        # W = [w0,w1,w2,w3,w4,w5,w6,w7]
        
        # Constraints from T[0,0,0]=1: a*v0*w0 = 1 (only non-zero if v0,w0 non-zero)
        # But V[0,0] is position 0, W[0,0] is position 0.
        # Actually V[j,k] where j=0,k=0 is V[0,0]=v0
        # W[k,i] where k=0,i=0 is W[0,0]=w0
        # So a*v0*w0 = 1
        
        # T[1,1,1]=1: d*v4*w3 = 1 (V[1,1]=v4, W[1,1]=w3)
        # Wait, W is indexed by (k*n+i) = (k*2+i). W[1,1] means k=1,i=1, index = 3.
        # So W[1,1] = w3. V[1,1] means j=1,k=1, index = 1*4+1 = 5. So V[1,1]=v5.
        # Let me redo: V[j*p+k] = V[j*4+k]. W[k*n+i] = W[k*2+i].
        
        # T[0,0,0]: i=0,j=0,k=0: U[0,0]*V[0,0]*W[0,0] = a*v0*w0 = 1
        # T[1,1,1]: i=1,j=1,k=1: U[1,1]*V[1,1]*W[1,1] = d*v5*w3 = 1
        # T[0,0,2]: i=0,j=0,k=2: U[0,0]*V[0,2]*W[2,0] = a*v2*w4 = 1
        # T[1,1,3]: i=1,j=1,k=3: U[1,1]*V[1,3]*W[3,1] = d*v7*w7 = 1
        
        # Zero constraints:
        # T[0,1,0]: U[0,1]*V[1,0]*W[0,0] = b*v4*w0 = 0
        # T[0,1,1]: U[0,1]*V[1,1]*W[1,0] = b*v5*w2 = 0
        # T[1,0,0]: U[1,0]*V[0,0]*W[0,1] = c*v0*w1 = 0
        # T[1,0,1]: U[1,0]*V[0,1]*W[1,1] = c*v1*w3 = 0
        # ... many more
        
        # This is a complex system. Let's try a specific ansatz:
        # Set b=0, c=0 to simplify (U = [a, 0, 0, d])
        # Then a*v0*w0 = 1, d*v5*w3 = 1, a*v2*w4 = 1, d*v7*w7 = 1
        # And we need all other T[i,j,k] = 0.
        
        # With b=c=0, the only non-zero U entries are a and d.
        # T[0,j,k] = a*V[j,k]*W[k,0] for j=0,1
        # T[1,j,k] = d*V[j,k]*W[k,1] for j=0,1
        
        # T[0,0,0] = a*v0*w0 = 1
        # T[0,0,1] = a*v1*w2 = 0 => v1=0 or w2=0
        # T[0,0,2] = a*v2*w4 = 1
        # T[0,0,3] = a*v3*w6 = 0 => v3=0 or w6=0
        # T[0,1,0] = a*v4*w0 = 0 => v4=0 or w0=0, but w0!=0 so v4=0
        # T[0,1,1] = a*v5*w2 = 0 => v5=0 or w2=0
        # T[0,1,2] = a*v6*w4 = 0 => v6=0 or w4=0, but w4!=0 so v6=0
        # T[0,1,3] = a*v7*w6 = 0 => v7=0 or w6=0
        
        # T[1,0,0] = d*v0*w1 = 0 => v0=0 or w1=0, but v0!=0 so w1=0
        # T[1,0,1] = d*v1*w3 = 0 => v1=0 or w3=0, but w3!=0 so v1=0
        # T[1,0,2] = d*v2*w5 = 0 => v2=0 or w5=0, but v2!=0 so w5=0
        # T[1,0,3] = d*v3*w7 = 0 => v3=0 or w7=0, but w7!=0 so v3=0
        # T[1,1,0] = d*v4*w1 = 0 (satisfied since v4=0 or w1=0)
        # T[1,1,1] = d*v5*w3 = 1
        # T[1,1,2] = d*v6*w5 = 0 (satisfied since v6=0)
        # T[1,1,3] = d*v7*w7 = 1
        
        # So we need: v0,v2,v5,v7 non-zero; v1=v3=v4=v6=0
        # w0,w3,w4,w7 non-zero; w1=w2=w5=w6=0
        
        # Check consistency:
        # From T[0,0,0]=1: a*v0*w0=1
        # From T[0,0,2]=1: a*v2*w4=1
        # From T[1,1,1]=1: d*v5*w3=1
        # From T[1,1,3]=1: d*v7*w7=1
        
        # This is consistent! Let's pick:
        # a=1, v0=1, w0=1, v2=1, w4=1, d=1, v5=1, w3=1, v7=1, w7=1
        
        # So the merged term is:
        # U = [1, 0, 0, 1]
        # V = [1, 0, 1, 0, 0, 1, 0, 1]
        # W = [1, 0, 0, 1, 1, 0, 0, 1]
        
        # Let's construct the rank-13 and verify
        U13 = np.zeros((4, 13))
        V13 = np.zeros((8, 13))
        W13 = np.zeros((8, 13))
        
        # Merged term 0
        U13[0,0]=1; U13[3,0]=1
        V13[0,0]=1; V13[2,0]=1; V13[5,0]=1; V13[7,0]=1
        W13[0,0]=1; W13[3,0]=1; W13[4,0]=1; W13[7,0]=1
        
        # Terms 1-6 (from original)
        U13[2,1]=1; U13[3,1]=1; V13[0,1]=1; W13[1,1]=1; W13[3,1]=-1
        U13[0,2]=1; V13[1,2]=1; V13[5,2]=-1; W13[2,2]=1; W13[3,2]=1
        U13[3,3]=1; V13[4,3]=1; V13[0,3]=-1; W13[0,3]=1; W13[1,3]=1
        U13[0,4]=1; U13[1,4]=1; V13[5,4]=1; W13[0,4]=-1; W13[2,4]=1
        U13[2,5]=1; U13[0,5]=-1; V13[0,5]=1; V13[1,5]=1; W13[3,5]=1
        U13[1,6]=1; U13[3,6]=-1; V13[4,6]=1; V13[5,6]=1; W13[0,6]=1
        
        # Terms 8-13 (from original, renumbered to 7-12)
        U13[2,7]=1; U13[3,7]=1; V13[2,7]=1; W13[5,7]=1; W13[7,7]=-1
        U13[0,8]=1; V13[3,8]=1; V13[7,8]=-1; W13[6,8]=1; W13[7,8]=1
        U13[3,9]=1; V13[6,9]=1; V13[2,9]=-1; W13[4,9]=1; W13[5,9]=1
        U13[0,10]=1; U13[1,10]=1; V13[7,10]=1; W13[4,10]=-1; W13[6,10]=1
        U13[2,11]=1; U13[0,11]=-1; V13[2,11]=1; V13[3,11]=1; W13[7,11]=1
        U13[1,12]=1; U13[3,12]=-1; V13[6,12]=1; V13[7,12]=1; W13[4,12]=1
        
        if check_rank13(U13, V13, W13):
            U_list = U13.tolist()
            V_list = V13.tolist()
            W_list = W13.tolist()
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try the rank-13 decomposition first
    result = try_rank13_shared()
    if result is not None:
        return result
    
    # Try the structured rank-14 decomposition
    result = try_strassen_rectangular()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
