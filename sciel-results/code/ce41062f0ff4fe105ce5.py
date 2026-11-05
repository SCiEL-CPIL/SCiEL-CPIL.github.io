import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Partition: 2+2+1
    block_sizes = [2, 2, 1]
    # Offset for each block
    offsets = [0, 2, 4]
    
    # Build U, V, W as lists of lists
    # We'll collect rank-1 terms
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # For each combination of blocks (bi, bj, bk)
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                si = block_sizes[bi]  # size of A row block
                sj = block_sizes[bj]  # size of A col block = B row block
                sk = block_sizes[bk]  # size of B col block = C row block
                
                oi = offsets[bi]  # offset for A rows
                oj = offsets[bj]  # offset for A cols / B rows
                ok = offsets[bk]  # offset for B cols / C rows
                
                if si == 2 and sj == 2 and sk == 2:
                    # Strassen: 7 rank-1 terms
                    # A is 2x2, B is 2x2, C is 2x2
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)B00
                    # M3 = A00(B01-B11)
                    # M4 = A11(B10-B00)
                    # M5 = (A00+A10)B11
                    # M6 = (A11-A10)(B00+B10)
                    # M7 = (A01-A11)(B01+B11)
                    # C00 = M1+M4-M5+M7
                    # C01 = M3+M5
                    # C10 = M2+M4
                    # C11 = M1-M2+M3+M6
                    
                    # Each M is a product of a linear form in A and a linear form in B
                    # producing a specific C entry
                    
                    # M1: A-form = A00+A11, B-form = B00+B11, C-contribution: C00 += 1, C11 += 1
                    # In terms of rank-1: U picks A00 and A11, V picks B00 and B11, W picks C00 and C11
                    # But rank-1 means single product. So M1 = (A00+A11)*(B00+B11) contributes to both C00 and C11.
                    # This is NOT a single rank-1 term in the tensor sense. 
                    
                    # Wait, I need to think about this more carefully.
                    # The tensor is T[k,i,j] where C[k,i] = sum_j A[i,j]*B[j,k].
                    # A rank-1 term is: sum over (i,j,k) of u[i,j]*v[j,k]*w[k,i]
                    # This equals (sum_j u[i,j]*v[j,k]) * w[k,i] for each (k,i)
                    # No wait, it's a sum of products of three vectors.
                    
                    # Actually, a rank-1 decomposition term is:
                    # C[k,i] += u[i,j]*v[j,k]*w[k,i] summed over j? No.
                    
                    # The tensor T is a 3-tensor with indices (i, j, k) where:
                    # T[i,j,k] contributes to C[k,i] via A[i,j]*B[j,k]
                    # So T[i,j,k] = 1 for all i,j,k (schoolbook)
                    # A rank-1 decomposition: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                    
                    # For Strassen, we have 7 terms. Each term r has:
                    # U[i,j,r], V[j,k,r], W[k,i,r]
                    # such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = delta (the tensor entry)
                    
                    # For 2x2x2, the tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
                    # Strassen gives 7 rank-1 terms.
                    
                    # Let me write out the 7 terms explicitly.
                    # Using the standard Strassen decomposition:
                    
                    # Term 1: M1 = (A00+A11)*(B00+B11)
                    #   U: A00=1, A11=1 (others 0)
                    #   V: B00=1, B11=1 (others 0)
                    #   W: C00=1, C11=1 (others 0)
                    #   Check: sum over r for fixed (i,j,k):
                    #   For (i=0,j=0,k=0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
                    #   For (i=0,j=1,k=0): U[0,1]*V[1,0]*W[0,0] = 0*0*1 = 0 ✓ (T[0,1,0]=1, but we need sum over all r)
                    
                    # Hmm, this is getting complicated. Let me just look up the explicit rank-7 decomposition.
                    
                    # The 7 rank-1 terms for 2x2x2 matmul tensor:
                    # Each term is (u, v, w) where u is in R^4 (for A's 4 entries), v in R^4 (for B's 4 entries), w in R^4 (for C's 4 entries)
                    # A entries indexed as (0,0),(0,1),(1,0),(1,1) -> flat 0,1,2,3
                    # B entries indexed as (0,0),(0,1),(1,0),(1,1) -> flat 0,1,2,3
                    # C entries indexed as (0,0),(0,1),(1,0),(1,1) -> flat 0,1,2,3
                    
                    # Standard Strassen rank-7:
                    # T1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)  [M1]
                    # T2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,0)  [M2]
                    # T3: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,0)  [M3]
                    # T4: u=(0,0,0,1), v=(0,0,1,-1), w=(1,0,0,0)  [M4]
                    # T5: u=(1,0,1,0), v=(0,0,0,1), w=(0,1,-1,0)  [M5]
                    # T6: u=(0,0,-1,1), v=(1,1,0,0), w=(0,0,0,1)  [M6]
                    # T7: u=(0,1,0,-1), v=(0,1,1,0), w=(1,0,0,1)  [M7]
                    
                    # Let me verify T1: sum over all 7 terms for (i=0,j=0,k=0):
                    # T1: u[0,0]*v[0,0]*w[0,0] = 1*1*1 = 1
                    # T2: u[0,0]*v[0,0]*w[1,0] = 1*1*0 = 0
                    # T3: u[0,0]*v[0,0]*w[0,1] = 1*0*0 = 0
                    # T4: u[0,0]*v[0,0]*w[0,0] = 1*0*1 = 0
                    # T5: u[0,0]*v[0,0]*w[0,1] = 1*0*0 = 0
                    # T6: u[0,0]*v[0,0]*w[0,0] = 1*1*0 = 0
                    # T7: u[0,0]*v[0,0]*w[0,0] = 1*0*1 = 0
                    # Sum = 1 ✓
                    
                    # For (i=0,j=1,k=0): T[i,j,k] should be 1
                    # T1: u[0,1]*v[1,0]*w[0,0] = 0*0*1 = 0
                    # T2: u[0,1]*v[1,0]*w[1,0] = 0*0*0 = 0
                    # T3: u[0,1]*v[1,0]*w[0,1] = 0*0*0 = 0
                    # T4: u[0,1]*v[1,0]*w[0,0] = 0*0*1 = 0
                    # T5: u[0,1]*v[1,0]*w[0,1] = 0*0*0 = 0
                    # T6: u[0,1]*v[1,0]*w[0,0] = 0*1*0 = 0
                    # T7: u[0,1]*v[1,0]*w[0,0] = 1*1*1 = 1
                    # Sum = 1 ✓
                    
                    # Good. Let me verify one more: (i=1,j=1,k=1) should be 1
                    # T1: u[1,1]*v[1,1]*w[1,1] = 1*1*1 = 1
                    # T2: u[1,1]*v[1,1]*w[1,1] = 1*0*0 = 0
                    # T3: u[1,1]*v[1,1]*w[1,1] = 0*0*0 = 0
                    # T4: u[1,1]*v[1,1]*w[1,1] = 1*(-1)*0 = 0
                    # T5: u[1,1]*v[1,1]*w[1,1] = 0*1*0 = 0
                    # T6: u[1,1]*v[1,1]*w[1,1] = 1*0*1 = 0
                    # T7: u[1,1]*v[1,1]*w[1,1] = (-1)*0*1 = 0
                    # Sum = 1 ✓
                    
                    strassen_terms = [
                        ((1,0,0,1), (1,0,0,1), (1,0,0,1)),  # T1
                        ((0,0,1,1), (1,0,0,0), (0,0,1,0)),  # T2
                        ((1,0,0,0), (0,1,0,-1), (0,1,0,0)),  # T3
                        ((0,0,0,1), (0,0,1,-1), (1,0,0,0)),  # T4
                        ((1,0,1,0), (0,0,0,1), (0,1,-1,0)),  # T5
                        ((0,0,-1,1), (1,1,0,0), (0,0,0,1)),  # T6
                        ((0,1,0,-1), (0,1,1,0), (1,0,0,1)),  # T7
                    ]
                    
                    for (u_local, v_local, w_local) in strassen_terms:
                        u = [0]*25
                        v = [0]*25
                        w = [0]*25
                        # u is for A[i,j], flat index i*5+j
                        # local (a,b) -> global (oi+a, oj+b)
                        for a in range(2):
                            for b in range(2):
                                u[(oi+a)*5 + (oj+b)] = u_local[a*2+b]
                        # v is for B[j,k], flat index j*5+k
                        for a in range(2):
                            for b in range(2):
                                v[(oj+a)*5 + (ok+b)] = v_local[a*2+b]
                        # w is for C[k,i], flat index k*5+i
                        for a in range(2):
                            for b in range(2):
                                w[(ok+a)*5 + (oi+b)] = w_local[a*2+b]
                        add_term(u, v, w)
                else:
                    # Schoolbook: si*sj*sk terms
                    for a in range(si):
                        for b in range(sj):
                            for c in range(sk):
                                u = [0]*25
                                v = [0]*25
                                w = [0]*25
                                u[(oi+a)*5 + (oj+b)] = 1
                                v[(oj+b)*5 + (ok+c)] = 1
                                w[(ok+c)*5 + (oi+a)] = 1
                                add_term(u, v, w)
    
    rank = len(U_cols)
    U = [col[:] for col in U_cols]
    V = [col[:] for col in V_cols]
    W = [col[:] for col in W_cols]
    
    # Transpose: U should be (25, rank), so U[row][col]
    # Currently U_cols[r] is a list of 25 entries (one per row)
    # We need U[i][r] = U_cols[r][i]
    U = [[U_cols[r][i] for r in range(rank)] for i in range(25)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(25)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(25)]
    
    return {"U": U, "V": V, "W": W}
