import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    """
    Construct a rank-49 tensor decomposition for 4x4 matrix multiplication
    using the Kronecker product of the rank-7 Strassen decomposition for 2x2 MM.
    The output W follows the trace convention W[k*n + i, r].
    """
    n, m, p = 4, 4, 4
    
    # Strassen decomposition for 2x2 matrices
    # U is 4x7, V is 4x7, W is 4x7
    # For 2x2: A=[a,b;c,d], B=[e,f;g,h], C=[p,q;r,s]
    # Indices in U: a=0, b=1, c=2, d=3
    # Indices in V: e=0, f=1, g=2, h=3
    # Indices in W: p=0, q=1, r=2, s=3 (column-major: k*2+i)
    
    U_2x2 = np.zeros((4, 7))
    V_2x2 = np.zeros((4, 7))
    W_2x2 = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h) -> p, s
    U_2x2[0, 0] = 1; U_2x2[3, 0] = 1
    V_2x2[0, 0] = 1; V_2x2[3, 0] = 1
    W_2x2[0, 0] = 1; W_2x2[3, 0] = 1
    
    # M2 = (c+d)e -> r, s
    U_2x2[2, 1] = 1; U_2x2[3, 1] = 1
    V_2x2[0, 1] = 1
    W_2x2[2, 1] = 1; W_2x2[3, 1] = -1
    
    # M3 = a(f-h) -> q, s
    U_2x2[0, 2] = 1
    V_2x2[1, 2] = 1; V_2x2[3, 2] = -1
    W_2x2[1, 2] = 1; W_2x2[3, 2] = 1
    
    # M4 = d(g-e) -> p, r
    U_2x2[3, 3] = 1
    V_2x2[2, 3] = 1; V_2x2[0, 3] = -1
    W_2x2[0, 3] = 1; W_2x2[2, 3] = 1
    
    # M5 = (a+b)h -> p, q
    U_2x2[0, 4] = 1; U_2x2[1, 4] = 1
    V_2x2[3, 4] = 1
    W_2x2[0, 4] = -1; W_2x2[1, 4] = 1
    
    # M6 = (c-a)(e+f) -> s
    U_2x2[2, 5] = 1; U_2x2[0, 5] = -1
    V_2x2[0, 5] = 1; V_2x2[1, 5] = 1
    W_2x2[3, 5] = 1
    
    # M7 = (b-d)(g+h) -> p
    U_2x2[1, 6] = 1; U_2x2[3, 6] = -1
    V_2x2[2, 6] = 1; V_2x2[3, 6] = 1
    W_2x2[0, 6] = 1
    
    # Now we need to expand this to 4x4 using Kronecker product logic.
    # The 4x4 matrix multiplication tensor is the Kronecker product of two 2x2 tensors.
    # We need to map the indices correctly.
    #
    # Let's define the index mapping for 4x4.
    # U has shape (16, 49). Row index is i*4 + j (for A[i,j]).
    # V has shape (16, 49). Row index is j*4 + k (for B[j,k]).
    # W has shape (16, 49). Row index is k*4 + i (for C[k,i] in trace convention? 
    # Wait, the task says W[k*n+i, r]. n=4. So W index is k*4 + i.
    # Standard matrix product C = A*B has C[i,k] = sum_j A[i,j]B[j,k].
    # The task says "output factor follows the official trace convention W[k*n+i, r]".
    # This usually implies the tensor is defined such that C_{i,k} = sum_{r} U_{i,j,r} V_{j,k,r} W_{k,i,r}.
    # Let's verify the schoolbook implementation in the prompt.
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
    # So yes, W index is k*4+i.
    
    # The Kronecker product of the 2x2 decomposition gives us a 4x4 decomposition.
    # The rank will be 7*7 = 49.
    # We need to construct U, V, W of shape (16, 49).
    
    # The index in the rank dimension for the 4x4 case will be (r1, r2) where r1, r2 in [0..6].
    # Let rank_index = r1 * 7 + r2.
    
    U_4x4 = np.zeros((16, 49))
    V_4x4 = np.zeros((16, 49))
    W_4x4 = np.zeros((16, 49))
    
    for r1 in range(7):
        for r2 in range(7):
            rank_idx = r1 * 7 + r2
            
            # U: (i1, j1) from U_2x2[i1, j1, r1] and (i2, j2) from U_2x2[i2, j2, r2]
            # Combined index i = i1*2 + i2, j = j1*2 + j2
            # U_4x4[i*4+j, rank_idx] = U_2x2[i1, r1] * U_2x2[i2, r2]
            
            # V: (j1, k1) from V_2x2[j1, k1, r1] and (j2, k2) from V_2x2[j2, k2, r2]
            # Combined index j = j1*2 + j2, k = k1*2 + k2
            # V_4x4[j*4+k, rank_idx] = V_2x2[j1, r1] * V_2x2[j2, r2]
            
            # W: (k1, i1) from W_2x2[k1, i1, r1] and (k2, i2) from W_2x2[k2, i2, r2]
            # Combined index k = k1*2 + k2, i = i1*2 + i2
            # W_4x4[k*4+i, rank_idx] = W_2x2[k1, r1] * W_2x2[k2, r2]
            
            for i1 in range(2):
                for j1 in range(2):
                    u1 = U_2x2[i1, r1]
                    for i2 in range(2):
                        for j2 in range(2):
                            u2 = U_2x2[i2, r2]
                            u_val = u1 * u2
                            if u_val != 0:
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                U_4x4[i*4 + j, rank_idx] += u_val
            
            for j1 in range(2):
                for k1 in range(2):
                    v1 = V_2x2[j1, r1]
                    for j2 in range(2):
                        for k2 in range(2):
                            v2 = V_2x2[j2, r2]
                            v_val = v1 * v2
                            if v_val != 0:
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                V_4x4[j*4 + k, rank_idx] += v_val
            
            for k1 in range(2):
                for i1 in range(2):
                    w1 = W_2x2[k1, r1]
                    for k2 in range(2):
                        for i2 in range(2):
                            w2 = W_2x2[k2, r2]
                            w_val = w1 * w2
                            if w_val != 0:
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                W_4x4[k*4 + i, rank_idx] += w_val
    
    # Convert to lists as required by the interface
    U_list = U_4x4.tolist()
    V_list = V_4x4.tolist()
    W_list = W_4x4.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
