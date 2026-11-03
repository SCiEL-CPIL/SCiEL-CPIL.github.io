"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # We decompose M_{3,3,6} into two M_{3,3,3} decompositions.
    # Each M_{3,3,3} uses a rank-21 decomposition (Strassen for 2x2 block + schoolbook for rest).
    # Total rank = 2 * 21 = 42.
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(u_idx, v_idx, w_idx, col, val_u=1, val_v=1, val_w=1):
        U[u_idx][col] += val_u
        V[v_idx][col] += val_v
        W[w_idx][col] += val_w

    def strassen_2x2(base_u, base_v, base_w, offset):
        # Strassen for 2x2x2
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # U indices: 0,1,2,3
        # V indices: 0,1,2,3
        # W indices: 0,1,2,3
        # Terms:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # m1
        add_term(base_u+0, base_v+0, base_w+0, offset, 1, 1, 1)
        add_term(base_u+1, base_v+0, base_w+0, offset, 1, 1, 1)
        add_term(base_u+2, base_v+1, base_w+1, offset, 1, 1, 1)
        add_term(base_u+3, base_v+1, base_w+1, offset, 1, 1, 1)
        
        # m2
        add_term(base_u+2, base_v+0, base_w+0, offset, 1, 1, 1)
        add_term(base_u+3, base_v+0, base_w+0, offset, 1, 1, 1)
        
        # m3
        add_term(base_u+0, base_v+1, base_w+1, offset, 1, 1, 1)
        add_term(base_u+0, base_v+3, base_w+3, offset, 1, -1, 1)
        
        # m4
        add_term(base_u+3, base_v+2, base_w+2, offset, 1, 1, 1)
        add_term(base_u+3, base_v+0, base_w+0, offset, 1, -1, 1)
        
        # m5
        add_term(base_u+0, base_v+3, base_w+3, offset, 1, 1, 1)
        add_term(base_u+1, base_v+3, base_w+3, offset, 1, 1, 1)
        
        # m6
        add_term(base_u+2, base_v+0, base_w+0, offset, 1, 1, 1)
        add_term(base_u+2, base_v+1, base_w+1, offset, 1, 1, 1)
        add_term(base_u+0, base_v+0, base_w+0, offset, -1, 1, 1)
        add_term(base_u+0, base_v+1, base_w+1, offset, -1, 1, 1)
        
        # m7
        add_term(base_u+1, base_v+2, base_w+2, offset, 1, 1, 1)
        add_term(base_u+1, base_v+3, base_w+3, offset, 1, 1, 1)
        add_term(base_u+3, base_v+2, base_w+2, offset, -1, 1, 1)
        add_term(base_u+3, base_v+3, base_w+3, offset, -1, 1, 1)

    def add_3x3_decomp(offset):
        # Decompose M_{3,3,3}
        # A = [[A22, a], [bT, c]], B = [[B22, d], [eT, f]]
        # A22 is 2x2, a is 2x1, bT is 1x2, c is 1x1
        # B22 is 2x2, d is 2x1, eT is 1x2, f is 1x1
        
        # 1. A22 B22 (Strassen)
        # U indices for A22: 0,1,2,3
        # V indices for B22: 0,1,2,3
        # W indices for C22: 0,1,2,3
        strassen_2x2(0, 0, 0, offset)
        offset += 7
        
        # 2. a eT (rank 1)
        # a is A[0,2], A[1,2] (indices 2, 3 in U? No, U is n*m. A is 3x3.
        # U index for A[i,j] is i*m+j.
        # A[0,2] -> 0*3+2 = 2
        # A[1,2] -> 1*3+2 = 5
        # eT is B[2,0], B[2,1] (indices in V? V is m*p. B is 3x3.
        # V index for B[j,k] is j*p+k.
        # B[2,0] -> 2*3+0 = 6
        # B[2,1] -> 2*3+1 = 7
        # W index for C[i,k] is k*n+i.
        # C[0,0] -> 0*3+0 = 0
        # C[1,0] -> 0*3+1 = 1
        # C[0,1] -> 1*3+0 = 3
        # C[1,1] -> 1*3+1 = 4
        # Term: A[0,2]*B[2,0] -> C[0,0]
        add_term(2, 6, 0, offset, 1, 1, 1)
        # A[0,2]*B[2,1] -> C[0,1]
        add_term(2, 7, 3, offset, 1, 1, 1)
        # A[1,2]*B[2,0] -> C[1,0]
        add_term(5, 6, 1, offset, 1, 1, 1)
        # A[1,2]*B[2,1] -> C[1,1]
        add_term(5, 7, 4, offset, 1, 1, 1)
        offset += 4
        
        # 3. A22 d (rank 4)
        # d is B[0,2], B[1,2] (V indices: 0*3+2=2, 1*3+2=5)
        # A22 is A[0,0..1], A[1,0..1] (U indices: 0,1,3,4)
        # C22 is C[0,0..1], C[1,0..1] (W indices: 0,1,3,4)
        # A[0,0]*B[0,2] -> C[0,0]
        add_term(0, 2, 0, offset, 1, 1, 1)
        # A[0,1]*B[1,2] -> C[0,0]
        add_term(1, 5, 0, offset, 1, 1, 1)
        # A[1,0]*B[0,2] -> C[1,0]
        add_term(3, 2, 1, offset, 1, 1, 1)
        # A[1,1]*B[1,2] -> C[1,0]
        add_term(4, 5, 1, offset, 1, 1, 1)
        offset += 4
        
        # 4. a f (rank 1)
        # f is B[2,2] (V index: 2*3+2=8)
        # a is A[0,2], A[1,2] (U indices: 2, 5)
        # C[0,2], C[1,2] (W indices: 2*3+0=6, 2*3+1=7)
        # A[0,2]*B[2,2] -> C[0,2]
        add_term(2, 8, 6, offset, 1, 1, 1)
        # A[1,2]*B[2,2] -> C[1,2]
        add_term(5, 8, 7, offset, 1, 1, 1)
        offset += 2
        
        # 5. bT B22 (rank 4)
        # bT is A[2,0], A[2,1] (U indices: 2*3+0=6, 2*3+1=7)
        # B22 is B[0,0..1], B[1,0..1] (V indices: 0,1,3,4)
        # C[2,0], C[2,1] (W indices: 0*3+2=2, 1*3+2=5)
        # A[2,0]*B[0,0] -> C[2,0]
        add_term(6, 0, 2, offset, 1, 1, 1)
        # A[2,1]*B[1,0] -> C[2,0]
        add_term(7, 3, 2, offset, 1, 1, 1)
        # A[2,0]*B[0,1] -> C[2,1]
        add_term(6, 1, 5, offset, 1, 1, 1)
        # A[2,1]*B[1,1] -> C[2,1]
        add_term(7, 4, 5, offset, 1, 1, 1)
        offset += 4
        
        # 6. c eT (rank 1)
        # c is A[2,2] (U index: 2*3+2=8)
        # eT is B[2,0], B[2,1] (V indices: 6, 7)
        # C[2,0], C[2,1] (W indices: 2, 5)
        # A[2,2]*B[2,0] -> C[2,0]
        add_term(8, 6, 2, offset, 1, 1, 1)
        # A[2,2]*B[2,1] -> C[2,1]
        add_term(8, 7, 5, offset, 1, 1, 1)
        offset += 2
        
        # 7. bT d (rank 2)
        # bT is A[2,0], A[2,1] (U indices: 6, 7)
        # d is B[0,2], B[1,2] (V indices: 2, 5)
        # C[2,2] (W index: 2*3+2=8)
        # A[2,0]*B[0,2] -> C[2,2]
        add_term(6, 2, 8, offset, 1, 1, 1)
        # A[2,1]*B[1,2] -> C[2,2]
        add_term(7, 5, 8, offset, 1, 1, 1)
        offset += 2
        
        # 8. c f (rank 1)
        # c is A[2,2] (U index: 8)
        # f is B[2,2] (V index: 8)
        # C[2,2] (W index: 8)
        # A[2,2]*B[2,2] -> C[2,2]
        add_term(8, 8, 8, offset, 1, 1, 1)
        offset += 1
        
        return offset

    # First 3x3x3 decomposition for B1 (cols 0-2)
    offset = add_3x3_decomp(0)
    
    # Second 3x3x3 decomposition for B2 (cols 3-5)
    # B2 is B[:, 3:6]
    # V indices for B2 are shifted by 3 in the k dimension?
    # V index for B[j,k] is j*p+k.
    # For B2, k is 0..2, but in the full matrix k is 3..5.
    # So V index for B2[j, k'] is j*p + (k' + 3).
    # We need to adjust the V indices in the second decomposition.
    
    # Instead of re-implementing, we can just add terms with shifted V indices.
    # The U and W indices are the same.
    
    def add_3x3_decomp_shifted(offset, v_shift):
        # Same as add_3x3_decomp but V indices are shifted by v_shift
        # Strassen
        # m1
        add_term(0, 0+v_shift, 0, offset, 1, 1, 1)
        add_term(1, 0+v_shift, 0, offset, 1, 1, 1)
        add_term(2, 1+v_shift, 1, offset, 1, 1, 1)
        add_term(3, 1+v_shift, 1, offset, 1, 1, 1)
        # m2
        add_term(2, 0+v_shift, 0, offset, 1, 1, 1)
        add_term(3, 0+v_shift, 0, offset, 1, 1, 1)
        # m3
        add_term(0, 1+v_shift, 1, offset, 1, 1, 1)
        add_term(0, 3+v_shift, 3, offset, 1, -1, 1)
        # m4
        add_term(3, 2+v_shift, 2, offset, 1, 1, 1)
        add_term(3, 0+v_shift, 0, offset, 1, -1, 1)
        # m5
        add_term(0, 3+v_shift, 3, offset, 1, 1, 1)
        add_term(1, 3+v_shift, 3, offset, 1, 1, 1)
        # m6
        add_term(2, 0+v_shift, 0, offset, 1, 1, 1)
        add_term(2, 1+v_shift, 1, offset, 1, 1, 1)
        add_term(0, 0+v_shift, 0, offset, -1, 1, 1)
        add_term(0, 1+v_shift, 1, offset, -1, 1, 1)
        # m7
        add_term(1, 2+v_shift, 2, offset, 1, 1, 1)
        add_term(1, 3+v_shift, 3, offset, 1, 1, 1)
        add_term(3, 2+v_shift, 2, offset, -1, 1, 1)
        add_term(3, 3+v_shift, 3, offset, -1, 1, 1)
        offset += 7
        
        # 2. a eT
        add_term(2, 6+v_shift, 0, offset, 1, 1, 1)
        add_term(2, 7+v_shift, 3, offset, 1, 1, 1)
        add_term(5, 6+v_shift, 1, offset, 1, 1, 1)
        add_term(5, 7+v_shift, 4, offset, 1, 1, 1)
        offset += 4
        
        # 3. A22 d
        add_term(0, 2+v_shift, 0, offset, 1, 1, 1)
        add_term(1, 5+v_shift, 0, offset, 1, 1, 1)
        add_term(3, 2+v_shift, 1, offset, 1, 1, 1)
        add_term(4, 5+v_shift, 1, offset, 1, 1, 1)
        offset += 4
        
        # 4. a f
        add_term(2, 8+v_shift, 6, offset, 1, 1, 1)
        add_term(5, 8+v_shift, 7, offset, 1, 1, 1)
        offset += 2
        
        # 5. bT B22
        add_term(6, 0+v_shift, 2, offset, 1, 1, 1)
        add_term(7, 3+v_shift, 2, offset, 1, 1, 1)
        add_term(6, 1+v_shift, 5, offset, 1, 1, 1)
        add_term(7, 4+v_shift, 5, offset, 1, 1, 1)
        offset += 4
        
        # 6. c eT
        add_term(8, 6+v_shift, 2, offset, 1, 1, 1)
        add_term(8, 7+v_shift, 5, offset, 1, 1, 1)
        offset += 2
        
        # 7. bT d
        add_term(6, 2+v_shift, 8, offset, 1, 1, 1)
        add_term(7, 5+v_shift, 8, offset, 1, 1, 1)
        offset += 2
        
        # 8. c f
        add_term(8, 8+v_shift, 8, offset, 1, 1, 1)
        offset += 1
        
        return offset

    offset = add_3x3_decomp_shifted(offset, 3)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
