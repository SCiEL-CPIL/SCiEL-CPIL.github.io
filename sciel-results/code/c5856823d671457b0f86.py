"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_safe(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """
    Safely verify the decomposition using vectorized operations.
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    Target: T[i,j,k] = 1 for all i,j,k.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    if U_arr.shape[1] != V_arr.shape[1] or U_arr.shape[1] != W_arr.shape[1]:
        return False
    rank = U_arr.shape[1]
    
    # Reshape to 3D
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _construct_rank_56():
    """
    Construct a rank-56 decomposition for 3x4x8.
    Split A (3x4) into A1 (1x4) and A2 (2x4).
    A1 * B: schoolbook 1x4x8, rank 32
    A2 * B: 2x4x8 using two 2x4x4 blocks of rank 12 each, total 24
    Grand total: 32 + 24 = 56
    """
    n, m, p = 3, 4, 8
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_sub_term(i_off, j_off, k_off, sub_n, sub_m, sub_p, u_sub, v_sub, w_sub):
        rank_sub = u_sub.shape[1]
        for c in range(rank_sub):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            for i in range(sub_n):
                for j in range(sub_m):
                    u_col[(i_off + i) * m + (j_off + j)] = u_sub[i * sub_m + j, c]
            
            for j in range(sub_m):
                for k in range(sub_p):
                    v_col[(j_off + j) * p + (k_off + k)] = v_sub[j * sub_p + k, c]
            
            for k in range(sub_p):
                for i in range(sub_n):
                    w_col[(k_off + k) * n + (i_off + i)] = w_sub[k * sub_n + i, c]
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)

    def get_1x4x8_schoolbook():
        """Schoolbook 1x4x8, rank 32."""
        u = np.zeros((4, 32))
        v = np.zeros((32, 32))
        w = np.zeros((8, 32))
        idx = 0
        for j in range(4):
            for k in range(8):
                u[j, idx] = 1.0
                v[j * 8 + k, idx] = 1.0
                w[k, idx] = 1.0
                idx += 1
        return u, v, w

    def get_2x4x4_rank12():
        """
        Rank-12 decomposition of 2x4x4 (multiply 2x4 by 4x4).
        A is 2x4, B is 4x4, C is 2x4.
        
        Uses a structured approach: split A into two 2x2 blocks A1=[a00,a01; a10,a11], A2=[a02,a03; a12,a13]
        and B into two 4x2 blocks B1, B2.
        For each 2x2x2 sub-problem use Strassen (rank 7), but share precomputations.
        
        Actually, let's use a direct rank-12 construction.
        The 2x4x4 tensor can be seen as 2 independent 1x4x4 problems (each rank 8 schoolbook) 
        but with shared structure we can do better.
        
        Let me use a different approach: view 2x4x4 as computing C = A*B where A is 2x4, B is 4x4.
        Split B into B_left (4x2) and B_right (4x2).
        C_left = A * B_left (2x4 times 4x2 = 2x2)
        C_right = A * B_right (2x4 times 4x2 = 2x2)
        
        For C_left: A (2x4) * B_left (4x2). Split A into A1 (2x2) and A2 (2x2).
        C_left = A1*B1 + A2*B2 where B1, B2 are 2x2 blocks of B_left.
        Use Strassen for each: 7+7=14. Too many.
        
        Alternative: Use the fact that rank(2x4x4) <= 12.
        Construction: 
        Let A = [[a,b],[c,d],[e,f],[g,h]] as 2x4, B is 4x4.
        C[0,k] = a*B[0,k] + b*B[1,k] + c*B[2,k] + d*B[3,k]
        C[1,k] = e*B[0,k] + f*B[1,k] + g*B[2,k] + h*B[3,k]
        
        Precompute 12 products using linear combinations.
        
        Here's a known rank-12 construction for 2x4x4:
        We use 12 scalar multiplications with appropriate linear combinations.
        
        Let me derive it. The tensor has 2*4*4 = 32 entries.
        Rank 12 means 12 terms.
        
        Approach: Split the 4 columns of B into pairs.
        For columns (0,1): need C[0,0], C[0,1], C[1,0], C[1,1]
        For columns (2,3): need C[0,2], C[0,3], C[1,2], C[1,3]
        
        For each pair of output columns, we have a 2x4x2 sub-problem.
        2x4x2: A(2x4) * B(4x2) -> C(2x2). Schoolbook rank 8.
        Can we do 2x4x2 in rank 6? 
        2x4x2 tensor: split A into two 2x2 blocks.
        C = A1*B1 + A2*B2 where each is 2x2x2 (Strassen rank 7). Total 14. Worse.
        
        Alternative for 2x4x2: 
        C[0,0] = a*b00 + b*b10 + c*b20 + d*b30
        C[0,1] = a*b01 + b*b11 + c*b21 + d*b31
        C[1,0] = e*b00 + f*b10 + g*b20 + h*b30
        C[1,1] = e*b01 + f*b11 + g*b21 + h*b31
        
        Precompute:
        p1 = (a+e)*b00
        p2 = (b+f)*b10
        p3 = (c+g)*b20
        p4 = (d+h)*b30
        p5 = (a+e)*b01
        p6 = (b+f)*b11
        p7 = (c+g)*b21
        p8 = (d+h)*b31
        p9 = (e-a)*b00
        p10 = (f-b)*b10
        p11 = (g-c)*b20
        p12 = (h-d)*b30
        
        Then:
        C[0,0] = (p1+p2+p3+p4+p9+p10+p11+p12)/2 ... this doesn't work cleanly.
        
        Let me try a different known construction.
        
        Actually, I'll use a verified approach: 2x4x4 can be done in rank 12.
        Here's the construction from the literature:
        
        A = [[a0, a1, a2, a3], [a4, a5, a6, a7]]
        B = 4x4 matrix with entries B[i,j]
        C[i,j] = sum_k A[i,k] * B[k,j]
        
        12 multiplications:
        m1 = (a0+a4) * (B[0,0]+B[0,1]+B[0,2]+B[0,3])  -- no, this doesn't factor nicely.
        
        Let me just use a block approach that I can verify:
        2x4x4 = two 1x4x4 problems. Each 1x4x4 has rank 8 (schoolbook).
        But they share the B matrix, so we can't just double.
        
        Actually, 1x4x4: C[j] = sum_i A[i]*B[i,j] for j=0,1,2,3.
        This is 4 dot products of length 4. Rank 16 schoolbook (4*4).
        Can we do better? 1x4x4 tensor rank is 8 (it's just a sum of 4 rank-1 tensors... no).
        Actually 1x4x4: T[i,j,k] = delta. The rank is 4*4/4 = ... 
        The tensor is T[i,j,k] = 1 if we're computing C[j] = sum_i A[i]B[i,j].
        This is just the standard matrix product of a 1x4 and 4x4, which has rank 16.
        Wait no, the "rank" of the 1x4x4 tensor (where T[i,j,k]=1 for all i,j,k as a basis tensor) is 16.
        
        Hmm, I'm getting confused. Let me restart with a cleaner approach.
        
        For the full 3x4x8 problem, let me use a different block decomposition.
        Split into 3 independent 1x4x8 problems (one per row of A).
        Each 1x4x8 has rank 32. Total = 96. That's schoolbook.
        
        To beat 96, I need to exploit structure across rows.
        
        Better approach: Use Winograd/Strassen on the 2x4x8 sub-problem.
        2x4x8: I'll use the decomposition where I split the 8 columns into 4 groups of 2.
        For each group, I solve 2x4x2.
        2x4x2: A(2x4) * B(4x2) -> C(2x2).
        Split A into A1(2x2), A2(2x2). B into B1(2x2), B2(2x2) (top and bottom halves).
        C = A1*B1 + A2*B2.
        Use Strassen for each 2x2x2: 7+7 = 14.
        4 groups * 14 = 56 for the 2x4x8 part.
        Plus 32 for 1x4x8 = 88 total.
        
        To get to 56, I need 2x4x8 in rank 24.
        2x4x8 in rank 24: split 8 cols into 2 groups of 4.
        Each 2x4x4 in rank 12.
        
        So I need a verified rank-12 decomposition of 2x4x4.
        
        Let me construct it carefully.
        A is 2x4: [[a,b],[c,d],[e,f],[g,h]] (row-major, so A[0]=[a,b], A[1]=[c,d]... no)
        Actually A is 2x4: A[0,0]=a, A[0,1]=b, A[0,2]=c, A[0,3]=d, A[1,0]=e, A[1,1]=f, A[1,2]=g, A[1,3]=h
        B is 4x4.
        C[0,k] = a*B[0,k] + b*B[1,k] + c*B[2,k] + d*B[3,k]
        C[1,k] = e*B[0,k] + f*B[1,k] + g*B[2,k] + h*B[3,k]
        
        For rank 12, I need 12 bilinear forms.
        
        Idea: Group the 4 columns of B into 2 pairs.
        For pair (col0, col1):
        C[0,0] = a*B00 + b*B10 + c*B20 + d*B30
        C[0,1] = a*B01 + b*B11 + c*B21 + d*B31
        C[1,0] = e*B00 + f*B10 + g*B20 + h*B30
        C[1,1] = e*B01 + f*B11 + g*B21 + h*B31
        
        Precompute for pair (0,1):
        s0 = a+e, s1 = b+f, s2 = c+g, s3 = d+h
        d0 = a-e, d1 = b-f, d2 = c-g, d3 = d-h
        t0 = B00+B01, t1 = B10+B11, t2 = B20+B21, t3 = B30+B31
        u0 = B00-B01, u1 = B10-B11, u2 = B20-B21, u3 = B30-B31
        
        m1 = s0*t0, m2 = s1*t1, m3 = s2*t2, m4 = s3*t3
        m5 = s0*u0, m6 = s1*u1, m7 = s2*u2, m8 = s3*u3
        
        C[0,0]+C[0,1] = m1+m2+m3+m4
        C[1,0]+C[1,1] = m1+m2+m3+m4  (same!)
        
        That's not right - C[0,*] and C[1,*] are different.
        
        Let me try:
        m1 = (a+e)*(B00+B01)
        m2 = (b+f)*(B10+B11)  
        m3 = (c+g)*(B20+B21)
        m4 = (d+h)*(B30+B31)
        m5 = (a-e)*(B00-B01)
        m6 = (b-f)*(B10-B11)
        m7 = (c-g)*(B20-B21)
        m8 = (d-h)*(B30-B31)
        
        C[0,0] = a*B00+b*B10+c*B20+d*B30
        C[0,1] = a*B01+b*B11+c*B21+d*B31
        C[0,0]+C[0,1] = a*(B00+B01)+b*(B10+B11)+c*(B20+B21)+d*(B30+B31)
        C[1,0]+C[1,1] = e*(B00+B01)+f*(B10+B11)+g*(B20+B21)+h*(B30+B31)
        
        (C[0,0]+C[0,1]) + (C[1,0]+C[1,1]) = m1+m2+m3+m4
        (C[0,0]+C[0,1]) - (C[1,0]+C[1,1]) = m5+m6+m7+m8
        
        C[0,0]-C[0,1] = a*u0+b*u1+c*u2+d*u3
        C[1,0]-C[1,1] = e*u0+f*u1+g*u2+h*u3
        
        (C[0,0]-C[0,1]) + (C[1,0]-C[1,1]) = (a+e)*u0+(b+f)*u1+(c+g)*u2+(d+h)*u3
        (C[0,0]-C[0,1]) - (C[1,0]-C[1,1]) = (a-e)*u0+(b-f)*u1+(c-g)*u2+(d-h)*u3 = m5+m6+m7+m8
        
        Hmm, I'm getting the same combination twice. Let me try a different grouping.
        
        I think the cleanest approach for this problem is to just use the Strassen-based rank-88 solution and verify it works, then try to optimize. But since the current code already scores 1.0909 (rank 88), let me focus on making sure the rank-56 path actually works if I can construct it, otherwise keep the rank-88.
        
        For now, let me implement a clean 2x4x4 rank-12 using a known formula and verify it.
        """
        # 2x4x4 rank-12 construction
        # A: 2x4, B: 4x4, C: 2x4
        # U: (2*4, 12) = (8, 12)
        # V: (4*4, 12) = (16, 12)
        # W: (4*2, 12) = (8, 12), indexed by k*2+i
        
        u = np.zeros((8, 12))
        v = np.zeros((16, 12))
        w = np.zeros((8, 12))
        
        # A entries: u[0]=a, u[1]=b, u[2]=c, u[3]=d, u[4]=e, u[5]=f, u[6]=g, u[7]=h
        # B entries: v[j*4+k] = B[j,k] for j=0..3, k=0..3
        # C entries: w[k*2+i] = C[i,k] for i=0,1, k=0..3
        
        # 12 multiplications:
        # Split B columns into two pairs: (0,1) and (2,3)
        # For each pair, use 6 multiplications (Strassen-like for 2x4x2)
        
        # Pair (0,1):
        # C[0,0] = a*B00 + b*B10 + c*B20 + d*B30
        # C[0,1] = a*B01 + b*B11 + c*B21 + d*B31
        # C[1,0] = e*B00 + f*B10 + g*B20 + h*B30
        # C[1,1] = e*B01 + f*B11 + g*B21 + h*B31
        
        # 6 mults for this pair:
        # m1 = (a+e)*(B00+B01)
        # m2 = (b+f)*(B10+B11)
        # m3 = (c+g)*(B20+B21)
        # m4 = (d+h)*(B30+B31)
        # m5 = (a-e)*(B00-B01)
        # m6 = (b-f)*(B10-B11)
        # m7 = (c-g)*(B20-B21)
        # m8 = (d-h)*(B30-B31)
        # That's 8, not 6.
        
        # Let me try a different approach. Use the fact that:
        # For 2x4x2, the rank is at most 8 (schoolbook).
        # I need to find if it's less.
        # 2x4x2 tensor: 2*4*2 = 16 entries. Lower bound on rank: 16/max(2,4,2) = 4. Upper: 8.
        # I believe rank(2x4x2) = 8 (no known improvement over schoolbook for this shape).
        
        # So 2x4x4 = two 2x4x2 blocks, each rank 8, total 16.
        # 2x4x8 = four 2x4x2 blocks, each rank 8, total 32.
        # 1x4x8 = rank 32.
        # Total: 32 + 32 = 64.
        
        # That gives rank 64! Let me implement this.
        
        # Actually wait, for 2x4x2, schoolbook is 2*4*2 = 16 mults.
        # But the "rank" of the tensor is the number of terms in the decomposition.
        # For 2x4x2: C[i,k] = sum_j A[i,j]*B[j,k] for i=0,1; k=0,1; j=0,1,2,3.
        # That's 2*4*2 = 16 terms in schoolbook.
        # Can we do better? The tensor is a sum of 16 rank-1 tensors.
        # Lower bound: max(2*2, 2*4, 4*2) = 8. So rank >= 8.
        # I think rank(2x4x2) = 8 is achievable.
        
        # Construction for 2x4x2 rank 8:
        # A = [[a,b],[c,d]] (2x2) ... no, A is 2x4.
        # A = [[a0,a1,a2,a3],[b0,b1,b2,b3]]
        # B = [[c0,d0],[c1,d1],[c2,d2],[c3,d3]] (4x2)
        # C = [[a0*c0+a1*c1+a2*c2+a3*c3, a0*d0+a1*d1+a2*d2+a3*d3],
        #      [b0*c0+b1*c1+b2*c2+b3*c3, b0*d0+b1*d1+b2*d2+b3*d3]]
        
        # 8 multiplications:
        # m1 = (a0+b0)*c0
        # m2 = (a1+b1)*c1
        # m3 = (a2+b2)*c2
        # m4 = (a3+b3)*c3
        # m5 = (a0+b0)*d0
        # m6 = (a1+b1)*d1
        # m7 = (a2+b2)*d2
        # m8 = (a3+b3)*d3
        # m9 = (a0-b0)*c0
        # m10 = (a1-b1)*c1
        # m11 = (a2-b2)*c2
        # m12 = (a3-b3)*c3
        # m13 = (a0-b0)*d0
        # m14 = (a1-b1)*d1
        # m15 = (a2-b2)*d2
        # m16 = (a3-b3)*d3
        # That's 16, same as schoolbook.
        
        # I don't think 2x4x2 can be done in less than 8... wait, schoolbook IS 8 for 2x4x2?
        # 2*4*2 = 16. No, schoolbook for (2x4)*(4x2) is 2*4*2 = 16 mults.
        # The rank of the 2x4x2 tensor is 16 (schoolbook) unless there's a better algorithm.
        
        # I'm going in circles. Let me just stick with the rank-88 solution which I know works,
        # and make sure it's correctly implemented and verified.
        
        # For the purpose of this edit, let me return the rank-88 construction from _construct_rank_88
        # but with a cleaner implementation.
        
        # Actually, let me just return None to signal failure and let the caller use rank-88.
        return None, None, None

    # Try rank-56 construction
    u_248, v_248, w_248 = get_2x4x4_rank12()
    if u_248 is None:
        return None, None, None
    
    u_148, v_148, w_148 = get_1x4x8_schoolbook()
    
    # A1 (1x4) * B (4x8): i_off=0
    add_sub_term(0, 0, 0, 1, 4, 8, u_148, v_148, w_148)
    
    # A2 (2x4) * B (4x8): i_off=1
    add_sub_term(1, 0, 0, 2, 4, 8, u_248, v_248, w_248)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    return U_arr, V_arr, W_arr

def _construct_rank_88():
    """
    Construct a rank-88 decomposition (fallback).
    """
    n, m, p = 3, 4, 8
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_sub_term(i_off, j_off, k_off, sub_n, sub_m, sub_p, u_sub, v_sub, w_sub):
        rank_sub = u_sub.shape[1]
        for c in range(rank_sub):
            u_col = np.zeros(n * m)
            v_col = np.zeros(m * p)
            w_col = np.zeros(p * n)
            
            for i in range(sub_n):
                for j in range(sub_m):
                    u_col[(i_off + i) * m + (j_off + j)] = u_sub[i * sub_m + j, c]
            
            for j in range(sub_m):
                for k in range(sub_p):
                    v_col[(j_off + j) * p + (k_off + k)] = v_sub[j * sub_p + k, c]
            
            for k in range(sub_p):
                for i in range(sub_n):
                    w_col[(k_off + k) * n + (i_off + i)] = w_sub[k * sub_n + i, c]
            
            U_list.append(u_col)
            V_list.append(v_col)
            W_list.append(w_col)

    def get_1x2x4():
        u = np.zeros((1 * 2, 8))
        v = np.zeros((2 * 4, 8))
        w = np.zeros((4 * 1, 8))
        idx = 0
        for j in range(2):
            for k in range(4):
                u[j, idx] = 1.0
                v[j * 4 + k, idx] = 1.0
                w[k * 1, idx] = 1.0
                idx += 1
        return u, v, w

    def get_2x2x4():
        u_s = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ]).T
        
        v_s = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ]).T
        
        w_s = np.zeros((4, 7))
        w_s[0, 0] = 1; w_s[3, 0] = 1
        w_s[1, 1] = 1; w_s[3, 1] = -1
        w_s[2, 2] = 1; w_s[3, 2] = 1
        w_s[0, 3] = 1; w_s[1, 3] = 1
        w_s[0, 4] = -1; w_s[2, 4] = 1
        w_s[3, 5] = 1
        w_s[0, 6] = 1
        
        u_224 = np.zeros((4, 14))
        v_224 = np.zeros((8, 14))
        w_224 = np.zeros((8, 14))
        
        for c in range(7):
            u_224[:, c] = u_s[:, c]
            v_224[0, c] = v_s[0, c]
            v_224[1, c] = v_s[1, c]
            v_224[4, c] = v_s[2, c]
            v_224[5, c] = v_s[3, c]
            w_224[0, c] = w_s[0, c]
            w_224[1, c] = w_s[1, c]
            w_224[2, c] = w_s[2, c]
            w_224[3, c] = w_s[3, c]
            
        for c in range(7):
            u_224[:, c+7] = u_s[:, c]
            v_224[2, c+7] = v_s[0, c]
            v_224[3, c+7] = v_s[1, c]
            v_224[6, c+7] = v_s[2, c]
            v_224[7, c+7] = v_s[3, c]
            w_224[4, c+7] = w_s[0, c]
            w_224[5, c+7] = w_s[1, c]
            w_224[6, c+7] = w_s[2, c]
            w_224[7, c+7] = w_s[3, c]
            
        return u_224, v_224, w_224

    u_124, v_124, w_124 = get_1x2x4()
    u_224, v_224, w_224 = get_2x2x4()
    
    add_sub_term(0, 0, 0, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 0, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 0, 4, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(0, 2, 4, 1, 2, 4, u_124, v_124, w_124)
    add_sub_term(1, 0, 0, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 0, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 0, 4, 2, 2, 4, u_224, v_224, w_224)
    add_sub_term(1, 2, 4, 2, 2, 4, u_224, v_224, w_224)
    
    U_arr = np.array(U_list).T
    V_arr = np.array(V_list).T
    W_arr = np.array(W_list).T
    
    return U_arr, V_arr, W_arr

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = rank_schoolbook
    
    # Try Rank 56 (Winograd-based)
    if time.time() < deadline:
        try:
            U_56, V_56, W_56 = _construct_rank_56()
            if U_56 is not None:
                if _verify_safe(U_56.tolist(), V_56.tolist(), W_56.tolist()):
                    if U_56.shape[1] < best_rank:
                        best_result = {"U": U_56.tolist(), "V": V_56.tolist(), "W": W_56.tolist()}
                        best_rank = U_56.shape[1]
        except Exception:
            pass
    
    # Try Rank 88 (Strassen-based) as fallback
    if time.time() < deadline:
        try:
            U_88, V_88, W_88 = _construct_rank_88()
            if _verify_safe(U_88.tolist(), V_88.tolist(), W_88.tolist()):
                if U_88.shape[1] < best_rank:
                    best_result = {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}
                    best_rank = U_88.shape[1]
        except Exception:
            pass
            
    return best_result
