"""Winograd Fourier-based decomposition for 2x4x5 matrix multiplication."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    # U: (n*m, rank) = (8, rank)
    # V: (m*p, rank) = (20, rank)
    # W: (p*n, rank) = (10, rank)
    
    # We'll build the decomposition using Winograd-style Fourier transforms
    # for small blocks, then combine them.
    
    # Strategy: Split the 4x4 middle dimension into two 2x2 blocks.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_{j=0,1} A[i,j] B[j,k] + sum_{j=2,3} A[i,j] B[j,k]
    
    # For each 2x2 block of A times the full 2x5 B block:
    # C_block[i,k] = A_block[i,0]*B_block[0,k] + A_block[i,1]*B_block[1,k]
    # This is a 2x2 * 2x5 multiplication.
    
    # For 2x2 * 2x5, we can use Winograd with FFT-like structure.
    # The tensor is T[i,j,k] = delta_{i,j} (where i in {0,1}, j in {0,1}, k in {0,1,2,3,4})
    # Actually it's simpler: for each output (i,k), C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    # Rank of 2x2*2x5: Let's think of it as two independent 2x1*2x5 problems (one for each i).
    # 2x1*2x5: C[k] = a0*B[0,k] + a1*B[1,k], for k=0..4.
    # This is like multiplying a 2-element vector by a 2x5 matrix.
    # The tensor rank of this is 5 (each k gives one term, and they're independent in the output).
    # Wait, no. The tensor is T[a,b,k] where a in {0,1} (input vector), b in {0,1} (rows of B), k in {0..4}.
    # T[a,b,k] = 1 if a==b, else 0. So it's two disjoint copies of the identity for each k.
    # Actually T[a,b,k] = delta_{a,b}. So for each k, we have a 2x2 identity tensor.
    # The rank of a single k-slice is 2 (two rank-1 terms: a0*B[0,k] and a1*B[1,k]).
    # Since the k-slices are independent (different output indices), the total rank is 5*2 = 10?
    # No wait, the rank-1 terms can share the same a and b indices but have different k.
    # A rank-1 term is u(a) * v(b) * w(k).
    # For term t: u_t(a) * v_t(b) * w_t(k).
    # We need sum_t u_t(a) v_t(b) w_t(k) = delta_{a,b} for all a,b,k.
    # This means for each k, sum_t u_t(a) v_t(b) w_t(k) = delta_{a,b}.
    # Let's try: for each k, use two terms: one with u=(1,0), v=(1,0), w=(0,...,1,...,0) and one with u=(0,1), v=(0,1), w=(0,...,1,...,0).
    # That gives rank 2*5 = 10.
    # Can we do better? If we allow w_t(k) to be non-zero for multiple k, we might save.
    # But the constraint is sum_t u_t(a) v_t(b) w_t(k) = delta_{a,b} for ALL k.
    # This is a system of 5*4 = 20 equations (for a,b in {0,1} and k in {0..4}).
    # Unknowns: for each term t, u_t(0), u_t(1), v_t(0), v_t(1), w_t(0),...,w_t(4). That's 10 unknowns per term.
    # With rank r, we have 10r unknowns. We need 10r >= 20, so r >= 2.
    # But we also need the structure. Let's try r=5:
    # Term t (t=0..4): u_t = (1,0), v_t = (1,0), w_t = e_t (unit vector at position t).
    # Then sum_t u_t(a) v_t(b) w_t(k) = sum_t delta_{a,0} delta_{b,0} delta_{k,t} = delta_{a,0} delta_{b,0} delta_{k,t}... no that's not right.
    # Actually: sum_t u_t(a) v_t(b) w_t(k) = sum_t [a==0 ? 1 : 0] [b==0 ? 1 : 0] [k==t ? 1 : 0]
    # = [a==0 and b==0] * sum_t [k==t] = [a==0 and b==0] * 1 (since exactly one t matches k).
    # So we get delta_{a,0} delta_{b,0}, not delta_{a,b}.
    # We need to also cover a=1,b=1.
    # So we need: for a=0,b=0: sum_t u_t(0) v_t(0) w_t(k) = 1 for all k.
    # For a=0,b=1: sum_t u_t(0) v_t(1) w_t(k) = 0 for all k.
    # For a=1,b=0: sum_t u_t(1) v_t(0) w_t(k) = 0 for all k.
    # For a=1,b=1: sum_t u_t(1) v_t(1) w_t(k) = 1 for all k.
    
    # Let's try rank 5 with:
    # t=0..2: u_t=(1,0), v_t=(1,0), w_t has 1 at position t.
    # t=3..4: u_t=(0,1), v_t=(0,1), w_t has 1 at position t.
    # Then for a=0,b=0: sum_{t=0}^{2} 1*1*w_t(k) = w_0(k)+w_1(k)+w_2(k). This equals 1 only if k in {0,1,2}.
    # That doesn't work for k=3,4.
    
    # So rank 5 doesn't work with this simple structure. Let's try rank 10 as the safe bet.
    # Actually, the schoolbook for 2x2*2x5 is just 4*5=20 scalar multiplications, but the tensor rank is lower.
    # Let me just use the schoolbook decomposition: for each (i,j,k), one term A[i,j]*B[j,k]*C[i,k].
    # That's 2*2*5 = 20 terms per 2x2*2x5 block. Two blocks give 40.
    # But we can do better with Winograd.
    
    # Let me just implement a clean schoolbook-based decomposition first to ensure correctness,
    # then try to optimize.
    
    # Schoolbook: C[i,k] = sum_j A[i,j] B[j,k]
    # For each (i,j,k), add a rank-1 term: U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1.
    # Total terms: n*m*p = 2*4*5 = 40.
    
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 8 x 40
    V = [[0] * rank for _ in range(m * p)]  # 20 x 40
    W = [[0] * rank for _ in range(p * n)]  # 10 x 40
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
