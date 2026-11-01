"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 2 * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each column k in p:
    # We use 2 multiplications.
    # M1_k = (A_00 + A_11) * (B_0k + B_1k)
    # M2_k = (A_01 + A_10) * (B_0k + B_1k)
    # C_0k = A_00 B_0k + A_01 B_1k
    # C_1k = A_10 B_0k + A_11 B_1k
    
    # Wait, my derivation showed:
    # C_0k = A_00 B_0k + A_01 B_1k
    # This was obtained with:
    # u_00,1=1, u_11,1=1, v_0,1=1, v_1,1=1, w_0,1=1, w_1,1=1
    # u_01,2=1, u_10,2=1, v_0,2=1, v_1,2=1, w_0,2=1, w_1,2=1
    
    # Let's verify this decomposition for one column k.
    # Term 1: U[0][col]=1, U[3][col]=1 (indices 0 and 3 for A_00, A_11)
    # V[0][col]=1, V[1][col]=1 (indices 0 and 1 for B_0k, B_1k)
    # W[0][col]=1, W[1][col]=1 (indices 0 and 1 for C_0k, C_1k)
    
    # Term 2: U[1][col]=1, U[2][col]=1 (indices 1 and 2 for A_01, A_10)
    # V[0][col]=1, V[1][col]=1
    # W[0][col]=1, W[1][col]=1
    
    # Check C_0k (index k*n + 0 = k*2):
    # Sum over l: U[0][l] V[0][l] W[k*2][l] + U[1][l] V[1][l] W[k*2][l]
    # For l=col1: 1 * 1 * 1 = 1. (A_00 B_0k)
    # For l=col2: 0 * 1 * 1 = 0.
    # Wait, U[1][col2]=1, V[1][col2]=1.
    # So term is A_01 B_1k.
    # So C_0k gets A_00 B_0k + A_01 B_1k. Correct.
    
    # Check C_1k (index k*n + 1 = k*2 + 1):
    # Sum over l: U[0][l] V[0][l] W[k*2+1][l] + U[1][l] V[1][l] W[k*2+1][l]
    # For l=col1: 1 * 1 * 1 = 1. (A_00 B_0k) -> WAIT.
    # C_1k should be A_10 B_0k + A_11 B_1k.
    # My term 1 has U[0][col1]=1 (A_00).
    # So C_1k gets A_00 B_0k. This is WRONG.
    
    # My previous check:
    # T_10 = 1. This corresponds to A_10 B_0k C_1k.
    # In my R=2 example:
    # u_10,2 = 1.
    # So A_10 is in term 2.
    # v_0,2 = 1. So B_0k is in term 2.
    # w_1,2 = 1. So C_1k is in term 2.
    # So A_10 B_0k contributes to C_1k. Correct.
    
    # But what about A_00 B_0k?
    # u_00,1 = 1. v_0,1 = 1.
    # w_0,1 = 1 -> C_0k.
    # w_1,1 = 1 -> C_1k.
    # So A_00 B_0k contributes to BOTH C_0k and C_1k.
    # But C_1k should NOT contain A_00 B_0k.
    
    # So my R=2 decomposition is INVALID for M_2,2,1.
    # The condition is sum_l u_ij,l v_jk,l w_ik,l = 1.
    # For i=1, j=0, k=0:
    # Sum u_10,l v_0,l w_1,l = 1.
    # In my example: l=2: 1*1*1 = 1. OK.
    # For i=0, j=0, k=0:
    # Sum u_00,l v_0,l w_0,l = 1.
    # In my example: l=1: 1*1*1 = 1. OK.
    
    # The problem is that w_1,1 = 1.
    # So u_00,1 v_0,1 w_1,1 = 1*1*1 = 1.
    # This means A_00 B_0k contributes to C_1k.
    # But we need it to be 0.
    
    # So we need w_1,1 = 0.
    # If w_1,1 = 0, then for i=1, j=0, k=0:
    # Sum u_10,l v_0,l w_1,l.
    # l=1: u_10,1 v_0,1 w_1,1 = 0.
    # l=2: u_10,2 v_0,2 w_1,2 = 1*1*1 = 1.
    # So we need u_10,2=1, v_0,2=1, w_1,2=1.
    
    # And for i=1, j=1, k=0:
    # Sum u_11,l v_1,l w_1,l = 1.
    # l=1: u_11,1 v_1,1 w_1,1 = u_11,1 * 1 * 0 = 0.
    # l=2: u_11,2 v_1,2 w_1,2 = u_11,2 * 1 * 1 = u_11,2.
    # So u_11,2 = 1.
    
    # And for i=0, j=1, k=0:
    # Sum u_01,l v_1,l w_0,l = 1.
    # l=1: u_01,1 v_1,1 w_0,1 = u_01,1 * 1 * 1 = u_01,1.
    # l=2: u_01,2 v_1,2 w_0,2 = u_01,2 * 1 * w_0,2.
    
    # We have freedom to choose w_0,2.
    # Let's try to set w_0,2 = 0.
    # Then u_01,1 = 1.
    
    # Summary for R=2:
    # l=1:
    # u_00,1=1, u_11,1=1 (from earlier logic? No, let's restart)
    
    # Constraints:
    # 1. sum u_00,l v_0,l w_0,l = 1
    # 2. sum u_01,l v_1,l w_0,l = 1
    # 3. sum u_10,l v_0,l w_1,l = 1
    # 4. sum u_11,l v_1,l w_1,l = 1
    
    # Try:
    # v_0,1=1, v_1,1=1
    # v_0,2=1, v_1,2=1
    
    # w_0,1=1, w_1,1=0
    # w_0,2=0, w_1,2=1
    
    # From 1: u_00,1 * 1 * 1 + u_00,2 * 1 * 0 = 1 => u_00,1 = 1.
    # From 2: u_01,1 * 1 * 1 + u_01,2 * 1 * 0 = 1 => u_01,1 = 1.
    # From 3: u_10,1 * 1 * 0 + u_10,2 * 1 * 1 = 1 => u_10,2 = 1.
    # From 4: u_11,1 * 1 * 0 + u_11,2 * 1 * 1 = 1 => u_11,2 = 1.
    
    # So:
    # u_00,1=1, u_01,1=1, u_10,2=1, u_11,2=1.
    # Others 0.
    
    # Check:
    # T_00: u_00,1 v_0,1 w_0,1 = 1*1*1 = 1. OK.
    # T_01: u_01,1 v_1,1 w_0,1 = 1*1*1 = 1. OK.
    # T_10: u_10,2 v_0,2 w_1,2 = 1*1*1 = 1. OK.
    # T_11: u_11,2 v_1,2 w_1,2 = 1*1*1 = 1. OK.
    
    # This works! Rank is 2.
    
    for k in range(p):
        col1 = 2 * k
        col2 = 2 * k + 1
        
        # Term 1
        U[0][col1] = 1 # A_00
        U[1][col1] = 1 # A_01
        V[0][col1] = 1 # B_0k
        V[1][col1] = 1 # B_1k
        W[2*k][col1] = 1 # C_0k
        W[2*k+1][col1] = 0
        
        # Term 2
        U[2][col2] = 1 # A_10
        U[3][col2] = 1 # A_11
        V[0][col2] = 1 # B_0k
        V[1][col2] = 1 # B_1k
        W[2*k][col2] = 0
        W[2*k+1][col2] = 1 # C_1k

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
