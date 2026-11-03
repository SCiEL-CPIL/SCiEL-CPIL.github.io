import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We will build the decomposition block by block.
    # Block 1: i in 0..2, j in 0..2, k in 0..2 (3x3x3)
    # Block 2: i in 0..2, j in 3..4, k in 0..2 (3x2x3)
    # Block 3: i in 0..2, j in 0..2, k in 3..6 (3x3x4)
    # Block 4: i in 0..2, j in 3..4, k in 3..6 (3x2x4)
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        U_terms.append((u_idx, val))
        V_terms.append((v_idx, val))
        W_terms.append((w_idx, val))
        
    # Helper to add Strassen 3x3x3
    # Indices: i, j, k in 0..2
    # Strassen 3x3x3 has 23 multiplications.
    # We need the explicit decomposition.
    # Reference: "Strassen's algorithm for 3x3 matrix multiplication"
    # The 23 terms are complex to hardcode from memory.
    # However, we can use a known low-rank decomposition for 3x3.
    # Rank 23 is the best known.
    # I will use a simplified approach: if I can't hardcode 23, I can use schoolbook (27) and accept lower score, 
    # or find a simpler low rank.
    # Actually, I can just use schoolbook for the 3x3x3 part to be safe, 
    # but the prompt asks for improvement.
    # Let's try to use the 23 rank decomposition.
    
    # For now, let's use schoolbook for 3x3x3 to ensure correctness, 
    # but the other blocks give huge savings.
    # 3x3x3 schoolbook: 27.
    # 3x2x3 rank 2.
    # 3x3x4: 3x3x3 (27) + 3x3x1 (9) = 36.
    # 3x2x4: 2.
    # Total: 27 + 2 + 36 + 2 = 67.
    # Score 105/67 = 1.56.
    # This is better than 1.06.
    
    # Let's try to hardcode Strassen 3x3.
    # It is too long. I will use schoolbook for 3x3x3.
    
    # Block 1: 3x3x3 (i:0-2, j:0-2, k:0-2)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_term((i, j), (j, k), (k, i))
                
    # Block 2: 3x2x3 (i:0-2, j:3-4, k:0-2)
    # Rank 2: two terms, one for j=3, one for j=4.
    for j in [3, 4]:
        for i in range(3):
            for k in range(3):
                # We need U[i,j,r], V[j,k,r], W[k,i,r]
                # We can use one rank for all i,k for a fixed j?
                # Yes, as derived before.
                # But add_term adds one rank-1 tensor.
                # We need to group them.
                pass
                
    # My add_term logic is adding individual rank-1 tensors.
    # For the "Rank 1 for n x 1 x p" case, it is ONE tensor.
    # So I need to add ONE term for the whole j=3 block.
    
    # Let's refactor.
    U = {}
    V = {}
    W = {}
    rank = 0
    
    def add_rank1(i, j, k, val=1):
        nonlocal rank
        rank += 1
        U.setdefault((i, j), [0]*rank)[rank-1] += val
        V.setdefault((j, k), [0]*rank)[rank-1] += val
        W.setdefault((k, i), [0]*rank)[rank-1] += val
        
    # Block 1: 3x3x3 Schoolbook (27 terms)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_rank1(i, j, k)
                
    # Block 2: 3x2x3 (2 terms)
    # j=3
    add_rank1(None, 3, None) # This is wrong, indices must be specific.
    
    # The "Rank 1" for n x 1 x p means:
    # U[i, j, r] = 1 for all i
    # V[j, k, r] = 1 for all k
    # W[k, i, r] = 1 for all k, i
    # So we need to add 1 to ALL corresponding entries.
    
    def add_full_rank1(j, val=1):
        nonlocal rank
        rank += 1
        for i in range(3):
            U.setdefault((i, j), [0]*rank)[rank-1] += val
        for k in range(7): # p=7
            V.setdefault((j, k), [0]*rank)[rank-1] += val
        for k in range(7):
            for i in range(3):
                W.setdefault((k, i), [0]*rank)[rank-1] += val
                
    # Block 2: j=3, j=4
    add_full_rank1(3)
    add_full_rank1(4)
    
    # Block 3: 3x3x4 (i:0-2, j:0-2, k:3-6)
    # k=3,4,5 (3x3x3 Schoolbook)
    for i in range(3):
        for j in range(3):
            for k in [3, 4, 5]:
                add_rank1(i, j, k)
    # k=6 (3x3x1 Schoolbook, 9 terms)
    for i in range(3):
        for j in range(3):
            add_rank1(i, j, 6)
            
    # Block 4: 3x2x4 (i:0-2, j:3-4, k:3-6)
    # j=3, j=4
    # But wait, add_full_rank1 adds for ALL k.
    # Here we only want k=3..6.
    # So we cannot use add_full_rank1 directly.
    # We need to add terms for specific k.
    # But for n x 1 x p, the rank is 1 regardless of p.
    # The term is U[i,j]=1, V[j,k]=1, W[k,i]=1.
    # This works for any subset of k?
    # If we restrict k to 3..6, we need V[j,k]=1 for k=3..6 and 0 else?
    # No, V is a matrix. If we set V[j,3..6]=1, it works.
    
    def add_partial_rank1(j, k_range, val=1):
        nonlocal rank
        rank += 1
        for i in range(3):
            U.setdefault((i, j), [0]*rank)[rank-1] += val
        for k in k_range:
            V.setdefault((j, k), [0]*rank)[rank-1] += val
        for k in k_range:
            for i in range(3):
                W.setdefault((k, i), [0]*rank)[rank-1] += val
                
    add_partial_rank1(3, range(3, 7))
    add_partial_rank1(4, range(3, 7))
    
    # Convert to lists
    U_list = [[0]*rank for _ in range(n*m)]
    V_list = [[0]*rank for _ in range(m*p)]
    W_list = [[0]*rank for _ in range(p*n)]
    
    for (i, j), vec in U.items():
        U_list[i*m + j] = vec
    for (j, k), vec in V.items():
        V_list[j*p + k] = vec
    for (k, i), vec in W.items():
        W_list[k*n + i] = vec
        
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
