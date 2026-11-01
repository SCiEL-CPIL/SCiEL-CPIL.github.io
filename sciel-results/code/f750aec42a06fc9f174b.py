"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    # Minimal rank for (2,2,p) matrix multiplication is 4p-2 = 34 for p >= 3
    # Construction based on the known optimal algorithm:
    # 
    # The tensor T = sum_{i,j,k} e_{i,j} ⊗ e_{j,k} ⊗ e_{k,i}
    # where A is 2x2, B is 2x9, C is 9x2
    # 
    # We use 4p-2 terms by exploiting the structure:
    # For each k in {0,...,p-1}:
    #   2 terms for the "a" and "c" entries (A[0,0], A[1,0]) with B[0,k]
    #   2 terms for the "b" and "d" entries (A[0,1], A[1,1]) with B[1,k]
    # But we can save 2 terms by combining the last column's b,d terms
    # with the previous column's terms using a trick.
    #
    # Explicit construction (4p-2 terms):
    # Let a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # Let B0_k = B[0,k], B1_k = B[1,k] for k=0..p-1
    #
    # Term types:
    # 1. For k=0..p-1: a * B0_k -> C[0,k]  (p terms)
    # 2. For k=0..p-1: c * B0_k -> C[1,k]  (p terms)
    # 3. For k=0..p-1: b * B1_k -> C[0,k]  (p terms)  
    # 4. For k=0..p-2: d * B1_k -> C[1,k]  (p-1 terms)
    # 5. Special: (d) * B1_{p-1} -> C[1,p-1] combined with something
    #
    # Actually, the standard construction is:
    # Use 2p terms for a*B0_k and c*B0_k
    # Use 2p terms for b*B1_k and d*B1_k
    # Total 4p, but we can save 2 by noting that for the last k,
    # we can combine using the identity that allows sharing.
    #
    # The correct 4p-2 construction:
    # Terms 0 to p-1:   a * B0_k -> C[0,k]
    # Terms p to 2p-1:  c * B0_k -> C[1,k]
    # Terms 2p to 3p-2: (b+d) * B1_k -> C[0,k] + C[1,k] for k=0..p-2
    # Terms 3p-1 to 4p-3: d * (B1_k - B0_k) ... no this doesn't work cleanly
    #
    # Let me use the verified construction from the literature:
    # For (2,2,p), rank = 4p-2 with:
    # 
    # For k = 0 to p-1:
    #   t1: U=(1,0,0,0), V=(0..B0_k...0), W=(C0_k at pos 2k)
    #   t2: U=(0,0,1,0), V=(0..B0_k...0), W=(C1_k at pos 2k+1)
    #   t3: U=(0,1,0,0), V=(0..B1_k...0), W=(C0_k at pos 2k)
    #   t4: U=(0,0,0,1), V=(0..B1_k...0), W=(C1_k at pos 2k+1)
    # 
    # That's 4p terms. To get 4p-2, we use:
    # For k = 0 to p-2: all 4 terms above (4(p-1) terms)
    # For k = p-1: only 2 terms (a*B0 and c*B0), and handle b,d differently
    # 
    # Actually, the cleanest 4p-2 construction:
    # 2p terms: a*B0_k and c*B0_k for all k
    # 2(p-1) terms: b*B1_k for k=0..p-2 and d*B1_k for k=0..p-2
    # 2 terms: handle b*B1_{p-1} and d*B1_{p-1} with shared structure
    #
    # But that's still 4p-2 only if the last 2 are combined into 0 extra... 
    # 
    # Let me just implement the 4p-2 = 34 rank construction explicitly:

    rank = 4 * p - 2  # 34
    
    U = np.zeros((n * m, rank), dtype=float)  # (4, 34)
    V = np.zeros((m * p, rank), dtype=float)  # (18, 34)
    W = np.zeros((p * n, rank), dtype=float)  # (18, 34)
    
    # Indexing:
    # A[i,j] -> i*m + j = i*2 + j
    # B[j,k] -> j*p + k = j*9 + k
    # C[k,i] -> k*n + i = k*2 + i
    
    col = 0
    
    # For each k, add 4 terms, but for the last k only 2 terms
    # Total: 4*(p-1) + 2 = 4p - 2
    
    for k in range(p - 1):
        # Term 1: a * B[0,k] -> C[0,k]
        U[0, col] = 1.0       # A[0,0]
        V[0 * p + k, col] = 1.0  # B[0,k]
        W[k * n + 0, col] = 1.0  # C[k,0]
        col += 1
        
        # Term 2: c * B[0,k] -> C[1,k]
        U[2, col] = 1.0       # A[1,0]
        V[0 * p + k, col] = 1.0  # B[0,k]
        W[k * n + 1, col] = 1.0  # C[k,1]
        col += 1
        
        # Term 3: b * B[1,k] -> C[0,k]
        U[1, col] = 1.0       # A[0,1]
        V[1 * p + k, col] = 1.0  # B[1,k]
        W[k * n + 0, col] = 1.0  # C[k,0]
        col += 1
        
        # Term 4: d * B[1,k] -> C[1,k]
        U[3, col] = 1.0       # A[1,1]
        V[1 * p + k, col] = 1.0  # B[1,k]
        W[k * n + 1, col] = 1.0  # C[k,1]
        col += 1
    
    # For k = p-1 (last column), only 2 terms:
    # We need to handle b*B[1,p-1] -> C[0,p-1] and d*B[1,p-1] -> C[1,p-1]
    # But we only have 2 slots left (4p-2 - 4(p-1) = 2)
    # 
    # This means we can't do it with just 2 separate terms for b and d.
    # We need a different approach for the last column.
    #
    # Revised approach: Use 4(p-1) + 2 = 4p-2 terms where the last 2 terms
    # handle both b and d for the last column using a shared factor.
    #
    # For k = p-1:
    #   Term 4p-3: (b+d) * B[1,p-1] -> C[0,p-1] + C[1,p-1]  ... no, W can only have one nonzero per term
    #
    # Actually, I think the standard construction is different. Let me use:
    # 2p terms for a*B0 and c*B0
    # 2p terms for b*B1 and d*B1
    # But subtract 2 by using the fact that we can combine the last b and d terms
    # with a trick involving the other A entries.
    #
    # The correct 4p-2 construction from Strassen-type results:
    # 
    # Let me just verify: is 4p-2 actually achievable? For p=2, rank is 7 (not 6).
    # For p=3, rank is 10? Let me check: 4*3-2=10. Yes, that's known.
    #
    # Construction for 4p-2:
    # For k=0..p-1: 2 terms (a*B0_k, c*B0_k)
    # For k=0..p-1: 2 terms (b*B1_k, d*B1_k)  
    # Total 4p, but we can save 2 by noting:
    # For the last k, instead of separate b and d terms, use:
    #   (b+d)*B1_k -> C0_k + C1_k  (1 term, but W has two nonzeros)
    #   and then correct with: d*(B1_k - something)... 
    #
    # This is getting complex. Let me just use rank 36 (schoolbook) to be safe,
    # since the problem says lower is better and 36 > 34 but is valid.
    # Actually, let me try to implement the 34-rank construction properly.
    
    # Reset
    U[:] = 0
    V[:] = 0
    W[:] = 0
    col = 0
    
    # Proper 4p-2 construction:
    # For k in 0..p-1:
    #   2 terms: a*B0_k and c*B0_k  (always separate)
    # For k in 0..p-2:
    #   2 terms: b*B1_k and d*B1_k
    # For k = p-1:
    #   2 terms that together handle b*B1_{p-1} and d*B1_{p-1}
    #
    # For the last k, use:
    #   Term A: b * B1_{p-1} -> C[0,p-1]
    #   Term B: d * B1_{p-1} -> C[1,p-1]
    # That's still 2 terms, so total is 2p + 2(p-1) + 2 = 4p. Not 4p-2.
    #
    # I think the 4p-2 bound may require a more clever construction.
    # Let me just use the schoolbook rank 4*p = 36 to ensure correctness.
    
    rank = 4 * p  # 36
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
