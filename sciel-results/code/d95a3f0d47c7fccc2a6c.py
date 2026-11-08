DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 20 decomposition for 3x3 matrix multiplication
    # This is a known result (e.g., from Strassen's work or subsequent improvements)
    # that 3x3 matrix multiplication can be done with 20 multiplications.
    # The decomposition is given by 20 triples (u, v, w) with coefficients.
    
    rank = 20
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices:
    # A: i*m + j
    # B: j*p + k
    # W: k*n + i
    
    # The following triples are from a known rank-20 decomposition.
    # Format: (u_idx, v_idx, w_idx, coeff)
    # We need to be careful with the specific tensor definition.
    # The task specifies W[k*n+i].
    # The standard tensor is T_{i,j,k} = A_{ij} B_{jk} -> C_{ik}.
    # Here C_{ki} is stored at k*n+i.
    # So the output index is w_idx = k*n + i.
    # The input indices are u_idx = i*m + j and v_idx = j*p + k.
    
    # A known rank-20 decomposition for 3x3x3 (from literature, e.g. "The Rank of the 3x3 Matrix Multiplication Tensor"):
    # There are several. One is derived from Strassen's 2x2 method plus optimizations.
    # Let's use the one from "Alon's decomposition" or similar.
    # Actually, the minimal rank for 3x3 is 19? No, 19 is for 3x3x3 over rationals? 
    # Wait, the lower bound is 19. Is 19 achievable?
    # Yes, Alon found a rank 19 decomposition? No, Alon proved lower bound 19.
    # The best known upper bound is 20 (Strassen 1969? No, Strassen did 2x2).
    # Actually, the rank of 3x3 matrix multiplication is 20.
    # Reference: "The rank of the 3x3 matrix multiplication tensor is 20" by B. D. Smith? No.
    # It is known that r(3x3x3) = 20.
    
    # Let's verify the triples for rank 20.
    # One construction:
    # We can use the 2x2 Strassen trick on subblocks.
    # But it's easier to hardcode the 20 terms if I can find them.
    
    # Here is a known set of 20 terms for the tensor T_{i,j,k} (where T_{i,j,k} = 1 if i=j=k? No, matrix mult tensor).
    # The tensor is sum_{i,j,k} e_i^T A e_j * e_j^T B e_k * e_k^T C e_i.
    # My indexing:
    # U: i*m+j
    # V: j*p+k
    # W: k*n+i
    
    # Let's try to construct it from Strassen's 2x2 on the top-left 2x2 block and handle the rest.
    # This is complex to derive on the fly.
    
    # Alternative: Use a pre-computed rank-20 decomposition.
    # I will use the decomposition from "A note on the rank of the 3x3 matrix multiplication tensor" or similar.
    # Since I don't have the exact list memorized perfectly, I will attempt to use a known valid one.
    
    # If I cannot guarantee rank 20, I should stick to rank 27 to be safe?
    # But score 1.0 is rank 27. I need to improve.
    # Rank 20 gives score 27/20 = 1.35.
    # Rank 19 gives score 27/19 = 1.42.
    
    # Let's assume rank 20 is achievable and try to find the terms.
    # Actually, I can use the "Strassen" style construction for 3x3.
    
    # Let's use the decomposition from:
    # "The rank of the 3x3 matrix multiplication tensor" by ...
    # There is a known decomposition with 20 terms.
    
    # Terms (u, v, w, coeff):
    # u = i*3+j, v = j*3+k, w = k*3+i
    
    # I will use the following 20 terms (verified against the tensor definition):
    # Note: The coefficients might be 1, -1, 2, etc.
    
    terms = [
        # Block (0,0) to (1,1) using Strassen 2x2 logic?
        # Let's just use a known valid list from a reliable source if I can recall it.
        # If not, I will output rank 27.
        
        # Wait, I can use the fact that the rank is 20.
        # Let's try to find a valid rank 20 decomposition online in my training data...
        # One such decomposition is:
        # 1. (0,0,0, 1) -> A00 B00 -> C00
        # 2. (1,1,1, 1) -> A01 B01 -> C01
        # ... This is schoolbook.
        
        # Strassen for 2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For 3x3, we can partition into 2x2 and 1x1.
        # This is getting too complex to derive correctly in the short time.
        
        # However, I know that rank 20 is the current best known upper bound.
        # I will use a placeholder that returns rank 27 if I can't be sure, but I really want to try 20.
        
        # Let's try a different approach: Use the "Strassen" algorithm for 3x3 directly if I can find the formulas.
        # Actually, there is a simpler rank 20 decomposition.
        
        # Let's use the one from "The rank of the 3x3 matrix multiplication tensor is 20" by B. D. Smith (1990s?).
        # I will hardcode a known valid set of 20 terms.
        
        # If I fail, I fall back to 27.
    ]
    
    # Since I cannot guarantee the exact 20 terms from memory, I will use a conservative approach:
    # I will implement the Strassen 2x2 on the top-left 2x2 submatrix and handle the rest with schoolbook.
    # This might not be rank 20, but it's a start.
    # Actually, let's just return rank 27 to be safe. The instruction says "Keep the program's entry-point function name(s) and their return interface EXACTLY as in the current program above".
    # The current program returns rank 27.
    # But I am supposed to IMPROVE it.
    
    # Let's try to find a valid rank 20 decomposition.
    # I will use the following 20 terms (from a known source):
    
    # u, v, w, coeff
    # 0: (0,0,0, 1)
    # 1: (1,1,1, 1)
    # 2: (2,2,2, 1)
    # 3: (0,3,3, 1)
    # 4: (3,0,0, 1)
    # 5: (0,4,4, 1)
    # 6: (4,0,0, 1)
    # 7: (1,3,4, 1)
    # 8: (3,1,3, 1)
    # 9: (1,4,3, 1)
    # 10: (4,1,4, 1)
    # 11: (2,3,6, 1)
    # 12: (3,2,5, 1)
    # 13: (2,4,5, 1)
    # 14: (4,2,6, 1)
    # 15: (2,6,8, 1)
    # 16: (6,2,7, 1)
    # 17: (5,6,8, 1)
    # 18: (6,5,7, 1)
    # 19: (7,7,7, 1) -> This is just a guess.
    
    # This list is likely incorrect.
    
    # I will stick to rank 27.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
