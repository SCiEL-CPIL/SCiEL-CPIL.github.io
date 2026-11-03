"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We split the multiplication C = A * B (3x3 * 3x5) into:
    # C1 = A * B1 (3x3 * 3x3)
    # C2 = A * B2 (3x3 * 3x2)
    
    # Part 1: Schoolbook for 3x3 * 3x3 (Rank 27)
    rank_3x3 = 27
    # Part 2: Rank 15 algorithm for 3x3 * 3x2
    # Note: 3x3 * 3x2 is isomorphic to 2x3 * 3x3.
    # The rank of 2x3 * 3x3 is known to be <= 15.
    rank_3x2 = 15
    
    total_rank = rank_3x3 + rank_3x2
    
    U = [[0] * total_rank for _ in range(n * m)] # 9 x 42
    V = [[0] * total_rank for _ in range(m * p)] # 15 x 42
    W = [[0] * total_rank for _ in range(p * n)] # 15 x 42
    
    # --- Part 1: 3x3 * 3x3 Schoolbook ---
    # Indices: i in 0..2, j in 0..2, k in 0..2
    # U index: i*m + j
    # V index: j*p + k (k in 0..2)
    # W index: k*n + i
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # --- Part 2: 3x3 * 3x2 Rank 15 Algorithm ---
    # We need to compute C2_{ik} = sum_j A_{ij} B_{j, 3+k} for k=0,1
    # B indices in V: j*p + (3+k)
    # W indices for C2: (3+k)*n + i
    
    # We use the Rank 15 algorithm for 2x3 * 3x3 (isomorphic to 3x3 * 3x2).
    # Let A' be 2x3 (corresponding to B2^T), B' be 3x3 (corresponding to A^T).
    # C' = A' * B' (2x3).
    # C'_{u,v} = sum_w A'_{u,w} B'_{w,v}.
    # Mapping:
    # A'_{u,w} = B2_{w, u} (where B2 is 3x2, u=0,1; w=0,1,2)
    # B'_{w,v} = A_{v, w} (where A is 3x3, v=0,1,2; w=0,1,2)
    # C'_{u,v} = C2_{v, u} (where C2 is 3x2)
    
    # The 15-term algorithm for 2x3 * 3x3:
    # This is a specific set of 15 bilinear forms.
    # One known construction uses the following structure (coefficients are 1, -1, or 0.5):
    # We define the terms (M_t) for t=1..15.
    
    # Since hardcoding the 15-rank coefficients is prone to error, 
    # we will use a verification-based approach.
    # We will attempt to construct a Rank 15 decomposition for 3x3x2.
    # If verification fails, we fall back to Rank 18 (Schoolbook) for this part,
    # resulting in Total Rank 45.
    
    # However, to guarantee an improvement over 45, we need a valid lower rank.
    # Let's try to use the **Rank 15** algorithm from a known source.
    # If I cannot guarantee it, I will use **Rank 18** for the second part 
    # and **Rank 21** for the first part? No.
    
    # Let's use a **Rank 42** construction (27 + 15).
    # I will implement the 15-rank algorithm for 2x3x3.
    
    # Coefficients for 2x3x3 Rank 15:
    # U' (2*3=6 rows, 15 cols), V' (3*3=9 rows, 15 cols), W' (2*3=6 rows, 15 cols)
    # M_t = U'_t * V'_t
    # C' = sum M_t * W'_t
    
    # One such algorithm (from "Fast Matrix Multiplication" by Strassen, extended):
    # It is complex. 
    # To ensure validity and improvement, I will use the **Rank 15** algorithm 
    # for 3x3x2 directly if possible, or fall back to 45.
    
    # Given the risk, I will implement the **Rank 15** algorithm for 3x3x2 
    # using a verified set of coefficients from a standard library or known list.
    # Since I don't have it, I will use **Rank 18** for 3x3x2 (Schoolbook) 
    # and **Rank 23** for 3x3x3?
    
    # Actually, the safest improvement is to use **Rank 21** for 3x3x3.
    # There is a known Rank 21 algorithm for 3x3x3.
    # Let's assume I can find it.
    
    # For now, I will return the **Rank 45** Schoolbook to ensure validity, 
    # but I will structure the code to allow easy insertion of the 15-rank part.
    
    # Wait, I can use **NumPy** to check if a random low-rank approximation works? No.
    
    # I will use the **Rank 45** baseline.
    
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
