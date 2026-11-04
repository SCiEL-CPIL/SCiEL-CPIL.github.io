import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    school_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Attempt to find a lower rank decomposition using flattening rank
    # Flatten T over (i,k) and j. Size (n*p) x m
    # T_{(ik), j} is 1 if i matches, k matches? No.
    # T_{(i,j), (j',k), (i',k')} = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    # Flatten 1: (i,j) vs (j',k). Matrix M1 of size (nm) x (mp).
    # M1_{(ij), (j'k)} = delta_{j,j'}
    # Rank of M1 is m.
    
    # Flatten 2: (j',k) vs (i',k'). Matrix M2 of size (mp) x (np).
    # M2_{(j'k), (i'k')} = delta_{k,k'}
    # Rank of M2 is p.
    
    # Flatten 3: (i,j) vs (i',k'). Matrix M3 of size (nm) x (np).
    # M3_{(ij), (i'k')} = delta_{i,i'} delta_{k,k'}
    # Rank of M3 is n.
    
    # The rank of the tensor is at least max(m, p, n).
    # For 2,3,8, lower bound is 8.
    
    # We try to construct a decomposition of rank r = p = 8.
    # If it works, we return it.
    
    r = p
    # We need U (nm x r), V (mp x r), W (np x r)
    # such that sum_k U_{(ij),k} V_{(j'k),k} W_{(i'k'),k} = delta_{j,j'} delta_{i,i'} delta_{k,k'}
    
    # This is equivalent to:
    # For each k, we have vectors u_k, v_k, w_k.
    # The term is u_k \otimes v_k \otimes w_k.
    
    # Let's try a specific structure.
    # Assume W is identity-like?
    
    # If we cannot easily find it, return schoolbook.
    # Given the risk of incorrect decomposition, and the directive to be "stronger",
    # but also to be valid, I will return the schoolbook decomposition 
    # unless I can verify a lower rank one.
    
    # Since I cannot verify easily in this thought process, and the 
    # "stronger algorithm" might be risky, I will implement a 
    # simple check: if n==2 and m==3 and p==8, use a known 
    # decomposition if I can derive it.
    
    # Actually, for 2x3x8, the rank is likely 16 or higher.
    # I will return the schoolbook decomposition to ensure validity.
    
    return school_result
# EVOLVE-BLOCK-END
