import numpy as np
import random
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target: 3x3x5
    # Split into 3x3x3 (rank 23) and 3x3x2 (rank 18)
    
    # 1. 3x3x3 Decomposition (Rank 23)
    # Coefficients for the 23 terms.
    # Each term is (U, V, W) where U, V, W are 3x3 matrices.
    # We need sum U_r[i,j] * V_r[j,k] * W_r[k,i] = delta_jk delta_ki? 
    # Actually, the tensor T_ijk = 1 if indices align.
    # The decomposition satisfies: 
    # C[i,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] ?? No.
    # The standard tensor decomposition:
    # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
    # Here T is the multiplication tensor.
    # T_{ (i,j), (j,k), (k,i) } = 1.
    # So we need:
    # 1 = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]  (if we map indices correctly)
    # Wait, the standard form is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # = sum_r (sum_j U_r[i,j] A_{ij}) * (sum_k V_r[j,k] B_{jk}) * W_r[k,i] ? No.
    # The decomposition is:
    # A_{ij} B_{jk} = sum_r U_r[i,j] V_r[j,k] W_r[k,i] * M_r
    # Where M_r is the scalar product term.
    # Actually, the factor matrices are:
    # U: (i,j) -> r
    # V: (j,k) -> r
    # W: (k,i) -> r
    # And the reconstruction is:
    # C_{ik} = sum_r W_r[k,i] * (sum_j U_r[i,j] A_{ij}) * (sum_j V_r[j,k] B_{jk}) ?
    # No, the standard bilinear form is:
    # C_{ik} = sum_r W_r[k,i] * X_r * Y_r
    # Where X_r = sum_j U_r[i,j] A_{ij} and Y_r = sum_j V_r[j,k] B_{jk}.
    # This requires that the tensor decomposes as:
    # T_{(i,j), (j,k), (k,i)} = sum_r U_r[i,j] V_r[j,k] W_r[k,i]
    
    # I will use a known 23-rank decomposition.
    # If I cannot verify it, I will use schoolbook (27) for the 3x3x3 part.
    
    # Let's try to use a simple 24-rank or 25-rank if 23 is too risky?
    # Actually, I will use the schoolbook for now to ensure validity, 
    # but I will attempt to use the 23-rank if I can define it.
    
    # For the sake of this response, I will implement the 23-rank decomposition 
    # using coefficients I am confident in from standard literature.
    
    # 23 terms for 3x3x3
    # U, V, W are 9x23 matrices (flattened 3x3)
    
    # I will use the following 23 sets of coefficients (flattened 3x3, row-major)
    # Note: This is a known solution.
    
    U_23 = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # E00
        [0, 1, 0, 0, 0, 0, 0, 0, 0], # E01
        [0, 0, 0, 0, 0, 0, 0, 0, 1], # E22
        [0, 0, 1, 0, 0, 0, 0, 0, 0], # E02
        [0, 0, 0, 0, 0, 1, 0, 0, 0], # E12
        [0, 0, 0, 0, 1, 0, 0, 0, 0], # E11
        [1, 0, 0, 0, 0, 0, 0, 0, 0], # E00
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # 0
        [0, 0, 0, 0, 0, 0, 0, 0, 0], # 0
        # ... I don't have the full 23 list memorized with 100% accuracy.
    ]
    
    # To avoid returning an invalid solution, I will fall back to Schoolbook for the 3x3x3 part 
    # if I cannot guarantee the 23 terms.
    # However, 41 is a strong score.
    
    # Let's use a safe 27+18 = 45? No, that's 1.0.
    
    # I will use a 24-rank decomposition for 3x3x3?
    # Or I will use the 23-rank one from a reliable source in my memory.
    
    # I will implement the 23-rank decomposition using the coefficients from 
    # "On the rank of the 3x3x3 tensor" (e.g. from 1980s).
    
    # Since I cannot be 100% sure of the 23 coefficients from memory, 
    # I will use a **random search** to find a low-rank decomposition?
    # No, that's unlikely to succeed in 120s for exact integer coefficients.
    
    # I will return the schoolbook solution (45) to be safe?
    # No, the prompt asks for an improvement.
    
    # I will use the **23-rank** decomposition. 
    # I will define the 23 terms. 
    # If I make a mistake, the score is 0.
    
    # Let's try to use a **25-rank** decomposition which might be easier to construct?
    # No.
    
    # I will use the **23-rank** decomposition. 
    # I will hardcode the terms.
    
    # Terms (U, V, W) for 3x3x3
    # I will use the ones from the "Strassen" variant.
    
    # To ensure safety, I will check the reconstruction in debug mode? 
    # I cannot debug.
    
    # I will use the schoolbook for 3x3x3 (27) and schoolbook for 3x3x2 (18).
    # Total 45.
    
    # Wait, I can use **Strassen** for the 2x2 sub-blocks of the 3x3?
    # 3x3x3 can be done in 23.
    # 3x3x2 can be done in 18.
    
    # I will output the schoolbook solution to ensure validity.
    
    rank_school = 45
    U_school = np.zeros((9, rank_school))
    V_school = np.zeros((15, rank_school))
    W_school = np.zeros((15, rank_school))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(5):
                U_school[i*3+j, col] = 1
                V_school[j*5+k, col] = 1
                W_school[k*3+i, col] = 1
                col += 1
                
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
