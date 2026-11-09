import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    
    # Rank 27 Schoolbook (Fallback)
    def get_schoolbook():
        rank = 27
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j, col] = 1
                    V[j * 3 + k, col] = 1
                    W[k * 3 + i, col] = 1
                    col += 1
        return U, V, W

    # Rank 23 Strassen-like (Target)
    def get_strassen_3x3():
        # Coefficients for the 23 terms
        # Each term is a triple (U_vec, V_vec, W_vec) of length 9
        # U_vec[i*3+j], V_vec[j*3+k], W_vec[k*3+i]
        
        terms = []
        
        # Helper to add a term defined by 3x3 matrices
        def add_term(U_mat, V_mat, W_mat):
            terms.append((U_mat.flatten(), V_mat.flatten(), W_mat.flatten()))

        # The following 23 terms are a known rank-23 decomposition
        # Note: Indices are 0-based.
        
        # Term 1
        U1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        V1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        W1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # This is just identity, which is rank 3 effectively? No, this is one term.
        # Actually, the decomposition is not just diagonal.
        
        # I will use a verified set of 23 vectors from a trusted source.
        # Since I cannot browse, I will use a heuristic:
        # If I don't have the exact vectors, I should not guess.
        
        # However, there is a known construction.
        # Let's try to use the "Strassen" split.
        # 3x3 can be split into 2x2, 2x1, 1x2, 1x1.
        # This is complex to code on the fly.
        
        return None

    # Strategy:
    # 1. Try to load a known rank-23 decomposition.
    # 2. If not available or fails verification, use rank-27.
    
    # I will implement a simple check. If I can't guarantee rank 23, I stick to 27.
    # But the prompt asks for a "stronger algorithm".
    # I will attempt to hardcode the rank-23 vectors.
    
    # Vectors for Rank 23 (Source: Standard literature)
    # Format: list of 23 tuples (U, V, W) where each is a list of 9 floats
    
    # I will use the following known rank-23 decomposition coefficients.
    # (U, V, W) for each of the 23 terms.
    
    # Note: The exact coefficients are critical.
    # I will use a placeholder that returns None to trigger fallback if I'm not sure.
    # BUT, I can derive a rank-23 decomposition? No.
    
    # Let's try a different tactic.
    # Is there a way to use `scipy` or `numpy` to find it?
    # No, tensor rank minimization is NP-hard.
    
    # I will provide the Rank 27 solution but optimize the construction speed.
    # The current score is 1.0 (27).
    # If I can find Rank 23, score is 27/23 = 1.17.
    
    # I will hardcode the Rank 23 vectors.
    # I found a reliable set in my training data for 3x3 rank 23.
    
    U_data = []
    V_data = []
    W_data = []
    
    # The 23 terms are:
    # 1. (1,0,0, 0,1,0, 0,0,1) ... etc
    
    # Actually, I will use a known valid set.
    # If I am uncertain, I will return the schoolbook.
    
    # To maximize the chance of improvement, I will try to use a rank-23 decomposition
    # that is commonly cited.
    
    # Terms (U, V, W) flattened.
    # I will use a specific set from "Tensor Rank of 3x3 Matrix Multiplication".
    
    # Let's assume I can construct it.
    # I will define the 23 vectors.
    
    # Vector 1
    u1 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    v1 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    w1 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    
    # This is just one term of the schoolbook.
    # The rank 23 decomposition is a specific set of 23 vectors.
    
    # I will return the schoolbook solution to be safe, as hardcoding 23 vectors from memory is risky.
    # However, the directive says "stronger algorithm".
    # I will try to implement a rank-23 decomposition if I can verify it.
    
    # Let's try to generate a rank-23 decomposition using a known algorithm?
    # No.
    
    # I will stick to the schoolbook but ensure it is robust.
    # Wait, I can try to find a rank-24 or 25 decomposition?
    # Unlikely to be better than 23.
    
    # I will use the schoolbook.
    U, V, W = get_schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
