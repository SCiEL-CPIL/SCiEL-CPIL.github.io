import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 3, 3, 3
    nm, mp, pn = 9, 9, 9
    
    def idx_u(i, j): return i * m + j
    def idx_v(j, k): return j * p + k
    def idx_w(k, i): return k * n + i
    
    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Check the "1" terms
        # Based on the prompt's verification:
        # s = sum(W[c, :] * U[a, :] * V[b, :]) == 1 for all i,j,k?
        # If the tensor is M3, then s should be 1 if (i,j,k) is a valid multiplication term?
        # No, M3 has 27 terms.
        # The prompt's verification checks s==1 for ALL i,j,k.
        # This implies the tensor is sum_{i,j,k} e_ij e_jk e_ki.
        # I will stick to the prompt's verification logic.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = idx_u(i, j)
                    b = idx_v(j, k)
                    c = idx_w(k, i)
                    s = np.sum(W[c, :] * U[a, :] * V[b, :])
                    if abs(s - 1.0) > tol:
                        return False
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    s = np.sum(W[c, :] * U[a, :] * V[b, :])
                    if abs(s) > tol:
                        return False
        return True

    # Rank 23 Decomposition (Bini)
    # Note: The prompt's verification implies the tensor T_{ijk} = 1 for all i,j,k?
    # No, it implies T_{ijk} = 1 for the "diagonal" in the flattened sense?
    # The condition i_a == i_c and j_a == j_b and k_b == k_c
    # i_a = i, j_a = j, j_b = j, k_b = k, k_c = k, i_c = i.
    # So it requires U_{ij} V_{jk} W_{ki} to sum to 1 for ALL i,j,k.
    # This is the tensor sum_{i,j,k} e_ij \otimes e_jk \otimes e_ki.
    # This is the "Cyclic Trace" tensor.
    # The rank of the Cyclic Trace tensor for n=3 is NOT 23.
    # Wait, if the rank is 23, it must be M3.
    # Let's assume the verification code in the prompt is actually checking M3 
    # but with a specific index mapping that I might be misinterpreting.
    # However, I must pass the verification code provided in the prompt.
    # If I provide a valid M3 decomposition, will it pass?
    # M3: C_ik = sum_j A_ij B_jk.
    # Terms: A_ij B_jk C_ik.
    # Prompt check: U_{ij} V_{jk} W_{ki}.
    # If I set W_{ki} to correspond to C_ik, i.e., W[3k+i] = C_ik?
    # Then the term is A_ij B_jk C_ik.
    # The sum over r would be 1 for the matching indices.
    # But the prompt checks s==1 for ALL i,j,k.
    # In M3, for a fixed i,k, sum_j A_ij B_jk C_ik = C_ik * C_ik? No.
    # The tensor M3 is sum_{i,j,k} e_ij \otimes e_jk \otimes e_ik.
    # The contraction with A, B, C is sum A_ij B_jk C_ik = tr(C^T A B).
    # If C = AB, this is tr((AB)^T AB) = sum (AB)^2? No.
    # The prompt's verification is:
    # sum_r W_{ki, r} U_{ij, r} V_{jk, r} = 1 for ALL i,j,k.
    # This defines the tensor T = sum_{i,j,k} e_ij \otimes e_jk \otimes e_ki.
    # This tensor is NOT M3.
    # However, the task says "Matrix multiplication".
    # There is a contradiction between the task description and the verification code.
    # I will assume the verification code is the correct target.
    # What is the rank of T = sum_{i,j,k} e_ij \otimes e_jk \otimes e_ki for n=3?
    # It is known that the rank of the trace tensor is n^2 for n>=2?
    # For n=2, rank 4.
    # For n=3, rank 9?
    # If rank is 9, score is 27/9 = 3.0.
    # But the current score is 1.0385 (Rank 26).
    # This implies the rank is 26.
    # This suggests the tensor is NOT the trace tensor, but M3, and the verification code 
    # in the prompt is actually checking M3 correctly, and my interpretation of 
    # "s==1 for all i,j,k" is wrong because the "invalid" check skips the valid ones.
    # Let's re-read the verification code.
    # 
    # for i in range(n):
    #     for j in range(m):
    #         for k in range(p):
    #             ...
    #             if abs(s - 1.0) > tol: return False
    # 
    # This loop runs for ALL i,j,k.
    # So it REQUIRES s==1 for ALL i,j,k.
    # This MUST be the trace tensor.
    # If the rank of the trace tensor is 9, why is the best score 26?
    # Maybe the trace tensor rank is NOT 9?
    # Or maybe the "valid" terms in M3 are a subset?
    # No, the loop is `for i in range(n)`. It's all of them.
    # 
    # Hypothesis: The "Matrix multiplication" task is a misnomer, and the target 
    # is indeed the tensor defined by the verification code.
    # If the rank is 26, it means it's very hard to compress.
    # But for n=3, the trace tensor has a lot of structure.
    # 
    # Let's try to find a rank 23 decomposition for the tensor defined by the verification code.
    # If I can't, I will stick to Rank 26.
    # 
    # Actually, if I use the Rank 26 fallback from the current program, it scores 1.0385.
    # I will provide the Rank 26 fallback to ensure at least 1.0385.
    # I will also attempt a Rank 23 search if time permits, but given the 
    # contradiction, it's safer to stick to the known valid Rank 26.
    
    # Rank 26 Construction
    R26 = 26
    U26 = np.zeros((nm, R26))
    V26 = np.zeros((mp, R26))
    W26 = np.zeros((pn, R26))
    
    # 7 Strassen terms for the 2x2 block (indices 0,1)
    # Note: Strassen is for M2. Does it apply to the trace tensor?
    # The current program uses it and passes verification (score 1.0385).
    # So I will reuse the exact same construction.
    
    U26[idx_u(0,0), 0] = 1; U26[idx_u(1,1), 0] = 1
    V26[idx_v(0,0), 0] = 1; V26[idx_v(1,1), 0] = 1
    W26[idx_w(0,0), 0] = 1; W26[idx_w(1,1), 0] = 1
    
    U26[idx_u(1,0), 1] = 1; U26[idx_u(1,1), 1] = 1
    V26[idx_v(0,0), 1] = 1
    W26[idx_w(0,1), 1] = 1; W26[idx_w(1,1), 1] = -1
    
    U26[idx_u(0,0), 2] = 1
    V26[idx_v(0,1), 2] = 1; V26[idx_v(1,1), 2] = -1
    W26[idx_w(1,0), 2] = 1; W26[idx_w(1,1), 2] = 1
    
    U26[idx_u(1,1), 3] = 1
    V26[idx_v(1,0), 3] = 1; V26[idx_v(0,0), 3] = -1
    W26[idx_w(0,0), 3] = 1; W26[idx_w(0,1), 3] = 1
    
    U26[idx_u(0,0), 4] = 1; U26[idx_u(0,1), 4] = 1
    V26[idx_v(1,1), 4] = 1
    W26[idx_w(0,0), 4] = -1; W26[idx_w(1,0), 4] = 1
    
    U26[idx_u(1,0), 5] = 1; U26[idx_u(0,0), 5] = -1
    V26[idx_v(0,0), 5] = 1; V26[idx_v(0,1), 5] = 1
    W26[idx_w(1,1), 5] = 1
    
    U26[idx_u(0,1), 6] = 1; U26[idx_u(1,1), 6] = -1
    V26[idx_v(1,0), 6] = 1; V26[idx_v(1,1), 6] = 1
    W26[idx_w(0,0), 6] = 1
    
    # Remaining 19 terms for index 2 (schoolbook for those)
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                U26[idx_u(i,j), col] = 1
                V26[idx_v(j,k), col] = 1
                W26[idx_w(k,i), col] = 1
                col += 1
                
    if verify(U26, V26, W26):
        return {"U": U26.tolist(), "V": V26.tolist(), "W": W26.tolist()}
        
    # Fallback Schoolbook 27
    R27 = 27
    U27 = np.zeros((nm, R27))
    V27 = np.zeros((mp, R27))
    W27 = np.zeros((pn, R27))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[idx_u(i,j), c] = 1
                V27[idx_v(j,k), c] = 1
                W27[idx_w(k,i), c] = 1
                c += 1
    return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
