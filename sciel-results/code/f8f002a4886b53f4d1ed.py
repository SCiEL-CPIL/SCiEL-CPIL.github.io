import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Standard rank for this problem: 7 * 7 = 49
    # We use Strassen's algorithm for 2x2 blocks.
    # There are 7 blocks of 2x2 within the 2x14 matrix B (since p=14, m=2, n=2)
    # Actually, B is 2x14. We can split B into 7 blocks of size 2x2 along the column dimension.
    # Let B = [B1, B2, ..., B7] where each Bk is 2x2.
    # Then C = A * B = [A*B1, A*B2, ..., A*B7].
    # Each A*Bk is a 2x2 multiplication.
    # Strassen reduces the cost of each 2x2 multiplication from 8 to 7 scalar multiplications.
    # So total rank = 7 (Strassen terms) * 7 (blocks) = 49.
    
    rank = 49
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2 multiplication
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    
    # One valid set of Strassen coefficients (from standard literature):
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)(b00)
    # M3 = (a00)(b01 - b11)
    # M4 = (a11)(b10 - b00)
    # M5 = (a00 + a01)(b11)
    # M6 = (a10 - a00)(b00 + b01)
    # M7 = (a01 - a11)(b10 + b11)
    
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 - M6
    
    # Let's verify the coefficients for each output element.
    # U coefficients (for a00, a01, a10, a11)
    # V coefficients (for b00, b01, b10, b11)
    # W coefficients (for c00, c01, c10, c11)
    
    U_coeffs = [
        (1, 0, 0, 1),   # M1: a00 + a11
        (0, 0, 1, 1),   # M2: a10 + a11
        (1, 0, 0, 0),   # M3: a00
        (0, 0, 0, 1),   # M4: a11
        (1, 1, 0, 0),   # M5: a00 + a01
        (-1, 0, 1, 0),  # M6: a10 - a00
        (0, 1, 0, -1)   # M7: a01 - a11
    ]
    
    V_coeffs = [
        (1, 0, 0, 1),   # M1: b00 + b11
        (1, 0, 0, 0),   # M2: b00
        (0, 1, 0, -1),  # M3: b01 - b11
        (-1, 0, 1, 0),  # M4: b10 - b00
        (0, 0, 0, 1),   # M5: b11
        (1, 1, 0, 0),   # M6: b00 + b01
        (0, 0, 1, 1)    # M7: b10 + b11
    ]
    
    W_coeffs = [
        (1, 0, 0, 1),   # M1: c00 + c11
        (0, 0, 1, -1),  # M2: c10 - c11
        (0, 1, 0, 1),   # M3: c01 + c11
        (1, 0, 1, 0),   # M4: c00 + c10
        (-1, 1, 0, 0),  # M5: -c00 + c01
        (0, 0, 0, 1),   # M6: c11
        (1, 0, 0, 0)    # M7: c00
    ]
    
    # Let's re-verify the W coefficients against the formulas:
    # c00 = M1 + M4 - M5 + M7  -> coeffs: (1, 0, 0, 1) for M1, (1,0,1,0) for M4, (-1,1,0,0) for M5, (1,0,0,0) for M7
    # Summing these column vectors in W (which maps to c00, c01, c10, c11):
    # M1 contributes (1, 0, 0, 1) to W? No, W[s] is the contribution of M_s to the output.
    # So W[0] (M1) should have c00=1, c11=1. -> (1, 0, 0, 1) is correct if order is c00, c01, c10, c11.
    # W[3] (M4) should have c00=1, c10=1. -> (1, 0, 1, 0) is correct.
    # W[4] (M5) should have c00=-1, c01=1. -> (-1, 1, 0, 0) is correct.
    # W[6] (M7) should have c00=1. -> (1, 0, 0, 0) is correct.
    
    # c01 = M3 + M5
    # M3: c01=1, c11=1 -> (0, 1, 0, 1)
    # M5: c01=1 -> (0, 1, 0, 0) but W[4] is (-1, 1, 0, 0). So c01 gets +1 from M5. Correct.
    
    # c10 = M2 + M4
    # M2: c10=1, c11=-1 -> (0, 0, 1, -1)
    # M4: c10=1 -> (1, 0, 1, 0). So c10 gets +1 from M4. Correct.
    
    # c11 = M1 - M2 + M3 - M6
    # M1: c11=1
    # M2: c11=-1
    # M3: c11=1
    # M6: c11=-1 -> W[5] should be (0, 0, 0, -1).
    # But my W_coeffs[5] is (0, 0, 0, 1).
    # Let's check M6: M6 = (a10-a00)(b00+b01).
    # c11 = M1 - M2 + M3 - M6.
    # So M6 contributes -1 to c11.
    # So W[5] should be (0, 0, 0, -1).
    
    # Let's re-read the standard Strassen W coefficients.
    # Some sources define it slightly differently.
    # Let's just solve for the W coefficients to be safe.
    # We know U and V. We need W such that U^T W V = Tensor.
    # Actually, it's easier to just compute the product for each term s.
    # For a fixed s, the contribution to C is U_s * M_s * V_s^T (conceptually, but they are vectors).
    # Actually, the term is U[:, s] * M_s * V[:, s]^T? No.
    # The decomposition is T = sum_s u_s \otimes v_s \otimes w_s.
    # Here T_{i,j,k} = sum_s U_{i,j,s} V_{j,k,s} W_{k,i,s}? No, indices are tricky.
    # T_{i, j, k} where i is row of A (0..n-1), j is col of A (0..m-1), k is col of B (0..p-1).
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # So T_{i, j, k} = A_{i, j} B_{j, k} C_{i, k}^{-1}? No.
    # The tensor T has indices (i, j, k) and value 1 if the multiplication A_{ij} B_{jk} contributes to C_{ik}.
    # Actually, the multiplication tensor \mu_{n,m,p} has entries T_{(i,j),(j,k),(i,k)} = 1.
    # Rank decomposition: T = sum_s U_s \otimes V_s \otimes W_s.
    # U_s is a vector of length n*m (flattened A indices).
    # V_s is a vector of length m*p (flattened B indices).
    # W_s is a vector of length p*n (flattened C indices, but wait, output is C which is n x p).
    # The problem states W has shape (p*n, rank).
    # And "W[k*n+i, r]". This implies W index is k*n + i.
    # So W_{k, i} corresponds to C_{i, k}.
    
    # Let's verify the mapping in the previous code.
    # U rows: i*m + j. (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3. Correct.
    # V rows: j*p + k. (0,0)->0, (0,1)->1, ..., (0,13)->13, (1,0)->14, ... Correct.
    # W rows: k*n + i.
    # C is 2x14. i is row (0,1), k is col (0..13).
    # C_{0,0} -> k=0, i=0 -> index 0.
    # C_{0,1} -> k=1, i=0 -> index 1.
    # C_{1,0} -> k=0, i=1 -> index 2.
    # C_{1,1} -> k=1, i=1 -> index 3.
    # So W index for C_{i,k} is k*2 + i.
    
    # In the previous code:
    # W[4*b][col] = W_coeffs[s][0]  # c00 of block b
    # Block b corresponds to columns 2b and 2b+1 of B, and thus C.
    # c00 of block b is C_{0, 2b}.
    # Index for C_{0, 2b} is (2b)*2 + 0 = 4b. Correct.
    # c01 of block b is C_{0, 2b+1}.
    # Index for C_{0, 2b+1} is (2b+1)*2 + 0 = 4b+2. Correct.
    # c10 of block b is C_{1, 2b}.
    # Index for C_{1, 2b} is (2b)*2 + 1 = 4b+1. Correct.
    # c11 of block b is C_{1, 2b+1}.
    # Index for C_{1, 2b+1} is (2b+1)*2 + 1 = 4b+3. Correct.
    
    # So the mapping is correct.
    # The only issue is the W_coeffs values.
    # Let's re-evaluate W_coeffs[5] (M6).
    # M6 = (a10 - a00)(b00 + b01).
    # c11 = M1 - M2 + M3 - M6.
    # So M6 contributes -1 to c11.
    # W_coeffs[5] should be (0, 0, 0, -1).
    # The previous code had (0, 0, 0, 1).
    # If the previous code scored 1.1429, it was valid.
    # Maybe the Strassen variant I'm using is different?
    # Let's check the product expansion again.
    # If W_coeffs[5] is (0,0,0,1), then c11 gets +M6.
    # Then c11 = M1 - M2 + M3 + M6.
    # M1 - M2 + M3 + M6 = (a00+a11)(b00+b11) - (a10+a11)b00 + a00(b01-b11) + (a10-a00)(b00+b01)
    # = a00b00 + a00b11 + a11b00 + a11b11 - a10b00 - a11b00 + a00b01 - a00b11 + a10b00 + a10b01 - a00b00 - a00b01
    # Terms:
    # a00b00: 1 - 1 = 0
    # a00b11: 1 - 1 = 0
    # a11b00: 1 - 1 = 0
    # a11b11: 1
    # a10b00: -1 + 1 = 0
    # a00b01: 1 - 1 = 0
    # a10b01: 1
    # Result: a11b11 + a10b01.
    # But C11 = a10b01 + a11b11.
    # So it works!
    # So the variant is:
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # Let's check c10 = M2 + M4.
    # M2 + M4 = (a10+a11)b00 + a11(b10-b00) = a10b00 + a11b00 + a11b10 - a11b00 = a10b00 + a11b10.
    # C10 = a10b00 + a11b10. Correct.
    
    # So the coefficients in the previous code were actually correct for this specific Strassen variant.
    
    # We will stick with the previous coefficients as they are known to work.
    
    # However, I will double check W_coeffs[5] in the code I write.
    # Previous code: W_coeffs[5] = (0, 0, 0, 1).
    # My derivation: c11 = ... + M6. So W_coeffs[5] should be (0,0,0,1).
    # Yes, matches.
    
    # Let's construct the matrices.
    # We have 7 blocks. For each block b (0..6) and each Strassen term s (0..6), we have a column in U, V, W.
    # Column index = s * 7 + b.
    
    for s in range(7):
        for b in range(7):
            col = s * 7 + b
            
            # U is the same for all blocks b (since A is the same).
            U[0, col] = U_coeffs[s][0]
            U[1, col] = U_coeffs[s][1]
            U[2, col] = U_coeffs[s][2]
            U[3, col] = U_coeffs[s][3]
            
            # V depends on block b.
            # V rows for block b:
            # b00 -> 0*p + 2b = 2b
            # b01 -> 0*p + 2b+1 = 2b+1
            # b10 -> 1*p + 2b = 14+2b
            # b11 -> 1*p + 2b+1 = 14+2b+1
            V[2*b, col] = V_coeffs[s][0]
            V[2*b+1, col] = V_coeffs[s][1]
            V[14+2*b, col] = V_coeffs[s][2]
            V[14+2*b+1, col] = V_coeffs[s][3]
            
            # W depends on block b.
            # W rows for block b:
            # c00 -> 4b
            # c01 -> 4b+2
            # c10 -> 4b+1
            # c11 -> 4b+3
            W[4*b, col] = W_coeffs[s][0]
            W[4*b+2, col] = W_coeffs[s][1]
            W[4*b+1, col] = W_coeffs[s][2]
            W[4*b+3, col] = W_coeffs[s][3]
    
    # Return as lists of lists to match the expected interface if strict types are required,
    # though the prompt says "factor matrices of shapes...". The previous code returned lists of lists.
    # The prompt example return type was {'U', 'V', 'W'}: factor matrices...
    # Usually numpy arrays are fine, but lists are safer if the evaluator does something weird.
    # However, the previous code used lists and got 1.1429.
    # I will return numpy arrays for efficiency and cleanliness, or lists to be safe?
    # The prompt says: "Keep the function names and signatures of the seed program."
    # Seed program returned: return {"U": U, "V": V, "W": W} where U,V,W were lists of lists.
    # I will return lists of lists to be exactly compatible.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
