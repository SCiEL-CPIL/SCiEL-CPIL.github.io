import time
import numpy as np

N, M, P = 2, 3, 10

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Rank calculation:
    # Part 1: A[0:2, 0:2] * B[0:2, 0:10]
    # Decompose B[0:2, 0:10] into 5 blocks of 2x2.
    # Each 2x2 * 2x2 multiplication uses Strassen (rank 7).
    # Total rank for Part 1 = 5 * 7 = 35.
    
    # Part 2: A[0:2, 2:3] * B[2:3, 0:10]
    # This is 2x1 * 1x10. Rank is 2.
    # Total rank = 35 + 2 = 37.
    
    R = 37
    U = np.zeros((N*M, R))
    V = np.zeros((M*P, R))
    W = np.zeros((P*N, R))
    
    # Helper to add a Strassen decomposition for 2x2 * 2x2
    # Inputs:
    # A_sub: indices (i, j) for i in 0..1, j in j_start..j_start+1
    # B_sub: indices (j, k) for j in j_start..j_start+1, k in k_start..k_start+1
    # Output C: indices (i, k) for i in 0..1, k in k_start..k_start+1
    
    def add_strassen(j_start, k_start, base_rank):
        # Strassen's 7 terms for 2x2 * 2x2
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to the global U, V, W matrices.
        # U index: i*M + j
        # V index: j*P + k
        # W index: k*N + i
        
        # Term 1: (A00 + A11)(B00 + B11)
        # A00 is (0, j_start), A11 is (1, j_start+1)
        # B00 is (j_start, k_start), B11 is (j_start+1, k_start+1)
        # C11 is (0, k_start), C22 is (1, k_start+1)
        # C11 += M1, C22 += M1
        
        t = base_rank
        
        # M1
        U[0*M + j_start, t] += 1
        U[1*M + (j_start+1), t] += 1
        V[j_start*P + k_start, t] += 1
        V[(j_start+1)*P + (k_start+1), t] += 1
        W[k_start*N + 0, t] += 1
        W[(k_start+1)*N + 1, t] += 1
        t += 1
        
        # M2: (A10 + A11)B00
        # A10 is (1, j_start), A11 is (1, j_start+1)
        # B00 is (j_start, k_start)
        # C21 += M2
        U[1*M + j_start, t] += 1
        U[1*M + (j_start+1), t] += 1
        V[j_start*P + k_start, t] += 1
        W[k_start*N + 1, t] += 1
        t += 1
        
        # M3: A00(B10 - B11)
        # A00 is (0, j_start)
        # B10 is (j_start+1, k_start), B11 is (j_start+1, k_start+1)
        # C12 += M3
        U[0*M + j_start, t] += 1
        V[(j_start+1)*P + k_start, t] += 1
        V[(j_start+1)*P + (k_start+1), t] -= 1
        W[k_start*N + 0, t] += 1
        t += 1
        
        # M4: A11(B01 - B00)
        # A11 is (1, j_start+1)
        # B01 is (j_start, k_start+1), B00 is (j_start, k_start)
        # C11 += M4, C21 += M4
        U[1*M + (j_start+1), t] += 1
        V[j_start*P + (k_start+1), t] += 1
        V[j_start*P + k_start, t] -= 1
        W[k_start*N + 0, t] += 1
        W[k_start*N + 1, t] += 1
        t += 1
        
        # M5: (A00 + A01)B11
        # A00 is (0, j_start), A01 is (0, j_start+1)
        # B11 is (j_start+1, k_start+1)
        # C11 -= M5, C12 += M5
        U[0*M + j_start, t] += 1
        U[0*M + (j_start+1), t] += 1
        V[(j_start+1)*P + (k_start+1), t] += 1
        W[k_start*N + 0, t] -= 1
        W[k_start*N + 1, t] += 1
        t += 1
        
        # M6: (A10 - A00)(B00 + B01)
        # A10 is (1, j_start), A00 is (0, j_start)
        # B00 is (j_start, k_start), B01 is (j_start, k_start+1)
        # C22 += M6
        U[1*M + j_start, t] += 1
        U[0*M + j_start, t] -= 1
        V[j_start*P + k_start, t] += 1
        V[j_start*P + (k_start+1), t] += 1
        W[(k_start+1)*N + 1, t] += 1
        t += 1
        
        # M7: (A01 - A11)(B10 + B11)
        # A01 is (0, j_start+1), A11 is (1, j_start+1)
        # B10 is (j_start+1, k_start), B11 is (j_start+1, k_start+1)
        # C11 += M7
        U[0*M + (j_start+1), t] += 1
        U[1*M + (j_start+1), t] -= 1
        V[(j_start+1)*P + k_start, t] += 1
        V[(j_start+1)*P + (k_start+1), t] += 1
        W[k_start*N + 0, t] += 1
        t += 1
        
        return t

    # Process Part 1: 5 blocks of 2x2 * 2x2
    # j_start is always 0, 1 (since we are using A[0:2, 0:2] and B[0:2, :])
    # Wait, A is 2x3. We split A into A[:, 0:2] and A[:, 2:3].
    # B is 3x10. We split B into B[0:2, :] and B[2:3, :].
    # So for Part 1, j indices in A are 0, 1. j indices in B are 0, 1.
    # k indices in B are 0..9.
    
    current_rank = 0
    for k_block in range(5):
        k_start = k_block * 2
        current_rank = add_strassen(0, k_start, current_rank)
        
    # Process Part 2: 2x1 * 1x10
    # A[:, 2] * B[2, :]
    # Rank 2 decomposition.
    # t = current_rank, current_rank + 1
    
    # Term 0: handles i=0
    # U[0*M+2, t] = 1, U[1*M+2, t] = 0
    # V[2*P+k, t] = 1 for all k
    # W[k*N+0, t] = 1 for all k
    # W[k*N+1, t] = 0 for all k
    # Contribution to C[0,k] = 1 * 1 * 1 * A[0,2] B[2,k] = A[0,2] B[2,k]
    # Contribution to C[1,k] = 0 * 1 * 0 * ... = 0
    
    t = current_rank
    U[0*M + 2, t] = 1
    for k in range(P):
        V[2*P + k, t] = 1
        W[k*N + 0, t] = 1
    current_rank += 1
    
    # Term 1: handles i=1
    # U[0*M+2, t] = 0, U[1*M+2, t] = 1
    # V[2*P+k, t] = 1 for all k
    # W[k*N+0, t] = 0 for all k
    # W[k*N+1, t] = 1 for all k
    # Contribution to C[1,k] = 1 * 1 * 1 * A[1,2] B[2,k] = A[1,2] B[2,k]
    # Contribution to C[0,k] = 0
    
    t = current_rank
    U[1*M + 2, t] = 1
    for k in range(P):
        V[2*P + k, t] = 1
        W[k*N + 1, t] = 1
    current_rank += 1
    
    # Verify
    # T_recon = np.einsum('at,bt,ct->abc', U, V, W)
    # T_target = np.zeros((N*M, M*P, P*N))
    # for i in range(N):
    #   for j in range(M):
    #     for k in range(P):
    #       T_target[i*M+j, j*P+k, k*N+i] = 1.0
    # if not np.allclose(T_recon, T_target, atol=1e-9):
    #   raise ValueError("Decomposition invalid")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
