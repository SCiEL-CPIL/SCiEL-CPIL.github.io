"""Matrix multiplication tensor decomposition search for 3x4x6."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    global START_TIME
    start_time = time.time()
    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    NM, MP, PN = N * M, M * P, P * N
    
    # Build target tensor T of shape (N*M, M*P, P*N)
    T = np.zeros((NM, MP, PN))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = i * M + j
                v = j * P + k
                w = k * N + i
                T[u, v, w] = 1.0
    
    def check_exact(U_list, V_list, W_list):
        """Check if the decomposition exactly reconstructs T."""
        if not U_list:
            return False
        r = len(U_list[0])
        U = np.array(U_list)
        V = np.array(V_list)
        W = np.array(W_list)
        
        T_rec = np.einsum('uc,vc,wc->uvw', U, V, W)
        return np.allclose(T_rec, T, atol=1e-9)
    
    # Strassen decomposition for 2x2x2
    # Returns U, V, W of shape (4, 7) each
    def strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Term 1: (a00+a11)(b00+b11)
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # Term 2: (a10+a11)b00
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1
        
        # Term 3: a00(b01-b11)
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        
        # Term 4: a11(b10-b00)
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        
        # Term 5: (a00+a01)b11
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[1,4] = 1; W[3,4] = -1
        
        # Term 6: (a10-a00)(b00+b01)
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # Term 7: (a01-a11)(b10+b11)
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1
        
        return U, V, W
    
    # Rank-1 decomposition for 1xm x px1 tensor (i.e., n=1)
    # T[i,j,k] = 1 for all i=0, j, k
    # U[0,0] = 1, V[j,0] = 1, W[k,0] = 1
    def rank1_1mp(M_local, P_local):
        U = np.zeros((M_local, 1))
        V = np.zeros((M_local * P_local, 1))
        W = np.zeros((P_local, 1))
        U[0, 0] = 1
        V[:, 0] = 1
        W[:, 0] = 1
        return U, V, W
    
    # Build the decomposition using additive structure
    # Partition i into {0} and {1,2}
    # Partition k into {0,1}, {2,3}, {4,5}
    # j remains as {0,1,2,3}
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Sub-tensors with i={0} (n_block=1)
    for k_start in [0, 2, 4]:
        k_end = k_start + 2
        # Local indices: i'=0, j'∈[0,4), k'∈[0,2)
        # Global: u = 0*4 + j' = j', v = j'*6 + (k_start + k'), w = (k_start+k')*3 + 0
        U_local, V_local, W_local = rank1_1mp(4, 2)
        
        # Map to global indices
        U_part = np.zeros((NM, 1))
        V_part = np.zeros((MP, 1))
        W_part = np.zeros((PN, 1))
        
        for j_prime in range(4):
            for k_prime in range(2):
                u_global = 0 * M + j_prime  # i=0
                v_global = j_prime * P + (k_start + k_prime)
                w_global = (k_start + k_prime) * N + 0  # i=0
                
                U_part[u_global, 0] += U_local[j_prime, 0]
                V_part[v_global, 0] += V_local[j_prime * 2 + k_prime, 0]
                W_part[w_global, 0] += W_local[k_prime, 0]
        
        U_parts.append(U_part)
        V_parts.append(V_part)
        W_parts.append(W_part)
    
    # Sub-tensors with i={1,2} (n_block=2)
    for k_start in [0, 2, 4]:
        k_end = k_start + 2
        # Split j into {0,1} and {2,3}
        for j_start in [0, 2]:
            j_end = j_start + 2
            # Local 2x2x2 tensor
            U_local, V_local, W_local = strassen_222()
            
            U_part = np.zeros((NM, 7))
            V_part = np.zeros((MP, 7))
            W_part = np.zeros((PN, 7))
            
            for i_prime in range(2):
                for j_prime in range(2):
                    for k_prime in range(2):
                        u_global = (1 + i_prime) * M + (j_start + j_prime)
                        v_global = (j_start + j_prime) * P + (k_start + k_prime)
                        w_global = (k_start + k_prime) * N + (1 + i_prime)
                        
                        # Local indices for 2x2x2: u_local = i_prime*2 + j_prime, etc.
                        u_local_idx = i_prime * 2 + j_prime
                        v_local_idx = j_prime * 2 + k_prime
                        w_local_idx = k_prime * 2 + i_prime
                        
                        U_part[u_global, :] += U_local[u_local_idx, :]
                        V_part[v_global, :] += V_local[v_local_idx, :]
                        W_part[w_global, :] += W_local[w_local_idx, :]
            
            U_parts.append(U_part)
            V_parts.append(V_part)
            W_parts.append(W_part)
    
    # Combine all parts
    U_full = np.hstack(U_parts)
    V_full = np.hstack(V_parts)
    W_full = np.hstack(W_parts)
    
    U_list = U_full.tolist()
    V_list = V_full.tolist()
    W_list = W_full.tolist()
    
    if check_exact(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
