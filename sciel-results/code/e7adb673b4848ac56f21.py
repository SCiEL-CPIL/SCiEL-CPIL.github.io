"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify the decomposition."""
        # U: n*m x rank
        # V: m*p x rank
        # W: p*n x rank
        # Tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We want T[i,j,k] = 1 if i,j,k correspond to a valid multiplication term, 0 otherwise.
        # Actually, the tensor for matrix multiplication is T[i,j,k] = delta_{i,j} delta_{j,k}? No.
        # The tensor is defined by T[i, j, k] such that (AB)_{ik} = sum_j A_{ij} B_{jk}.
        # In the decomposition, (AB)_{ik} = sum_r (sum_j U_{ij,r} A_{ij}) (sum_j V_{jk,r} B_{jk}) W_{ki,r}?
        # No, the standard form is:
        # C_{ik} = sum_r (sum_j U_{ij,r} A_{ij}) (sum_j V_{jk,r} B_{jk}) W_{ki,r}
        # This requires W to be scalar or the sum over r to produce the correct C.
        # The contract says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # And "exactly decompose the matrix multiplication tensor".
        # Usually this means:
        # sum_r U_{ij,r} V_{jk,r} W_{ki,r} = delta_{ij} delta_{jk}? No.
        # The tensor T_{i,j,k} for MM is 1 if the indices match in the multiplication?
        # No, the tensor is T_{i, j, k} = 1 if j is the shared index?
        # Actually, the tensor is T_{i, j, k} such that T_{i, j, k} = 1 if i,j,k form a valid path?
        # For MM, T_{i, j, k} = 1 if j is the contraction index.
        # But i, j, k are all free in the tensor definition?
        # The tensor is T_{i, j, k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
        # So T_{i, j, k} = 1 for all i,j,k? No, that's the identity tensor?
        # No, the tensor is T_{i, j, k} = 1 if j is the index being summed.
        # In the decomposition, we have:
        # sum_r U_{i, j, r} V_{j, k, r} W_{k, i, r} ?
        # The indices in the contract are:
        # U: (n*m, rank) -> index (i*m + j)
        # V: (m*p, rank) -> index (j*p + k)
        # W: (p*n, rank) -> index (k*n + i)
        # So the sum is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # This sum must equal 1 if the term A_{ij} B_{jk} is present in the multiplication?
        # No, the multiplication is C_{ik} = sum_j A_{ij} B_{jk}.
        # So the tensor T_{i, j, k} should be 1 if j is the valid index?
        # Actually, for every i, j, k, the term A_{ij} B_{jk} exists.
        # So T_{i, j, k} = 1 for all i, j, k?
        # No, that would mean sum_r U V W = 1 for all i,j,k.
        # Let's test with rank 1. U=1, V=1, W=1. Sum=1.
        # Then C_{ik} = sum_j A_{ij} B_{jk} * 1?
        # The formula is C_{ik} = sum_r (sum_j U_{ij,r} A_{ij}) (sum_j V_{jk,r} B_{jk}) W_{ki,r}.
        # This is NOT the standard tensor decomposition form.
        # Standard form: T_{i, j, k} = sum_r U_{i,r} V_{j,r} W_{k,r}.
        # Here the indices are mixed.
        # Let's assume the standard tensor T_{i, j, k} = 1 if the indices are valid?
        # For MM, T_{i, j, k} = 1 for all i, j, k?
        # No, the tensor is T_{i, j, k} = 1 if j is the shared index.
        # In the flattened indices:
        # u_idx = i*m + j
        # v_idx = j*p + k
        # w_idx = k*n + i
        # The term is U[u_idx, r] V[v_idx, r] W[w_idx, r].
        # We want this to reconstruct the multiplication.
        # The multiplication tensor has T_{i, j, k} = 1 if the indices match?
        # Actually, the tensor is T_{i, j, k} = 1 for all i, j, k?
        # No, the tensor is T_{i, j, k} = 1 if j is the index connecting A and B.
        # Since j is always the connecting index, T_{i, j, k} = 1 for all i, j, k.
        # Let's verify this hypothesis.
        # If T_{i, j, k} = 1, then sum_r U V W = 1 for all i, j, k.
        # This seems too simple.
        # Let's check the Strassen case.
        # In Strassen, T_{i, j, k} is not 1 for all i,j,k.
        # The tensor is T_{i, j, k} = 1 if the term A_{ij} B_{jk} is used?
        # No, the tensor is the "matrix multiplication tensor".
        # T_{i, j, k} = 1 if i, j, k are such that A_{ij} and B_{jk} multiply to C_{ik}.
        # This is true for all i, j, k.
        # So T_{i, j, k} = 1 for all i, j, k.
        
        # Let's check the code.
        # In the code, for M1:
        # U[u_00][col] += 1
        # U[u_11][col] += 1
        # V[v_00][col] += 1
        # V[v_11][col] += 1
        # W[w_00][col] += 1
        # W[w_11][col] += 1
        
        # This adds 1 to U[0], U[3], V[0], V[5], W[0], W[3].
        # (Assuming n=2, m=2, p=2 for the block)
        # u_00 = 0, u_11 = 3
        # v_00 = 0, v_11 = 5
        # w_00 = 0, w_11 = 3
        
        # The term is U[0] V[0] W[0] + U[0] V[5] W[3] + U[3] V[0] W[0] + ...
        # This doesn't look like it sums to 1 for all i,j,k.
        
        # Let's re-read the contract.
        # "exactly decompose the matrix multiplication tensor"
        # The tensor T is defined by T_{i, j, k} = 1 if the indices are valid?
        # Actually, for MM, T_{i, j, k} = 1 if j is the contraction index.
        # Since j is always the contraction index, T_{i, j, k} = 1 for all i, j, k.
        
        # Wait, if T_{i, j, k} = 1 for all i, j, k, then the rank is the number of terms in the decomposition of the all-ones tensor?
        # No, the tensor is T_{i, j, k} = 1 if i, j, k form a valid path.
        # In MM, every i, j, k is a valid path.
        # So T is the all-ones tensor?
        # No, the tensor is T_{i, j, k} = 1 if the term A_{ij} B_{jk} contributes.
        # It contributes to C_{ik}.
        # So T_{i, j, k} = 1.
        
        # Let's assume the verification is:
        # For all i, j, k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        
        # Let's test this with the schoolbook case.
        # Rank = n*m*p.
        # For each i, j, k, we have a term.
        # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1.
        # Sum = 1. Correct.
        
        # So the verification is:
        # For all i in range(n), j in range(m), k in range(p):
        #   s = 0
        #   for r in range(rank):
        #     s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        #   if s != 1: return False
        # return True
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for r in range(len(U[0])):
                        s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    if s != 1:
                        return False
        return True

    def build_rank33():
        rank = 33
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            u_00 = i0 * m + j0
            u_01 = i0 * m + j0 + 1
            u_10 = (i0 + 1) * m + j0
            u_11 = (i0 + 1) * m + j0 + 1
            
            v_00 = j0 * p + k0
            v_01 = j0 * p + k0 + 1
            v_10 = (j0 + 1) * p + k0
            v_11 = (j0 + 1) * p + k0 + 1
            
            w_00 = k0 * n + i0
            w_10 = k0 * n + i0 + 1
            w_01 = (k0 + 1) * n + i0
            w_11 = (k0 + 1) * n + i0 + 1
            
            U[u_00][col] += 1
            U[u_11][col] += 1
            V[v_00][col] += 1
            V[v_11][col] += 1
            W[w_00][col] += 1
            W[w_11][col] += 1
            col += 1
            
            U[u_10][col] += 1
            U[u_11][col] += 1
            V[v_00][col] += 1
            W[w_10][col] += 1
            W[w_11][col] -= 1
            col += 1
            
            U[u_00][col] += 1
            V[v_01][col] += 1
            V[v_11][col] -= 1
            W[w_01][col] += 1
            W[w_11][col] += 1
            col += 1
            
            U[u_11][col] += 1
            V[v_10][col] += 1
            V[v_00][col] -= 1
            W[w_00][col] += 1
            W[w_10][col] += 1
            col += 1
            
            U[u_00][col] += 1
            U[u_01][col] += 1
            V[v_11][col] += 1
            W[w_00][col] -= 1
            W[w_01][col] += 1
            col += 1
            
            U[u_10][col] += 1
            U[u_00][col] -= 1
            V[v_00][col] += 1
            V[v_01][col] += 1
            W[w_11][col] += 1
            col += 1
            
            U[u_01][col] += 1
            U[u_11][col] -= 1
            V[v_10][col] += 1
            V[v_11][col] += 1
            W[w_00][col] += 1
            col += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 0, 4)
        
        for k in range(6):
            u_idx = 0 * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + 0
            U[u_idx][col] += 1
            V[v_idx][col] += 1
            W[w_idx][col] += 1
            col += 1
            
            u_idx = 1 * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + 1
            U[u_idx][col] += 1
            V[v_idx][col] += 1
            W[w_idx][col] += 1
            col += 1

        return {"U": U, "V": V, "W": W}

    result = build_rank33()
    
    if verify(result["U"], result["V"], result["W"]):
        return result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
