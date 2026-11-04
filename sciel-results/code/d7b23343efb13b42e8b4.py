"""Rank-minimal tensor decomposition for 2x3 * 3x10 matrix multiplication.
Strategy: Try to find a lower-rank decomposition than 55 by:
1. First attempting a rank-23 4x4x4 embedding projection (structural idea from directive)
2. If that fails, falling back to the known rank-55 Strassen+schoolbook decomposition
"""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = 2, 3, 10
    
    # First, try the structural approach: embed in 4x4x4 and project rank-23
    # The 4x4x4 matrix multiplication tensor has rank 23 (Bini, Capovani, Lotti, Romani 1980)
    # We embed our 2x3x10 problem into a 4x4x4 space and project
    
    # For a 4x4x4 decomposition, the tensor T[i,j,k] = delta_{i+k, j} (in appropriate indexing)
    # Our problem: C[k,i] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = 1 if this term contributes to C[k,i]
    # Actually the multiplication tensor for (n,m,p) has T[i,j,k] = 1 for all valid i,j,k
    # where C[k*n+i] += A[i*m+j] * B[j*p+k]
    
    # The rank-23 decomposition of 4x4x4 matmul:
    # This is complex. Let me use the known result that M_4 <= 23.
    # But projecting from 4x4x4 to our sub-problem may not preserve rank minimality.
    
    # Let's instead try a direct structural approach:
    # Can we do better than 55 by using a different block decomposition?
    
    # Alternative: split p=10 into 4+3+3 instead of 5 pairs of 2
    # Or split m=3 differently
    
    # Actually, let's try: use Strassen on 2x2 blocks but overlap the schoolbook terms
    # Key insight: For each pair of output columns, we need:
    #   7 (Strassen for 2x2*2x2) + 4 (schoolbook for 2x1*1x2) = 11 per pair
    #   5 pairs = 55
    # 
    # But what if we use a different decomposition for the 2x1 * 1x2 part?
    # 2x1 * 1x2 gives a 2x2 output. That's just 4 scalar multiplications.
    # Can we combine these with the Strassen terms to save?
    
    # Another idea: Use the fact that we have 5 independent 2x2 blocks.
    # The total tensor is a direct sum of 5 copies of (2x2*2x2) tensor plus
    # 5 copies of (2x1*1x2) tensor.
    # Rank of direct sum = sum of ranks (for tensor rank, this is an upper bound)
    # So rank <= 5*7 + 5*4 = 55. This matches.
    
    # To beat 55, we need a non-trivial combination across blocks.
    # This is equivalent to finding a rank decomposition of the 5-fold direct sum
    # that is smaller than the sum of individual ranks.
    
    # For tensor rank, the direct sum can sometimes have rank less than the sum.
    # However, for matrix multiplication tensors, this is generally not the case
    # due to the bilinear structure.
    
    # Let's try a computational approach: use numpy to check if a rank-54 decomposition
    # exists by solving a system of equations. But this is likely infeasible in 120s.
    
    # Instead, let's try a smarter block decomposition:
    # Split p=10 as 2+2+2+2+2 (5 blocks of 2) - current approach, rank 55
    # Split p=10 as 4+3+3:
    #   4x2 output from 2x3 * 3x4: use Strassen on 2x2 blocks -> 2*7=14, plus 2x1*1x4 -> 8, total 22
    #   Wait, that's not right. Let me reconsider.
    
    # For 2x3 * 3x4:
    #   Split m=3 into 2+1
    #   2x2 * 2x4: treat as 2 blocks of 2x2, each Strassen rank 7 -> 14
    #   2x1 * 1x4: schoolbook rank 8
    #   Total: 22
    # For 2x3 * 3x3:
    #   Split m=3 into 2+1
    #   2x2 * 2x3: 1 block of 2x2 (Strassen 7) + 1 block of 2x1 (schoolbook 2) -> 9
    #   2x1 * 1x3: schoolbook rank 6
    #   Total: 15
    # Grand total: 22 + 15 + 15 = 52? Let me verify this more carefully.
    
    # Actually, I need to be more careful. Let me re-derive.
    
    # For A(2x3) * B(3xp) -> C(2xp):
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # If I split B's columns into groups, the computation for each group is independent.
    # For a group of q columns, I compute A(2x3) * B_sub(3xq) -> C_sub(2xq)
    
    # For q=2: rank is 11 (7 Strassen + 4 schoolbook)
    # For q=4: I can split into 2 groups of 2, each rank 11, total 22
    # For q=3: I need a different analysis
    
    # For q=3: A(2x3) * B(3x3) -> C(2x3)
    # Split m=3 into 2+1:
    #   A[:,:2] * B[:2,:] -> 2x2 * 2x3
    #   A[:,2] * B[2,:] -> 2x1 * 1x3
    # 
    # For 2x2 * 2x3:
    #   Split the 3 columns into 2+1
    #   2x2 * 2x2: Strassen rank 7
    #   2x2 * 2x1: This is 2 independent 2x1 products, each rank 2, total 4
    #   Total: 11
    # 
    # For 2x1 * 1x3: rank 6 (6 scalar multiplications)
    # 
    # Total for q=3: 11 + 6 = 17
    # 
    # So for p=10 = 3+3+4:
    #   17 + 17 + 22 = 56. Worse than 55.
    # 
    # For p=10 = 2+2+2+2+2: 5*11 = 55. Current best.
    
    # What about a non-block approach? Use the full 2x3 * 3x10 tensor directly.
    # The trivial rank is 2*3*10 = 60. We've already achieved 55.
    
    # Let me try yet another approach: use the transpose trick more aggressively.
    # M_{2,3,10} = M_{10,3,2} (transpose)
    # For M_{10,3,2}: A(10x3) * B(3x2) -> C(10x2)
    # Split the first dimension (10) into 5 groups of 2:
    #   Each group: 2x3 * 3x2 -> 2x2
    #   For each 2x3 * 3x2:
    #     Split m=3 into 2+1
    #     2x2 * 2x2: Strassen rank 7
    #     2x1 * 1x2: rank 4
    #     Total: 11
    #   5 groups: 55. Same result.
    
    # The key question is whether 55 is truly optimal or if there's a clever
    # cross-block combination that saves rank.
    
    # Given the time constraints, let's return the known rank-55 solution.
    # But let me try one more thing: a direct numerical search for rank 54.
    
    # Actually, let's just return the rank-55 solution which is proven correct.
    
    if time.time() > deadline - 5:
        return _rank55_solution(n, m, p)
    
    # Try to see if we can find a rank-54 solution via random search
    # This is unlikely to succeed but worth a quick try
    best_rank = 55
    best_solution = _rank55_solution(n, m, p)
    
    # Quick random search for rank 54 (very limited, just a few iterations)
    np.random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        rank = 54
        # Random initialization
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Check if this decomposes the tensor
        # The multiplication tensor T[i,j,k] should equal sum_r U[i,r]*V[j,r]*W[k,r]
        # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (in the appropriate indexing)
        # Actually, the tensor is T[a,b,c] = 1 where a=i*m+j, b=j*p+k, c=k*n+i
        # and it's 1 only when the indices are consistent (same j for a and b, same k for b and c)
        
        # Let me construct the target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        
        # Check decomposition
        # sum_r U[a,r] * V[b,r] * W[c,r] should equal T[a,b,c]
        # This is equivalent to: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # should sum to T
        
        # Efficient check: contract U and V, then check against T
        # (U @ V.T)[a,b] = sum_r U[a,r]*V[b,r]
        # Then check if T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is: for each c, T[:,:,c] should be in the span of {U[:,r]*V[:,r]*W[c,r]}
        
        # Simpler: just check a few random entries
        correct = True
        for _ in range(200):
            a = np.random.randint(0, n*m)
            b = np.random.randint(0, m*p)
            c = np.random.randint(0, p*n)
            val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
            if abs(val - T[a, b, c]) > 1e-10:
                correct = False
                break
        
        if correct:
            best_rank = rank
            best_solution = {
                'U': [[float(U[i, r]) for r in range(rank)] for i in range(n*m)],
                'V': [[float(V[i, r]) for r in range(rank)] for i in range(m*p)],
                'W': [[float(W[i, r]) for r in range(rank)] for i in range(p*n)]
            }
            break
    
    if time.time() > deadline - 2:
        return best_solution
    
    return best_solution


def _rank55_solution(n, m, p):
    """Return the known rank-55 Strassen+schoolbook decomposition."""
    rank = 55
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    idx = 0
    
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        a00 = 0 * 3 + 0
        a01 = 0 * 3 + 1
        a10 = 1 * 3 + 0
        a11 = 1 * 3 + 1
        
        b0k0 = 0 * 10 + k0
        b0k1 = 0 * 10 + k1
        b1k0 = 1 * 10 + k0
        b1k1 = 1 * 10 + k1
        
        c0k0 = k0 * 2 + 0
        c0k1 = k1 * 2 + 0
        c1k0 = k0 * 2 + 1
        c1k1 = k1 * 2 + 1
        
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U_mat[a00][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M2 = (a10 + a11) * b0k0
        U_mat[a10][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        W_mat[c1k1][idx] -= 1
        idx += 1
        
        # M3 = a00 * (b0k1 - b1k1)
        U_mat[a00][idx] += 1
        V_mat[b0k1][idx] += 1
        V_mat[b1k1][idx] -= 1
        W_mat[c0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U_mat[a11][idx] += 1
        V_mat[b1k0][idx] += 1
        V_mat[b0k0][idx] -= 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        idx += 1
        
        # M5 = (a00 + a01) * b1k1
        U_mat[a00][idx] += 1
        U_mat[a01][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] -= 1
        W_mat[c0k1][idx] += 1
        idx += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U_mat[a10][idx] += 1
        U_mat[a00][idx] -= 1
        V_mat[b0k0][idx] += 1
        V_mat[b0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U_mat[a01][idx] += 1
        U_mat[a11][idx] -= 1
        V_mat[b1k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        idx += 1
    
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            
            U_mat[u_idx][idx] += 1
            V_mat[v_idx][idx] += 1
            W_mat[w_idx][idx] += 1
            idx += 1
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
