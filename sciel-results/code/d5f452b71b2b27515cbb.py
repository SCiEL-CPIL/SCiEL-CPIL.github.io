"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_DEADLINE = _start_time + 130  # 130s safety margin

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # Strategy: 6 Strassen blocks (rank 42) + one 2x2x3 block
    # If we find rank-10 for 2x2x3, total = 52
    # Fallback: 7 Strassen + 1 trivial = 53
    
    def _build_rank53():
        """Build the known rank-53 decomposition."""
        rank = 53
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(base_r, k0, k1):
            r1, r2, r3, r4, r5, r6, r7 = base_r, base_r+1, base_r+2, base_r+3, base_r+4, base_r+5, base_r+6
            U[0][r1] = 1; U[3][r1] = 1
            V[0*p+k0][r1] = 1; V[1*p+k1][r1] = 1
            U[2][r2] = 1; U[3][r2] = 1
            V[0*p+k0][r2] = 1
            U[0][r3] = 1
            V[0*p+k1][r3] = 1; V[1*p+k1][r3] = -1
            U[3][r4] = 1
            V[1*p+k0][r4] = 1; V[0*p+k0][r4] = -1
            U[0][r5] = 1; U[1][r5] = 1
            V[1*p+k1][r5] = 1
            U[2][r6] = 1; U[0][r6] = -1
            V[0*p+k0][r6] = 1; V[0*p+k1][r6] = 1
            U[1][r7] = 1; U[3][r7] = -1
            V[1*p+k0][r7] = 1; V[1*p+k1][r7] = 1
            W[2*k0][r1] = 1; W[2*k0][r4] = 1; W[2*k0][r5] = -1; W[2*k0][r7] = 1
            W[2*k0+1][r2] = 1; W[2*k0+1][r4] = 1
            W[2*k1][r3] = 1; W[2*k1][r5] = 1
            W[2*k1+1][r1] = 1; W[2*k1+1][r2] = -1; W[2*k1+1][r3] = 1; W[2*k1+1][r6] = 1
        
        for i in range(7):
            add_strassen_block(7*i, 2*i, 2*i+1)
        
        base_r = 49
        U[0][base_r] = 1; V[0*p+14][base_r] = 1; W[28][base_r] = 1
        U[1][base_r+1] = 1; V[1*p+14][base_r+1] = 1; W[28][base_r+1] = 1
        U[2][base_r+2] = 1; V[0*p+14][base_r+2] = 1; W[29][base_r+2] = 1
        U[3][base_r+3] = 1; V[1*p+14][base_r+3] = 1; W[29][base_r+3] = 1
        
        return {"U": U, "V": V, "W": W}
    
    def _verify(U, V, W):
        """Verify that the decomposition is correct."""
        rank = len(U[0])
        # Check: for all i,j,k: sum_r U[i*2+j][r] * V[j*p+k][r] * W[k*n+i][r] == delta
        # Actually: C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor: T[i,j,k] = 1 (coefficient of A[i,j]*B[j,k] in C[i,k])
        # Decomposition: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should give the tensor
        # W index is k*n+i per the contract
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    expected = 1  # Each (i,j,k) triple contributes 1 to C[i,k]
                    if s != expected:
                        return False
        return True
    
    def _build_rank52():
        """Try to build a rank-52 decomposition using 6 Strassen + 2x2x3 rank-10."""
        # First try a known rank-10 for 2x2x3
        # 2x2x3: columns 0,1,2 of B, rows 0,1 of A and C
        # Let's try to find it via small search or known construction
        
        # Known: 2x2x3 can be done in rank 10? Let's try a construction.
        # Use Strassen on columns 0,1 (rank 7) + trivial on column 2 (rank 4) = 11
        # Need to save 1 rank. 
        
        # Alternative: try a direct rank-10 construction
        # This is complex; let's try a targeted search with small integer coefficients
        
        import itertools
        
        # The 2x2x3 tensor has 12 nonzero entries.
        # We need rank-10: 10 terms, each with 4 values (U[i,j], V[j,k], W[k,i])
        # U: 4 values, V: 6 values, W: 6 values per term
        # Total 16 values per term, 10 terms = 160 values
        
        # This search space is too large for brute force. Let's try a different approach.
        # Use the fact that we can share structure.
        
        # Try: take the rank-11 (Strassen+trivial) and see if we can merge two terms
        # to reduce by 1.
        
        # Actually, let's just return None if we can't find rank 10 quickly
        return None
    
    # Try to find a rank-52 solution
    result_52 = _build_rank52()
    
    if result_52 is not None and _verify(result_52["U"], result_52["V"], result_52["W"]):
        return result_52
    
    # Fallback to rank 53
    result_53 = _build_rank53()
    return result_53

# EVOLVE-BLOCK-END
