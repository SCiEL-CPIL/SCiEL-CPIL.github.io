import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _construct_strassen(n, m, p, blocks):
    """Construct factor matrices for given Strassen blocks, with schoolbook for the rest."""
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Schoolbook for the rest
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    if 0 <= i0+di < n and 0 <= j0+dj < m and 0 <= k0+dk < p:
                        covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
                    
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(n, m, p, factors):
    """Verify that the factorization is correct."""
    U = np.array(factors["U"])
    V = np.array(factors["V"])
    W = np.array(factors["W"])
    rank = U.shape[1]
    
    # Check tensor: T_ijk = sum_c U_{i*m+j,c} V_{j*p+k,c} W_{k*n+i,c}
    # Should equal 1 for all i,j,k
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_col = U[:, c]  # shape (n*m,)
        v_col = V[:, c]  # shape (m*p,)
        w_col = W[:, c]  # shape (p*n,)
        
        # Reshape to 3D
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        
        # Contribution: outer product
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)


def construct():
    """Find rank-minimal tensor decomposition for 3x5 @ 5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin for 120s limit
    
    # Known best: rank 71 with 4 Strassen blocks
    # Let's try to find better configurations
    
    best_rank = n * m * p  # Schoolbook
    best_factors = None
    
    # Try various block configurations
    # Strassen blocks are 2x2x2, can be placed at any (i0, j0, k0) with i0 in [0, n-2], etc.
    
    valid_i0 = list(range(n - 1))  # [0, 1] for n=3
    valid_j0 = list(range(m - 1))  # [0, 1, 2, 3] for m=5
    valid_k0 = list(range(p - 1))  # [0, 1, 2, 3] for p=5
    
    # Try 4 non-overlapping Strassen blocks
    # Two blocks overlap if they share at least one (i, j, k) triple
    
    def blocks_overlap(b1, b2):
        i0, j0, k0 = b1
        i1, j1, k1 = b2
        # Check if any (i, j, k) is in both
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i = i0 + di
                    j = j0 + dj
                    k = k0 + dk
                    if i1 <= i <= i1+1 and j1 <= j <= j1+1 and k1 <= k <= k1+1:
                        return True
        return False
    
    # Enumerate possible sets of 4 non-overlapping blocks
    all_blocks = [(i0, j0, k0) for i0 in valid_i0 for j0 in valid_j0 for k0 in valid_k0]
    
    # For 3x5x5, we have 2*4*4 = 32 possible block positions
    # We want to find sets of 4 non-overlapping blocks that maximize coverage
    
    def count_coverage(blocks):
        """Count how many (i,j,k) triples are covered by these blocks."""
        covered = set()
        for i0, j0, k0 in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        i, j, k = i0+di, j0+dj, k0+dk
                        covered.add((i, j, k))
        return len(covered)
    
    def get_rank_for_blocks(blocks):
        """Compute rank for a set of Strassen blocks."""
        covered = count_coverage(blocks)
        rank = len(blocks) * 7 + (n * m * p - covered)
        return rank
    
    # Search for good block configurations
    best_blocks = None
    best_rank_found = float('inf')
    
    # First, try the original 4 blocks
    original_blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    for blocks in [original_blocks]:
        if not any(blocks_overlap(blocks[i], blocks[j]) for i in range(len(blocks)) for j in range(i+1, len(blocks))):
            rank = get_rank_for_blocks(blocks)
            if rank < best_rank_found:
                best_rank_found = rank
                best_blocks = blocks
    
    # Try other configurations
    # For each combination of 4 blocks, check if they're non-overlapping
    # This could be expensive, so let's be smart about it
    
    # Actually, let's try to maximize coverage with 4 blocks
    # We can greedily add blocks that cover the most new triples
    
    # Start with empty set and greedily add blocks
    from itertools import product
    
    # Try all possible first blocks, then greedily extend
    for first_block in all_blocks:
        candidate = [first_block]
        covered = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((first_block[0]+di, first_block[1]+dj, first_block[2]+dk))
        
        # Greedily add up to 3 more blocks
        remaining = [b for b in all_blocks if b != first_block]
        
        while len(candidate) < 4:
            best_next = None
            best_new_cover = 0
            for b in remaining:
                if any(blocks_overlap(candidate[i], b) for i in range(len(candidate))):
                    continue
                new_covered = set(covered)
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            new_covered.add((b[0]+di, b[1]+dj, b[2]+dk))
                new_count = len(new_covered - covered)
                if new_count > best_new_cover:
                    best_new_cover = new_count
                    best_next = b
            
            if best_next is None:
                break
            
            candidate.append(best_next)
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((best_next[0]+di, best_next[1]+dj, best_next[2]+dk))
            remaining = [b for b in remaining if b != best_next]
        
        rank = get_rank_for_blocks(candidate)
        if rank < best_rank_found:
            best_rank_found = rank
            best_blocks = candidate
    
    # Build the factorization with the best blocks found
    if best_blocks is not None:
        factors = _construct_strassen(n, m, p, best_blocks)
        factors["U"] = factors["U"][:n*m]  # Trim to correct size
        factors["V"] = factors["V"][:m*p]
        factors["W"] = factors["W"][:p*n]
        
        # Trim to actual rank
        actual_rank = factors["U"][0].__len__() if factors["U"] else 0
        # Find actual rank used
        non_zero_rank = 0
        for c in range(len(factors["U"][0])):
            if any(x != 0 for x in factors["U"][:, c] if isinstance(factors["U"], list)) or \
               any(factors["U"][i][c] != 0 for i in range(len(factors["U"]))):
                non_zero_rank = c + 1
        
        # Trim
        U = [row[:non_zero_rank] for row in factors["U"]]
        V = [row[:non_zero_rank] for row in factors["V"]]
        W = [row[:non_zero_rank] for row in factors["W"]]
        
        result = {"U": U, "V": V, "W": W}
        
        if _verify_decomposition(n, m, p, result):
            return result
    else:
        # Fallback to schoolbook
        U = [[0.0]*(n*m*p) for _ in range(n*m)]
        V = [[0.0]*(n*m*p) for _ in range(m*p)]
        W = [[0.0]*(n*m*p) for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
        return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    # Verify
    T = np.zeros((n, m, p))
    rank = U.shape[1]
    for c in range(rank):
        u_2d = U[:, c].reshape(n, m)
        v_2d = V[:, c].reshape(m, p)
        w_2d = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
    expected = np.ones((n, m, p))
    if np.allclose(T, expected):
        print("Valid")
    else:
        print("Invalid")
