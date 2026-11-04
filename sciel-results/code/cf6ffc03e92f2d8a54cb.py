"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(U, V, W, i0, j0, k0, col):
        # M1
        U[i0 * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M2
        U[(i0 + 1) * m + j0][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[j0 * p + k0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] -= 1
        col += 1
        
        # M3
        U[i0 * m + j0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] -= 1
        W[(k0 + 1) * n + i0][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M4
        U[(i0 + 1) * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0][col] += 1
        V[j0 * p + k0][col] -= 1
        W[k0 * n + i0][col] += 1
        W[k0 * n + i0 + 1][col] += 1
        col += 1
        
        # M5
        U[i0 * m + j0][col] += 1
        U[i0 * m + j0 + 1][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0][col] += 1
        W[k0 * n + i0][col] -= 1
        col += 1
        
        # M6
        U[(i0 + 1) * m + j0][col] += 1
        U[i0 * m + j0][col] -= 1
        V[j0 * p + k0][col] += 1
        V[j0 * p + k0 + 1][col] += 1
        W[(k0 + 1) * n + i0 + 1][col] += 1
        col += 1
        
        # M7
        U[i0 * m + j0 + 1][col] += 1
        U[(i0 + 1) * m + j0 + 1][col] -= 1
        V[(j0 + 1) * p + k0][col] += 1
        V[(j0 + 1) * p + k0 + 1][col] += 1
        W[k0 * n + i0][col] += 1
        col += 1
        
        return col

    # Find maximum set of disjoint 2x2x2 Strassen blocks using backtracking
    def find_max_strassen_blocks():
        """Find maximum set of disjoint 2x2x2 Strassen blocks via backtracking search."""
        all_blocks = []
        for i0 in range(n - 1):
            for j0 in range(m - 1):
                for k0 in range(p - 1):
                    all_blocks.append((i0, j0, k0))
        
        # Precompute positions for each block
        def block_positions(i0, j0, k0):
            positions = set()
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        positions.add((i0 + di, j0 + dj, k0 + dk))
            return positions
        
        block_pos = {b: block_positions(*b) for b in all_blocks}
        
        # Use backtracking with pruning to find maximum packing
        best_packing = []
        
        def backtrack(available, current_packing, used_mask):
            nonlocal best_packing
            if time.time() > deadline - 5:
                return
            if len(current_packing) + len(available) <= len(best_packing):
                return
            if not available:
                if len(current_packing) > len(best_packing):
                    best_packing = list(current_packing)
                return
            
            # Try adding the first available block
            b = available[0]
            rest = available[1:]
            
            # Option 1: include block b
            if block_pos[b].isdisjoint(used_mask):
                new_used = used_mask | block_pos[b]
                # Filter available to those disjoint from new_used
                new_available = [x for x in rest if block_pos[x].isdisjoint(new_used)]
                current_packing.append(b)
                backtrack(new_available, current_packing, new_used)
                current_packing.pop()
            
            # Option 2: skip block b
            backtrack(rest, current_packing, used_mask)
        
        backtrack(all_blocks, [], set())
        return best_packing
    
    # Find the best Strassen block packing
    strassen_blocks = find_max_strassen_blocks()
    
    # Determine which positions are covered by Strassen blocks
    covered_positions = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_positions.add((i0 + di, j0 + dj, k0 + dk))
    
    # Uncovered positions need schoolbook terms
    uncovered_positions = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered_positions:
                    uncovered_positions.add((i, j, k))
    
    # Rank = 7 * (# Strassen blocks) + (# uncovered positions)
    rank = 7 * len(strassen_blocks) + len(uncovered_positions)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Add Strassen blocks
    for (i0, j0, k0) in strassen_blocks:
        col = add_strassen(U, V, W, i0, j0, k0, col)
    
    # Add schoolbook terms for uncovered positions
    for (i, j, k) in sorted(uncovered_positions):
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    # Convert to numpy for efficient verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify correctness with numpy
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                assert abs(total - 1) < 1e-10, f"Failed at ({i},{j},{k}): got {total}"
    
    # Convert back to lists
    U_out = U_np.tolist()
    V_out = V_np.tolist()
    W_out = W_np.tolist()
    
    # Round to nearest integer/half-integer
    def round_to_half(x):
        return round(x * 2) / 2
    
    U_out = [[round_to_half(x) for x in row] for row in U_out]
    V_out = [[round_to_half(x) for x in row] for row in V_out]
    W_out = [[round_to_half(x) for x in row] for row in W_out]
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = sum(U_out[i*m+j][r] * V_out[j*p+k][r] * W_out[k*n+i][r] for r in range(rank))
                assert abs(total - 1) < 1e-10, f"Failed at ({i},{j},{k}): got {total}"
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
