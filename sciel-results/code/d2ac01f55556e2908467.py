import time
import numpy as np
from scipy.optimize import linear_sum_assignment

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS
SCHOOLBOOK = N * M * P  # 72

def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[a,c] V[b,c] W[c] == 1 iff a,b,c correspond to a product."""
    try:
        rank = len(U[0])
        for a in range(N * M):
            for b in range(M * P):
                acc = [0.0] * (P * N)
                for c in range(rank):
                    u = U[a][c]
                    if u == 0:
                        continue
                    v = V[b][c]
                    if v == 0:
                        continue
                    uv = u * v
                    if uv == 0:
                        continue
                    wc = W[c]
                    for k in range(P * N):
                        acc[k] += uv * wc[k]
                i, j = divmod(a, M)
                j2, k2 = divmod(b, P)
                for kk in range(P * N):
                    expected = 1.0 if (kk % N == i and kk // N == j2) else 0.0
                    if abs(acc[kk] - expected) > tol:
                        return False
        return True
    except Exception:
        return False

def _build_strassen_block(u_base, v_base, w_base):
    """
    Returns lists of (u_idx, v_idx, w_idx, coeff) for a 2x2 Strassen block.
    u_base: [a00, a01, a10, a11]
    v_base: [b00, b01, b10, b11]
    w_base: [c00, c01, c10, c11]
    """
    a00, a01, a10, a11 = u_base
    b00, b01, b10, b11 = v_base
    c00, c01, c10, c11 = w_base
    
    terms = []
    
    # M1 = (a00+a11)(b00+b11) -> c00+=1, c11+=1, c01-=1, c10-=1
    # We represent each term as a tuple of (u_indices, v_indices, w_indices, coeffs)
    # Actually, for the matching, we need individual rank-1 tensors.
    # Strassen terms are sums of products. 
    # M1 = (a00+a11)(b00+b11) is NOT a rank-1 tensor in the standard basis of A and B.
    # Wait, the decomposition is sum_c U_c V_c W_c.
    # U_c is a vector in A-space, V_c in B-space, W_c in C-space.
    # A rank-1 tensor is u v^T w^T.
    # Strassen's M1 corresponds to the term: 
    # u = e_a00 + e_a11
    # v = e_b00 + e_b11
    # w = e_c00 + e_c11 - e_c01 - e_c10
    # This IS a valid rank-1 term in the tensor decomposition sense (it contributes to multiple C entries).
    
    # So we just need to list the 7 Strassen terms as (U_vec, V_vec, W_vec).
    
    # M1
    U1 = {a00: 1, a11: 1}
    V1 = {b00: 1, b11: 1}
    W1 = {c00: 1, c11: 1, c01: -1, c10: -1}
    terms.append((U1, V1, W1))
    
    # M2 = (a10+a11)b00 -> c10+=1, c11-=1
    U2 = {a10: 1, a11: 1}
    V2 = {b00: 1}
    W2 = {c10: 1, c11: -1}
    terms.append((U2, V2, W2))
    
    # M3 = a00(b01-b11) -> c01+=1, c11+=1
    U3 = {a00: 1}
    V3 = {b01: 1, b11: -1}
    W3 = {c01: 1, c11: 1}
    terms.append((U3, V3, W3))
    
    # M4 = a11(b10-b00) -> c00+=1, c10+=1
    U4 = {a11: 1}
    V4 = {b10: 1, b00: -1}
    W4 = {c00: 1, c10: 1}
    terms.append((U4, V4, W4))
    
    # M5 = (a00+a01)b11 -> c00-=1, c01+=1
    U5 = {a00: 1, a01: 1}
    V5 = {b11: 1}
    W5 = {c00: -1, c01: 1}
    terms.append((U5, V5, W5))
    
    # M6 = (a10-a00)(b00+b01) -> c11+=1
    U6 = {a10: 1, a00: -1}
    V6 = {b00: 1, b01: 1}
    W6 = {c11: 1}
    terms.append((U6, V6, W6))
    
    # M7 = (a01-a11)(b10+b11) -> c00+=1, c10-=1
    U7 = {a01: 1, a11: -1}
    V7 = {b10: 1, b11: 1}
    W7 = {c00: 1, c10: -1}
    terms.append((U7, V7, W7))
    
    return terms

def _build_direct_term(u_idx, v_idx, w_idx):
    """Build a single direct multiplication term."""
    return [({u_idx: 1}, {v_idx: 1}, {w_idx: 1})]

def construct():
    start = time.time()
    deadline = start + 140.0
    
    # 1. Generate candidate terms
    # We will try to cover the "essential" products with Strassen-like blocks or direct terms.
    # The "essential" products are the 72 entries of C.
    # However, Strassen blocks cover 4 C entries each (with signs).
    
    # Let's define the blocks we want to consider.
    # We can mix and match.
    # Strategy: 
    # - Identify all possible 2x2 sub-blocks of A and B.
    # - For each, we can either use Strassen (7 terms) or Direct (4 terms).
    # - We need to cover all C entries exactly once (linear combination sums to 1).
    # - This is a Set Covering / Packing problem.
    
    # Since the problem is small, we can enumerate "block decompositions".
    # A block decomposition is a way to partition the C matrix into sub-rectangles 
    # that are products of sub-matrices of A and B.
    
    # Let's stick to the partition used in the seed but allow "Direct" vs "Strassen" choice for 2x2 blocks.
    # Partition:
    # A is 3x3, B is 3x8.
    # C is 3x8.
    
    # We can split C into:
    # - Top-Left 2x2: A[0:2, 0:2] * B[0:2, 0:2]
    # - Top-Mid 2x2: A[0:2, 0:2] * B[0:2, 2:4] ... wait, B is 3x8.
    
    # Let's use a recursive or iterative search over the "tiling" of the C matrix.
    # But that's complex. Let's try a simpler heuristic:
    # Start with the known 68-rank tiling.
    # For each 2x2 block in that tiling, decide: Strassen (7) or Direct (4).
    # If we switch a 2x2 block from Strassen to Direct, rank increases by 3.
    # If we switch a 2x1 or 1x2 block...
    
    # Actually, the "MWPM" hint suggests finding a global optimum.
    # Let's formulate it as:
    # We have a set of "tiles" (sub-matrix multiplications).
    # Each tile covers a subset of C entries.
    # We want to select a set of tiles such that every C entry is covered exactly once (sum of contributions is 1).
    # This is an Exact Cover problem (or Integer Linear Programming).
    
    # Given the small size, we can use a simple backtracking or even brute force if the number of choices is small.
    # But let's try to be smarter.
    
    # Let's define the tiles available:
    # Tiles are defined by (A_rows, A_cols, B_rows, B_cols) -> C_rows, C_cols.
    # For a tile to be valid, A_rows and B_cols must match (inner dimension).
    
    # Let's enumerate all possible "atomic" tiles:
    # 1. 1x1 tiles: A[i,j] * B[j,k] -> C[i,k]. Rank 1.
    # 2. 2x2 tiles: A[i:i+2, j:j+2] * B[j:j+2, k:k+2] -> C[i:i+2, k:k+2].
    #    Can be done with Strassen (Rank 7) or Direct (Rank 4).
    # 3. 2x1 tiles: A[i:i+2, j] * B[j, k:k+2] -> C[i:i+2, k:k+2]. Rank 2 (Direct).
    # 4. 1x2 tiles: A[i, j:j+2] * B[j:j+2, k] -> C[i, k:k+2]. Rank 2 (Direct).
    # 5. 2x4 tiles: A[i:i+2, j:j+2] * B[j:j+2, k:k+4] -> C[i:i+2, k:k+4].
    #    Can be split into two 2x2s (14 or 8) or other ways?
    
    # This is getting complicated to enumerate all tilings.
    
    # Alternative Strategy:
    # Use the seed's 68-rank construction as a baseline.
    # Then, try to "improve" it by replacing groups of terms with more efficient ones.
    # Or, search for a different tiling.
    
    # Let's try a specific different tiling:
    # Instead of splitting B into 4+4, what if we split differently?
    # Or use 3x3 blocks?
    # A is 3x3.
    # If we use a 3x3 * 3x3 block, Strassen for 3x3 is Rank 23 (known best for 3x3 is 23? No, 23 is for 3x3? 
    # Actually, rank of 3x3 matrix multiplication is 23? 
    # Strassen 2x2 is 7.
    # 3x3 can be done with 23 mults (Winograd?).
    # If we use 3x3 * 3x4, we can split B into 3x2 + 3x2.
    # 3x3 * 3x2.
    
    # Let's try to find a decomposition for 3x3 * 3x2.
    # If we can do 3x3 * 3x2 in rank R, then 3x3 * 3x8 = 4 * R.
    # Schoolbook 3x3*3x2 is 18.
    # Can we do better?
    # 3x3 * 3x2.
    # A (3x3) * B (3x2).
    # C (3x2).
    
    # Let's try to construct a low rank for 3x3 * 3x2.
    # If we use Strassen for the 2x2 parts...
    # Split A into 2x2 (top-left) + rest.
    # This is similar to the seed.
    
    # Let's try a different approach: 
    # **Greedy Set Cover with Local Search**
    # 1. Generate a pool of candidate "blocks" (rank-1 or small-rank decompositions of sub-rectangles).
    # 2. Use a greedy algorithm to select blocks that cover the most uncovered C entries with the least rank cost.
    # 3. Verify the result.
    
    # Candidate Blocks:
    # - All 1x1 terms (72 of them).
    # - All 2x2 Strassen blocks (possible positions).
    # - All 2x2 Direct blocks (possible positions).
    # - All 2x1, 1x2 blocks.
    
    # Let's generate the 2x2 Strassen blocks.
    # A is 3x3, so possible 2x2 A sub-matrices: (0,0), (0,1), (1,0), (1,1).
    # B is 3x8, so possible 2x2 B sub-matrices: rows (0,1) or (1,2), cols (0,1), (1,2), ..., (6,7).
    # Total 4 * 2 * 7 = 56 possible 2x2 Strassen blocks.
    
    # Each block covers a 2x2 region of C.
    # Cost: 7 (Strassen) or 4 (Direct).
    
    # We want to cover the 3x8 C matrix (24 entries).
    # This is a Set Cover problem.
    
    # Let's implement a simple greedy set cover.
    
    # 1. Define all possible tiles.
    tiles = []
    
    # 1x1 Tiles
    for i in range(N):
        for j in range(M):
            for k in range(P):
                # Tile: A[i,j] * B[j,k] -> C[i,k]
                # Rank 1
                # Coverage: {(i,k)}
                # Contribution: U[i*M+j]=1, V[j*P+k]=1, W[k*N+i]=1
                tiles.append({
                    'type': 'direct_1x1',
                    'i': i, 'j': j, 'k': k,
                    'rank': 1,
                    'cover': frozenset([(i, k)]),
                    'terms': [({i*M+j: 1}, {j*P+k: 1}, {k*N+i: 1})]
                })
                
    # 2x2 Strassen Tiles
    for i in range(N-1): # 0, 1
        for j in range(M-1): # 0, 1
            for k in range(P-1): # 0..7
                # A[i:i+2, j:j+2] * B[j:j+2, k:k+2]
                # Note: B rows must match A cols.
                # A cols are j, j+1.
                # B rows must be j, j+1.
                # So B sub-matrix is B[j:j+2, k:k+2].
                
                # Check if B[j:j+2, k:k+2] is within bounds.
                # B is 3x8. j can be 0,1. j+1 <= 2. OK.
                # k can be 0..6. k+1 <= 7. OK.
                
                u_base = [i*M+j, i*M+j+1, (i+1)*M+j, (i+1)*M+j+1]
                v_base = [j*P+k, j*P+k+1, (j+1)*P+k, (j+1)*P+k+1]
                w_base = [k*N+i, k*N+i+1, (k+1)*N+i, (k+1)*N+i+1]
                
                terms = _build_strassen_block(u_base, v_base, w_base)
                
                # Cover: {(i,k), (i,k+1), (i+1,k), (i+1,k+1)}
                cover = frozenset([
                    (i, k), (i, k+1),
                    (i+1, k), (i+1, k+1)
                ])
                
                tiles.append({
                    'type': 'strassen_2x2',
                    'i': i, 'j': j, 'k': k,
                    'rank': 7,
                    'cover': cover,
                    'terms': terms
                })
                
                # Also add Direct 2x2 tile
                direct_terms = []
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                             # Wait, Direct 2x2 is 4 terms?
                             # A[i:i+2, j:j+2] * B[j:j+2, k:k+2]
                             # C[i:i+2, k:k+2]
                             # C[di, dk] = sum_{m=0}^1 A[i+di, j+m] B[j+m, k+dk]
                             # This requires 4 multiplications per C entry? No.
                             # Total 4 multiplications for the whole 2x2? No.
                             # 2x2 * 2x2 requires 8 multiplications if direct?
                             # No, 2x2 * 2x2 direct is 4*2 = 8?
                             # C00 = A00B00 + A01B10
                             # C01 = A00B01 + A01B11
                             # C10 = A10B00 + A11B10
                             # C11 = A10B01 + A11B11
                             # Total 8 multiplications.
                             # So Direct 2x2 rank is 8.
                             # Strassen is 7.
                             # So Strassen is always better for 2x2.
                             # I will skip Direct 2x2.
                             pass
                             
    # 2x1 Tiles (A 2x1 * B 1x2)
    # A[i:i+2, j] * B[j, k:k+2]
    for i in range(N-1):
        for j in range(M):
            for k in range(P-1):
                # A[i:i+2, j] is 2x1.
                # B[j, k:k+2] is 1x2.
                # C[i:i+2, k:k+2] is 2x2.
                
                u_base = [i*M+j, (i+1)*M+j]
                v_base = [j*P+k, j*P+k+1]
                w_base = [k*N+i, k*N+i+1, (k+1)*N+i, (k+1)*N+i+1]
                
                # Terms:
                # C[i, k] = A[i,j] B[j,k]
                # C[i, k+1] = A[i,j] B[j,k+1]
                # C[i+1, k] = A[i+1,j] B[j,k]
                # C[i+1, k+1] = A[i+1,j] B[j,k+1]
                
                terms = [
                    ({u_base[0]: 1}, {v_base[0]: 1}, {w_base[0]: 1}),
                    ({u_base[0]: 1}, {v_base[1]: 1}, {w_base[1]: 1}),
                    ({u_base[1]: 1}, {v_base[0]: 1}, {w_base[2]: 1}),
                    ({u_base[1]: 1}, {v_base[1]: 1}, {w_base[3]: 1})
                ]
                
                cover = frozenset([
                    (i, k), (i, k+1),
                    (i+1, k), (i+1, k+1)
                ])
                
                tiles.append({
                    'type': 'direct_2x1',
                    'rank': 4,
                    'cover': cover,
                    'terms': terms
                })
                
    # 1x2 Tiles (A 1x2 * B 2x1)
    # A[i, j:j+2] * B[j:j+2, k]
    for i in range(N):
        for j in range(M-1):
            for k in range(P):
                u_base = [i*M+j, i*M+j+1]
                v_base = [j*P+k, (j+1)*P+k]
                w_base = [k*N+i, (k+1)*N+i] # Wait, C is 1x2?
                # A is 1x2, B is 2x1. C is 1x1.
                # C[i, k] = A[i,j]B[j,k] + A[i,j+1]B[j+1,k]
                # This is 2 multiplications.
                
                terms = [
                    ({u_base[0]: 1}, {v_base[0]: 1}, {w_base[0]: 1}),
                    ({u_base[1]: 1}, {v_base[1]: 1}, {w_base[0]: 1})
                ]
                
                cover = frozenset([(i, k)])
                
                tiles.append({
                    'type': 'direct_1x2',
                    'rank': 2,
                    'cover': cover,
                    'terms': terms
                })
                
    # 2x4 Tiles? (A 2x2 * B 2x4)
    # Can be done with 2 Strassen blocks (14) or other ways.
    # Let's skip for now.
    
    # 3x3 Tiles?
    # A 3x3 * B 3x3.
    # Rank 23?
    # Let's not implement 3x3 Strassen yet.
    
    # Now we have a list of tiles.
    # We want to select a subset of tiles such that:
    # 1. Every cell (i, k) in C (3x8) is covered exactly once.
    # 2. The total rank is minimized.
    
    # This is a weighted Exact Cover problem.
    # We can use a backtracking search with pruning.
    
    target_cover = frozenset([(i, k) for i in range(N) for k in range(P)])
    
    best_rank = SCHOOLBOOK
    best_selection = None
    
    # Sort tiles by efficiency (rank / size)
    # Smaller rank per cell is better.
    # 1x1: 1/1 = 1
    # 2x2 Strassen: 7/4 = 1.75
    # 2x1 Direct: 4/4 = 1
    # 1x2 Direct: 2/1 = 2
    
    # Wait, 2x1 Direct covers 4 cells with rank 4. Efficiency 1.
    # 1x1 covers 1 cell with rank 1. Efficiency 1.
    # So they are equally efficient.
    # But 2x1 uses fewer terms? No, same rank.
    
    # Let's try to find a cover using mostly 2x1 and 1x2 tiles?
    # Or 2x2 Strassen?
    
    # Let's use a simple greedy first to get a baseline, then local search.
    
    # Greedy:
    remaining = set(target_cover)
    selected = []
    current_rank = 0
    
    while remaining:
        # Find tile with min rank/covered_cells_in_remaining
        best_tile = None
        best_ratio = float('inf')
        for tile in tiles:
            inter = len(tile['cover'] & remaining)
            if inter == 0:
                continue
            ratio = tile['rank'] / inter
            if ratio < best_ratio:
                best_ratio = ratio
                best_tile = tile
        
        if best_tile is None:
            break # Should not happen
            
        selected.append(best_tile)
        remaining -= best_tile['cover']
        current_rank += best_tile['rank']
        
    if len(remaining) == 0:
        if current_rank < best_rank:
            best_rank = current_rank
            best_selection = selected[:]
            
    # Now, try to improve with local search (swap tiles)
    # This is complex. Let's just try a few random restarts of the greedy with different tie-breaking.
    
    for restart in range(20):
        remaining = set(target_cover)
        selected = []
        current_rank = 0
        
        while remaining:
            candidates = []
            for tile in tiles:
                inter = len(tile['cover'] & remaining)
                if inter > 0:
                    ratio = tile['rank'] / inter
                    candidates.append((ratio, tile))
            
            if not candidates:
                break
                
            # Pick randomly from top 3 ratios
            candidates.sort(key=lambda x: x[0])
            top = candidates[:3]
            _, best_tile = top[restart % len(top)] # Simple deterministic variation
            
            selected.append(best_tile)
            remaining -= best_tile['cover']
            current_rank += best_tile['rank']
            
        if len(remaining) == 0:
            if current_rank < best_rank:
                best_rank = current_rank
                best_selection = selected[:]
                
    # If we found a good selection, build U, V, W
    if best_selection is not None:
        rank = best_rank
        U = [[0] * rank for _ in range(N * M)]
        V = [[0] * rank for _ in range(M * P)]
        W = [[0] * rank for _ in range(P * N)]
        
        col = 0
        for tile in best_selection:
            for term in tile['terms']:
                u_dict, v_dict, w_dict = term
                for u_idx, u_val in u_dict.items():
                    U[u_idx][col] += u_val
                for v_idx, v_val in v_dict.items():
                    V[v_idx][col] += v_val
                for w_idx, w_val in w_dict.items():
                    W[w_idx][col] += w_val
                col += 1
                
        if col == rank:
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
                
    # Fallback to schoolbook
    rank = SCHOOLBOOK
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    column = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][column] = 1
                V[j * P + k][column] = 1
                W[k * N + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
