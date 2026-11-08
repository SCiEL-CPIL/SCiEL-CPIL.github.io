import time
import numpy as np
from scipy.optimize import root

_DIMS = (3, 4, 7)

def construct():
    _t0 = time.time()
    n, m, p = _DIMS
    
    # --- Part 1: Improved Packing Search ---
    
    # Generate all possible 2x2x2 blocks
    blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                blocks.append((i0, j0, k0))
                
    num_blocks = len(blocks)
    
    # Build conflict graph
    # adj[i][j] = True if blocks i and j are INCOMPATIBLE (conflict)
    adj = [[False] * num_blocks for _ in range(num_blocks)]
    
    def are_incompatible(b1, b2):
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        # Overlap in all dimensions
        if abs(i1 - i2) < 2 and abs(j1 - j2) < 2 and abs(k1 - k2) < 2:
            return True
        return False

    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            if are_incompatible(blocks[i], blocks[j]):
                adj[i][j] = True
                adj[j][i] = True

    # Build compatibility graph (complement of conflict graph)
    # compat[i][j] = True if blocks i and j are COMPATIBLE
    compat = [[not adj[i][j] for j in range(num_blocks)] for i in range(num_blocks)]
    # Note: A block is compatible with itself? No, we pick distinct blocks.
    # For Max Clique, we look for a set of nodes where all pairs are connected in the clique graph.
    # Here, clique graph = compat graph.
    
    # Max Clique in compat graph using Tomita's algorithm (Coloring bound)
    # This is significantly faster than simple backtracking for N=36.
    
    best_clique = []
    
    def expand(clique, candidates):
        nonlocal best_clique
        if len(candidates) == 0:
            if len(clique) > len(best_clique):
                best_clique = clique[:]
            return
        
        # Greedy Coloring of the subgraph induced by candidates
        # colors[v] = color class index
        # We want to bound the size of the clique we can find in candidates
        # If len(clique) + max_color <= len(best_clique), prune
        
        # Simple coloring:
        # Order candidates by degree in candidates (descending)
        degs = [sum(compat[candidates[i]][candidates[j]] for j in range(len(candidates)) if i != j) for i in range(len(candidates))]
        indices = sorted(range(len(candidates)), key=lambda x: -degs[x])
        ordered_candidates = [candidates[i] for i in indices]
        
        # Assign colors
        color_map = {}
        max_color = 0
        for v in ordered_candidates:
            # Find smallest color not used by neighbors
            used_colors = set()
            for u in ordered_candidates:
                if u != v and compat[v][u] and u in color_map:
                    used_colors.add(color_map[u])
            c = 1
            while c in used_colors:
                c += 1
            color_map[v] = c
            max_color = max(max_color, c)
            
            # Pruning
            if len(clique) + max_color <= len(best_clique):
                return

        # Process nodes in reverse color order (highest color first)
        # Group by color
        color_classes = {}
        for v in ordered_candidates:
            c = color_map[v]
            if c not in color_classes:
                color_classes[c] = []
            color_classes[c].append(v)
            
        for c in sorted(color_classes.keys(), reverse=True):
            # If bound is hit, break
            if len(clique) + c <= len(best_clique):
                break
            
            for v in color_classes[c]:
                if v not in candidates: # Should not happen, but safety
                    continue
                new_clique = clique + [v]
                # New candidates must be compatible with v and in current candidates
                new_candidates = [u for u in candidates if u != v and compat[v][u]]
                
                # Prune if bound is not met
                # We can re-evaluate bound, but simple check:
                # The max clique in new_candidates is at most len(new_candidates)
                # But we used coloring bound in the parent, so we just recurse.
                expand(new_clique, new_candidates)
                
                candidates.remove(v)
                
    expand([], list(range(num_blocks)))
    
    best_block_indices = best_clique
    best_blocks = [blocks[i] for i in best_block_indices]
    
    # Construct decomposition from packing
    num_selected = len(best_blocks)
    
    covered_entries = set()
    for b in best_blocks:
        i0, j0, k0 = b
        for i in range(i0, i0 + 2):
            for j in range(j0, j0 + 2):
                for k in range(k0, k0 + 2):
                    covered_entries.add((i, j, k))
    
    uncovered_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered_entries:
                    uncovered_entries.append((i, j, k))
    
    rank = 7 * num_selected + len(uncovered_entries)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0 * m + j0][col] = 1; U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1; V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1; W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        # M2
        U[(i0+1) * m + j0][col] = 1; U[(i0+1) * m + (j0+1)][col] = 1
        V[j0 * p + k0][col] = 1
        W[k0 * n + (i0+1)][col] = 1; W[(k0+1) * n + (i0+1)][col] = -1
        col += 1
        # M3
        U[i0 * m + j0][col] = 1
        V[j0 * p + (k0+1)][col] = 1; V[(j0+1) * p + (k0+1)][col] = -1
        W[(k0+1) * n + i0][col] = 1; W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        # M4
        U[(i0+1) * m + (j0+1)][col] = 1
        V[(j0+1) * p + k0][col] = 1; V[j0 * p + k0][col] = -1
        W[k0 * n + i0][col] = 1; W[k0 * n + (i0+1)][col] = 1
        col += 1
        # M5
        U[i0 * m + j0][col] = 1; U[i0 * m + (j0+1)][col] = 1
        V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = -1; W[(k0+1) * n + i0][col] = 1
        col += 1
        # M6
        U[(i0+1) * m + j0][col] = 1; U[i0 * m + j0][col] = -1
        V[j0 * p + k0][col] = 1; V[j0 * p + (k0+1)][col] = 1
        W[(k0+1) * n + (i0+1)][col] = 1
        col += 1
        # M7
        U[i0 * m + (j0+1)][col] = 1; U[(i0+1) * m + (j0+1)][col] = -1
        V[(j0+1) * p + k0][col] = 1; V[(j0+1) * p + (k0+1)][col] = 1
        W[k0 * n + i0][col] = 1
        col += 1

    for b in best_blocks:
        add_strassen(*b)
        
    # Add schoolbook terms for uncovered entries
    for (i, j, k) in uncovered_entries:
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
        
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    # --- Part 2: Verification of Packing Solution ---
    if time.time() < _t0 + 110.0:
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx])
                    if val != 1:
                        raise ValueError(f"Packing decomposition failed at ({i},{j},{k}): {val}")
    
    # --- Part 3: Search for Better Rank (if time permits) ---
    # If we have time, try to find a rank < current rank using random search
    # This is a heuristic and may fail, but if it succeeds, it's a big win.
    
    if time.time() < _t0 + 100.0:
        best_rank_found = rank
        best_U, best_V, best_W = U, V, W
        
        # Try ranks from lower bound up to best_rank_found - 1
        # Lower bound: ceil(84/7) = 12. 
        # Realistic lower bound for 3x4x7 might be higher. 
        # Let's try a few specific ranks: 70, 60, 50, 40, 30, 20
        # If we find one, we return it.
        
        target_ranks = [r for r in range(20, best_rank_found) if r % 10 == 0]
        # Add best_rank_found - 1, -2, -3 for fine tuning
        target_ranks.extend([best_rank_found - 1, best_rank_found - 2, best_rank_found - 3])
        target_ranks = sorted(set(target_ranks))
        
        for R in target_ranks:
            if time.time() > _t0 + 115.0:
                break
                
            # Try a few random restarts for this R
            num_restarts = 5
            for _ in range(num_restarts):
                if time.time() > _t0 + 118.0:
                    break
                    
                # Random initialization
                # Scale: U, V ~ 1, W ~ 1
                U_init = np.random.randn(n * m, R)
                V_init = np.random.randn(m * p, R)
                W_init = np.random.randn(p * n, R)
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                def residual(x):
                    U = x[:n*m*R].reshape(n*m, R)
                    V = x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                    W = x[n*m*R+m*p*R:].reshape(p*n, R)
                    res = np.zeros(n*m*p)
                    idx = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                # T_ijk = 1
                                val = np.dot(U[i*m+j, :], V[j*p+k, :])
                                # Wait, W is indexed by (k, i)
                                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                                res[idx] = val - 1.0
                                idx += 1
                    return res

                try:
                    sol = root(residual, x0, method='hybr', tol=1e-8)
                    if sol.success and np.max(np.abs(sol.fun)) < 1e-6:
                        # Found a solution!
                        # Verify and extract
                        U_sol = sol.x[:n*m*R].reshape(n*m, R)
                        V_sol = sol.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                        W_sol = sol.x[n*m*R+m*p*R:].reshape(p*n, R)
                        
                        # Round to nearest 0.5 to check for exactness
                        U_int = np.round(U_sol * 2) / 2
                        V_int = np.round(V_sol * 2) / 2
                        W_int = np.round(W_sol * 2) / 2
                        
                        # Verify integer solution
                        valid = True
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = np.sum(U_int[i*m+j] * V_int[j*p+k] * W_int[k*n+i])
                                    if abs(val - 1.0) > 1e-9:
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                            
                        if valid:
                            best_rank_found = R
                            best_U = U_int.tolist()
                            best_V = V_int.tolist()
                            best_W = W_int.tolist()
                            break # Found best for this R, try lower R
                except Exception:
                    pass
                    
        if best_rank_found < rank:
            U = best_U
            V = best_V
            W = best_W
            rank = best_rank_found

    return {"U": U, "V": V, "W": W}
