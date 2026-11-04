import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 3, 3, 5
    deadline = time.time() + 130.0  # Safety margin before 150s kill

    def verify_decomposition(U, V, W, n, m, p):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][idx] = 1
                    V[j * p + k][idx] = 1
                    W[k * n + i][idx] = 1
                    idx += 1
        return U, V, W, rank

    # Try to find a rank-44 decomposition using structured search
    # The key insight: we can try to merge pairs of schoolbook terms
    # that share the same U-index (same i,j pair) but different k-values.
    
    def try_merge_pair(i, j, k1, k2):
        """
        Try to replace schoolbook terms (i,j,k1) and (i,j,k2) with 2 new terms
        that produce the same contributions to C[i,k1] and C[i,k2] while
        contributing 0 to all other outputs.
        
        Returns (U_new, V_new, W_new) for the 2 new terms, or None if not possible.
        """
        # We need 2 new terms c=0, c=1 such that:
        # For output (i', j', k'):
        #   sum_c U[i',j',c] * V[j',k',c] * W[k',i',c] = delta(i',i)*delta(j',j)*delta(k',k1) + delta(i',i)*delta(j',j)*delta(k',k2)
        
        # This means:
        # - For (i,j,k1): U[i,j,0]*V[j,k1,0]*W[k1,i,0] + U[i,j,1]*V[j,k1,1]*W[k1,i,1] = 1
        # - For (i,j,k2): U[i,j,0]*V[j,k2,0]*W[k2,i,0] + U[i,j,1]*V[j,k2,1]*W[k2,i,1] = 1
        # - For all other (i',j',k'): sum = 0
        
        # Simple approach: set U[i,j,0]=1, U[i,j,1]=1, and all other U entries to 0.
        # Then we need:
        #   V[j,k1,0]*W[k1,i,0] + V[j,k1,1]*W[k1,i,1] = 1
        #   V[j,k2,0]*W[k2,i,0] + V[j,k2,1]*W[k2,i,1] = 1
        #   For k' not in {k1,k2}: V[j,k',0]*W[k',i,0] + V[j,k',1]*W[k',i,1] = 0
        #   For j' != j: V[j',k,0]*W[k,i,0] + V[j',k,1]*W[k,i,1] = 0 for all k
        #   For i' != i: W[k,i',0]*V[j,k,0] + W[k,i',1]*V[j,k,1] = 0 for all k
        
        # This is a system of equations. Let's try a specific construction:
        # Set W[k1,i,0]=1, W[k1,i,1]=0, W[k2,i,0]=0, W[k2,i,1]=1
        # Set W[k,i,0]=0 for k not in {k1,k2}
        # Set W[k,i',0]=0 for i' != i
        # Set W[k,i',1]=0 for i' != i
        # Then: V[j,k1,0]*1 + V[j,k1,1]*0 = 1 => V[j,k1,0] = 1
        #        V[j,k2,0]*0 + V[j,k2,1]*1 = 1 => V[j,k2,1] = 1
        #        For k' not in {k1,k2}: V[j,k',0]*0 + V[j,k',1]*0 = 0 ✓
        #        For j' != j: V[j',k,0]*0 + V[j',k,1]*0 = 0 ✓
        
        # But wait, this doesn't work because we also need:
        #   V[j,k1,1] and V[j,k2,0] can be anything, but they must not affect other outputs.
        #   Actually, with the W values as set, V[j,k1,1] and V[j,k2,0] don't appear in any equation
        #   because W[k1,i,1]=0 and W[k2,i,0]=0.
        
        # So we have:
        # U[i,j,0]=1, U[i,j,1]=1, all other U=0
        # V[j,k1,0]=1, V[j,k2,1]=1, all other V=0 (except V[j,k1,1] and V[j,k2,0] which are free)
        # W[k1,i,0]=1, W[k2,i,1]=1, all other W=0
        
        # Let's set V[j,k1,1]=0 and V[j,k2,0]=0 for simplicity.
        
        U_new = [[0, 0] for _ in range(9)]
        V_new = [[0, 0] for _ in range(15)]
        W_new = [[0, 0] for _ in range(15)]
        
        U_new[i * m + j][0] = 1
        U_new[i * m + j][1] = 1
        V_new[j * p + k1][0] = 1
        V_new[j * p + k2][1] = 1
        W_new[k1 * n + i][0] = 1
        W_new[k2 * n + i][1] = 1
        
        return U_new, V_new, W_new

    def build_with_merge(i, j, k1, k2):
        """Build a rank-44 decomposition by merging terms (i,j,k1) and (i,j,k2)."""
        U_new, V_new, W_new = try_merge_pair(i, j, k1, k2)
        
        U = [[0] * 44 for _ in range(9)]
        V = [[0] * 44 for _ in range(15)]
        W = [[0] * 44 for _ in range(15)]
        
        # Add schoolbook terms except (i,j,k1) and (i,j,k2)
        idx = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    if ii == i and jj == j and kk in (k1, k2):
                        continue
                    U[ii * m + jj][idx] = 1
                    V[jj * p + kk][idx] = 1
                    W[kk * n + ii][idx] = 1
                    idx += 1
        
        # Add the 2 new merged terms
        for c in range(2):
            for ii in range(n):
                for jj in range(m):
                    U[ii * m + jj][idx + c] = U_new[ii * m + jj][c]
            for jj in range(m):
                for kk in range(p):
                    V[jj * p + kk][idx + c] = V_new[jj * p + kk][c]
            for kk in range(p):
                for ii in range(n):
                    W[kk * n + ii][idx + c] = W_new[kk * n + ii][c]
        
        return U, V, W, 44

    # Try all possible merges
    best_result = None
    for i in range(n):
        for j in range(m):
            for k1 in range(p):
                for k2 in range(k1 + 1, p):
                    if time.time() > deadline - 20:
                        break
                    U, V, W, rank = build_with_merge(i, j, k1, k2)
                    if verify_decomposition(U, V, W, n, m, p):
                        best_result = (U, V, W, rank)
                        break
                if best_result:
                    break
            if best_result:
                break
        if best_result:
            break
    
    if best_result:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback to schoolbook
    U, V, W, rank = make_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
