"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 110.0  # Safety margin
    
    # Schoolbook decomposition as fallback (rank 36)
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        """Verify that the decomposition exactly represents matrix multiplication."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Check if sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def compute_cost(U, V, W, rank):
        """Compute the squared error cost for the scaled integer problem."""
        # U, V, W are lists of lists of integers (scaled by 2)
        # Target sum for each (i,j,k) is 8 (since (1/2)^3 * 8 = 1)
        cost = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    cost += (s - 8) ** 2
        return cost

    def local_search(rank, max_steps=50000):
        """
        Perform local search on the integer grid.
        Returns (U, V, W) if cost is 0, else None.
        """
        # Initialize with random small integers
        # We need values roughly in range [-10, 10]
        U = [[random.randint(-5, 5) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.randint(-5, 5) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.randint(-5, 5) for _ in range(rank)] for _ in range(p * n)]
        
        current_cost = compute_cost(U, V, W, rank)
        
        if current_cost == 0:
            return U, V, W

        step = 0
        while step < max_steps and time.time() < deadline - 5:
            step += 1
            # Pick a random tensor and entry
            tensor_idx = random.randint(0, 2)
            if tensor_idx == 0:
                rows, cols = n * m, rank
                T = U
            elif tensor_idx == 1:
                rows, cols = m * p, rank
                T = V
            else:
                rows, cols = p * n, rank
                T = W
            
            r = random.randint(0, rows - 1)
            c = random.randint(0, cols - 1)
            
            delta = random.choice([-1, 1])
            
            # Calculate delta cost
            # This requires checking all (i,j,k) affected by the change
            # If we change U[i0, j0, c0], it affects all (i0, j0, k) for k in 0..p-1
            # If we change V[j0, k0, c0], it affects all (i, j0, k0) for i in 0..n-1
            # If we change W[k0, i0, c0], it affects all (i0, j, k0) for j in 0..m-1
            
            old_cost_part = 0
            new_cost_part = 0
            
            if tensor_idx == 0:
                i0 = r // m
                j0 = r % m
                old_val = U[r][c]
                new_val = old_val + delta
                for k in range(p):
                    old_s = sum(U[i0*m + j0][cc] * V[j0*p + k][cc] * W[k*n + i0][cc] for cc in range(rank))
                    # Wait, this is O(rank) per k, so O(p*rank). 
                    # We can optimize by precomputing partial sums?
                    # For now, p=4, rank<=36, so 4*36=144 ops. Acceptable.
                    # Actually, we can just compute the change in sum for that specific (i,j,k)
                    # S_{i0,j0,k} changes by delta * V[j0*p+k][c] * W[k*n+i0][c]
                    pass
            
            # Simpler incremental update:
            # Let's just recompute the cost of the affected terms.
            
            affected_ijk = []
            if tensor_idx == 0:
                i0 = r // m
                j0 = r % m
                for k in range(p):
                    affected_ijk.append((i0, j0, k))
            elif tensor_idx == 1:
                j0 = r // p
                k0 = r % p
                for i in range(n):
                    affected_ijk.append((i, j0, k0))
            else:
                k0 = r // n
                i0 = r % n
                for j in range(m):
                    affected_ijk.append((i0, j, k0))
            
            cost_diff = 0
            for (i, j, k) in affected_ijk:
                old_s = 0
                new_s = 0
                for cc in range(rank):
                    u_val = U[i*m + j][cc]
                    v_val = V[j*p + k][cc]
                    w_val = W[k*n + i][cc]
                    
                    if tensor_idx == 0 and i == r // m and j == r % m and cc == c:
                        u_val_new = u_val + delta
                    else:
                        u_val_new = u_val
                        
                    if tensor_idx == 1 and j == r // p and k == r % p and cc == c:
                        v_val_new = v_val + delta
                    else:
                        v_val_new = v_val
                        
                    if tensor_idx == 2 and k == r // n and i == r % n and cc == c:
                        w_val_new = w_val + delta
                    else:
                        w_val_new = w_val
                        
                    old_s += u_val * v_val * w_val
                    new_s += u_val_new * v_val_new * w_val_new
                
                cost_diff += (new_s - 8) ** 2 - (old_s - 8) ** 2
            
            if current_cost + cost_diff < current_cost:
                # Apply move
                T[r][c] += delta
                current_cost += cost_diff
                if current_cost == 0:
                    return U, V, W
            else:
                # Occasionally accept worse moves (Simulated Annealing lite)
                # Or just reject. Let's try simple hill climbing first.
                # If stuck, we restart.
                pass
                
        return None

    # Try to find a lower rank decomposition
    # We search for ranks from 24 to 35
    for target_rank in range(24, 36):
        if time.time() > deadline:
            break
            
        # Multiple restarts
        # The number of restarts depends on remaining time
        # Estimate time per restart: 50000 steps * ~100 ops = 5e6 ops. ~0.05s?
        # Let's allow more steps if rank is lower.
        
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Increase max_steps for early ranks?
            # Let's try 20000 steps per restart to be safe.
            res = local_search(target_rank, max_steps=20000)
            
            if res:
                U, V, W = res
                # Convert back to half-integers
                U_half = [[val / 2.0 for val in row] for row in U]
                V_half = [[val / 2.0 for val in row] for row in V]
                W_half = [[val / 2.0 for val in row] for row in W]
                
                # Final verification
                if verify(U_half, V_half, W_half):
                    return {"U": U_half, "V": V_half, "W": W_half}
        
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
