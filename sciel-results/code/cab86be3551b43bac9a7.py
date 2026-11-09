import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s margin for 120s limit
    
    n = 17
    # 4 colors
    # Adjacency matrices for each color
    # A[c][i][j] = 1 if edge (i,j) has color c
    
    # Initialize
    # We will try to find a solution for n=17
    # If we find one, return it.
    # If we don't, we might try n=16, 15...
    
    best_n = 0
    best_colors = None
    
    for target_n in [17, 16, 15]:
        if time.time() - start_time > time_limit * 0.8:
            break
            
        # Try to find valid coloring for target_n
        success = False
        best_penalty = float('inf')
        best_A = None
        best_colors_list = None
        
        # Simulated Annealing
        A = [np.zeros((target_n, target_n), dtype=int) for _ in range(4)]
        # Initialize randomly
        colors = []
        for i in range(target_n):
            for j in range(i+1, target_n):
                c = random.randint(0, 3)
                A[c][i][j] = 1
                A[c][j][i] = 1
                colors.append(c)
        
        # Calculate initial penalty
        # Penalty = sum over colors of sum of A^5 entries
        # Note: sum(A^5) counts closed walks of length 5? No.
        # We want number of C6.
        # A C6 is a cycle of length 6.
        # The number of C6s is related to the trace of A^6.
        # Tr(A^6) = 2 * (number of C6s) * 6 + (other closed walks).
        # Actually, for a simple cycle of length k, Tr(A^k) = 2k * (number of such cycles) + ...
        # It's easier to just count paths of length 5 between u and v.
        # Total C6s = 1/12 * sum_{u!=v} (A^5)[u,v]?
        # Let's check.
        # A C6 has 6 vertices.
        # Each vertex pair in the cycle has a unique path of length 5 between them along the cycle?
        # No, in a C6, for any two vertices, there are two paths.
        # If they are opposite (dist 3), path length 3 and 3.
        # If dist 1, path length 1 and 5.
        # If dist 2, path length 2 and 4.
        # So for each edge (u,v) in the C6, there is exactly one path of length 5 between u and v (the rest of the cycle).
        # So sum_{u<v} (A^5)[u,v] counts each C6 exactly 6 times (once for each edge).
        # So Number of C6s = (1/6) * sum_{u<v} (A^5)[u,v].
        # We can minimize sum_{u<v} (A^5)[u,v].
        
        def calc_penalty(A):
            pen = 0
            for c in range(4):
                # A^5
                # We only need the upper triangle sum
                A5 = np.linalg.matrix_power(A[c], 5)
                # Sum of upper triangle
                # Note: A5 is not symmetric if graph is directed? No, graph is undirected, so A is symmetric, A5 is symmetric.
                # But (A^5)[u,v] counts walks.
                # We want simple paths?
                # The formula "Number of C6s = 1/6 sum (A^5)[u,v]" holds if A^5 counts simple paths?
                # No, A^5 counts ALL walks of length 5.
                # This includes walks that repeat vertices.
                # So this penalty is an overestimate.
                # However, minimizing the number of walks of length 5 is a good proxy for minimizing cycles.
                # And if we reach 0, it means no walks of length 5, so no C6.
                # So this is a valid objective.
                pen += np.sum(A5) # Sum all entries
            return pen
        
        # This penalty is expensive to calculate from scratch every time.
        # But we can update it incrementally?
        # Or just recalculate for the two changed matrices.
        
        # Let's use the incremental update idea.
        # We store the current penalty for each color.
        pen = [0.0]*4
        for c in range(4):
            A5 = np.linalg.matrix_power(A[c], 5)
            pen[c] = np.sum(A5)
        total_pen = sum(pen)
        
        T = 1.0
        cooling = 0.999
        no_improve = 0
        
        while time.time() - start_time < time_limit * 0.8:
            # Pick random edge
            i = random.randint(0, target_n-1)
            j = random.randint(0, target_n-1)
            if i == j: continue
            if i > j: i, j = j, i
            
            c_old = None
            for c in range(4):
                if A[c][i][j] == 1:
                    c_old = c
                    break
            
            # Pick new color
            c_new = random.randint(0, 3)
            if c_new == c_old: continue
            
            # Calculate new penalty for c_old and c_new
            # Temporarily move
            A[c_old][i][j] = 0
            A[c_old][j][i] = 0
            A[c_new][i][j] = 1
            A[c_new][j][i] = 1
            
            # Recalculate pen for c_old and c_new
            # This is the bottleneck.
            # Matrix power 5 of 17x17.
            A5_old = np.linalg.matrix_power(A[c_old], 5)
            A5_new = np.linalg.matrix_power(A[c_new], 5)
            
            new_pen_old = np.sum(A5_old)
            new_pen_new = np.sum(A5_new)
            
            delta_pen = (new_pen_old + new_pen_new) - (pen[c_old] + pen[c_new])
            
            if delta_pen < 0 or random.random() < np.exp(-delta_pen / T):
                # Accept
                pen[c_old] = new_pen_old
                pen[c_new] = new_pen_new
                total_pen += delta_pen
                if total_pen < best_penalty:
                    best_penalty = total_pen
                    # Save state
                    best_A = [m.copy() for m in A]
                    # Reconstruct colors list
                    best_colors_list = [0]*(target_n*(target_n-1)//2)
                    idx = 0
                    for u in range(target_n):
                        for v in range(u+1, target_n):
                            for c in range(4):
                                if best_A[c][u][v] == 1:
                                    best_colors_list[idx] = c
                                    break
                            idx += 1
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                # Reject
                A[c_old][i][j] = 1
                A[c_old][j][i] = 1
                A[c_new][i][j] = 0
                A[c_new][j][i] = 0
            
            T *= cooling
            if no_improve > 1000:
                T *= 0.5
                no_improve = 0
                
            if total_pen == 0:
                success = True
                break

        if success:
            best_n = target_n
            best_colors = best_colors_list
            break
        elif best_penalty == 0:
            best_n = target_n
            best_colors = best_colors_list
            break
        else:
            # Keep the best found so far for this n?
            # Or just move to next n?
            # If we found a good one for 17 but not 0, it's invalid.
            # So we must find a valid one (pen=0).
            pass

    if best_n == 0:
        # Fallback to seed n=9
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
    else:
        return {'n': best_n, 'colors': best_colors}
