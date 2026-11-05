import numpy as np
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Fallback: The seed program's n=32 coloring
    def get_seed_coloring():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    best_result = get_seed_coloring()
    
    # Try to find larger n
    # We know n=50 is possible.
    # We will attempt n from 50 down to 33.
    
    def count_mono_triangles(M):
        # M is a list of 4 numpy arrays (n x n) boolean
        total = 0
        for k in range(4):
            # Trace of M_k^3
            # M_k @ M_k @ M_k
            # Using np.trace
            M_k = M[k]
            # Efficient way: sum of diagonal of M_k @ M_k @ M_k
            # Or simply: for each i, sum_j (M_k[i,j] * sum_l M_k[j,l] * M_k[l,i])
            # np.dot is optimized
            M_k2 = M_k @ M_k
            total += np.trace(M_k2 @ M_k)
        return total // 6  # Each triangle counted 6 times

    def sa_search(n, time_limit):
        if time.time() > time_limit:
            return None
            
        # Initialize
        # Random initialization is often sufficient, but let's try a structured one first?
        # Random is safer to avoid local minima traps if structured is bad.
        # Let's use random.
        
        # M[k] is an n x n boolean matrix
        M = [np.zeros((n, n), dtype=bool) for _ in range(4)]
        
        # Store colors in a flat array for easy access if needed, 
        # but M is the primary state for scoring.
        # We need to be able to flip an edge.
        
        current_score = count_mono_triangles(M)
        if current_score == 0:
             # Extract colors
             colors = []
             for i in range(n):
                 for j in range(i+1, n):
                     # Find which M[k] has True
                     # This is O(4) per edge
                     for k in range(4):
                         if M[k][i,j]:
                             colors.append(k)
                             break
             return {'n': n, 'colors': colors}
             
        best_score = current_score
        best_M = [m.copy() for m in M]
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.99
        
        steps = 0
        last_improve = time.time()
        
        while time.time() < time_limit:
            # Pick random edge i < j
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
                
            # Find current color
            old_k = -1
            for k in range(4):
                if M[k][i, j]:
                    old_k = k
                    break
            if old_k == -1:
                continue # Should not happen if initialized properly, but M is initially all 0?
                # Wait, initialization:
                # I initialized M to zeros. I need to set initial colors!
                
            # Re-initialization logic needed inside or before loop
            # Let's fix initialization:
            # Assign random colors to all edges
            # M is initially all False.
            # We need to set True for one color per edge.
            
            # Break and redo init properly.
            pass
            
        return None

    # Proper SA Implementation
    def run_sa(n, time_limit):
        if time.time() > time_limit:
            return None
            
        # Initialize M
        M = [np.zeros((n, n), dtype=bool) for _ in range(4)]
        
        # Assign random colors
        for i in range(n):
            for j in range(i + 1, n):
                k = random.randint(0, 3)
                M[k][i, j] = True
                M[k][j, i] = True
        
        current_score = count_mono_triangles(M)
        best_score = current_score
        best_M = [m.copy() for m in M]
        
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.995
        steps_since_improve = 0
        max_steps_no_improve = 5000
        
        while time.time() < time_limit:
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j: continue
            if i > j: i, j = j, i
            
            # Find old color
            old_k = -1
            for k in range(4):
                if M[k][i, j]:
                    old_k = k
                    break
            if old_k == -1: continue
            
            # Pick new color
            new_k = random.randint(0, 3)
            if new_k == old_k: continue
            
            # Update M
            M[old_k][i, j] = False
            M[old_k][j, i] = False
            M[new_k][i, j] = True
            M[new_k][j, i] = True
            
            # Calculate new score
            new_score = count_mono_triangles(M)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_score = new_score
                steps_since_improve += 1
                if current_score < best_score:
                    best_score = current_score
                    best_M = [m.copy() for m in M]
                    steps_since_improve = 0
                    if best_score == 0:
                        # Found valid coloring
                        colors = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                for k in range(4):
                                    if M[k][i, j]:
                                        colors.append(k)
                                        break
                        return {'n': n, 'colors': colors}
            else:
                # Revert
                M[old_k][i, j] = True
                M[old_k][j, i] = True
                M[new_k][i, j] = False
                M[new_k][j, i] = False
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # Optional: Reset to best and reheat?
                # For now, just keep going.
                
            if steps_since_improve > max_steps_no_improve:
                # Restart from best
                M = [m.copy() for m in best_M]
                current_score = best_score
                T = 1.0
                steps_since_improve = 0

        # If loop finishes, check if best is valid
        if best_score == 0:
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(4):
                        if best_M[k][i, j]:
                            colors.append(k)
                            break
            return {'n': n, 'colors': colors}
            
        return None

    # Try n values
    # Allocate time based on n
    # n=50 is most likely, give it more time.
    targets = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]
    
    for n in targets:
        if time.time() > deadline:
            break
            
        # Allocate 5 seconds per n, or less if running low
        time_left = deadline - time.time()
        if time_left < 2:
            break
            
        alloc = min(5.0, time_left - 1.0)
        
        result = run_sa(n, time.time() + alloc)
        
        if result:
            return result

    return best_result
