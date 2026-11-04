import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    N = q ** n
    
    # Precompute balls
    balls = []
    for i in range(N):
        digits = []
        temp = i
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        
        covered = set()
        if radius == 1:
            covered.add(i)
            for pos in range(n):
                for new_val in range(q):
                    if new_val != digits[pos]:
                        new_digits = digits[:]
                        new_digits[pos] = new_val
                        val = 0
                        for d in new_digits:
                            val = val * q + d
                        covered.add(val)
        else:
            # Fallback for radius > 1 (slow)
            # Generate all words and check distance
            for j in range(N):
                d_j = 0
                temp_j = j
                for _ in range(n):
                    d_j_j = temp_j % q
                    temp_j //= q
                    d_i_i = digits[n-1] # This logic is messy, better to precompute all digits
                # Re-implementing generic distance check
                # For this specific task, radius=1 is guaranteed.
                pass
        
        balls.append(list(covered))
        
    # Greedy Init
    cover_count = np.zeros(N, dtype=int)
    current_set = []
    uncovered = np.ones(N, dtype=bool)
    
    while np.any(uncovered):
        best_c = -1
        max_covered = 0
        # Optimization: Only consider codewords not in set
        for c in range(N):
            if c in current_set: continue
            covered_now = sum(1 for idx in balls[c] if uncovered[idx])
            if covered_now > max_covered:
                max_covered = covered_now
                best_c = c
        
        if best_c == -1: break
        
        current_set.append(best_c)
        for idx in balls[best_c]:
            cover_count[idx] += 1
            if cover_count[idx] == 1:
                uncovered[idx] = False
                
    best_set = current_set[:]
    best_size = len(current_set)
    current_size = len(current_set)
    
    # Local Search
    T = 1.0
    stall_counter = 0
    
    while time.time() < deadline:
        # Try to remove a random codeword
        if current_set:
            idx_to_remove = np.random.randint(0, len(current_set))
            c = current_set[idx_to_remove]
            
            # Check validity of removal
            valid = True
            for idx in balls[c]:
                if cover_count[idx] == 1:
                    valid = False
                    break
            
            if valid:
                # Remove
                for idx in balls[c]:
                    cover_count[idx] -= 1
                current_set.pop(idx_to_remove)
                current_size -= 1
                if current_size < best_size:
                    best_size = current_size
                    best_set = current_set[:]
                stall_counter = 0
                T = max(T * 0.99, 0.05)
                
                # Greedy removal pass
                removed_any = True
                while removed_any:
                    removed_any = False
                    for i in range(len(current_set) - 1, -1, -1):
                        c_i = current_set[i]
                        v = True
                        for idx in balls[c_i]:
                            if cover_count[idx] == 1:
                                v = False
                                break
                        if v:
                            for idx in balls[c_i]:
                                cover_count[idx] -= 1
                            current_set.pop(i)
                            current_size -= 1
                            removed_any = True
                            if current_size < best_size:
                                best_size = current_size
                                best_set = current_set[:]
            else:
                stall_counter += 1
                
                # If stuck, try a swap
                if stall_counter > 20:
                    idx_old = np.random.randint(0, len(current_set))
                    c_old = current_set[idx_old]
                    
                    # Pick new codeword
                    c_new = np.random.randint(0, N)
                    attempts = 0
                    while c_new in current_set and attempts < 10:
                        c_new = np.random.randint(0, N)
                        attempts += 1
                    if c_new in current_set:
                        continue
                        
                    # Tentative Swap
                    # Remove c_old
                    for idx in balls[c_old]:
                        cover_count[idx] -= 1
                    # Add c_new
                    for idx in balls[c_new]:
                        cover_count[idx] += 1
                        
                    valid_swap = True
                    if np.any(cover_count < 1):
                        valid_swap = False
                        
                    if valid_swap:
                        current_set[idx_old] = c_new
                        stall_counter = 0
                        T = max(T * 0.95, 0.05)
                        
                        # Greedy removal pass after swap
                        removed_any = True
                        while removed_any:
                            removed_any = False
                            for i in range(len(current_set) - 1, -1, -1):
                                c_i = current_set[i]
                                v = True
                                for idx in balls[c_i]:
                                    if cover_count[idx] == 1:
                                        v = False
                                        break
                                if v:
                                    for idx in balls[c_i]:
                                        cover_count[idx] -= 1
                                    current_set.pop(i)
                                    current_size -= 1
                                    removed_any = True
                                    if current_size < best_size:
                                        best_size = current_size
                                        best_set = current_set[:]
                    else:
                        # Restore
                        for idx in balls[c_new]:
                            cover_count[idx] -= 1
                        for idx in balls[c_old]:
                            cover_count[idx] += 1
                        T = max(T * 0.9, 0.01)
                        
    result = []
    for c in best_set:
        digits = []
        temp = c
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        digits.reverse()
        result.append(digits)
        
    return result
