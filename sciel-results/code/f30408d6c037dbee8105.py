import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        M = max(U)
        if M == 0:
            return 1.0
        
        # Determine FFT size
        # We need to represent sums up to 2M and diffs up to M.
        # Linear convolution of two vectors of length M+1 requires size 2M+1.
        n_fft = 1
        while n_fft < 2 * M + 1:
            n_fft *= 2
        
        v = np.zeros(n_fft, dtype=np.float64)
        for u in U:
            v[u] = 1.0
            
        fft_v = np.fft.fft(v)
        
        # Sumset count
        fft_sum = fft_v * fft_v
        sum_vec = np.fft.ifft(fft_sum).real
        sum_vec = np.round(sum_vec).astype(int)
        count_sum = np.count_nonzero(sum_vec)
        
        # Diffset count
        # Autocorrelation
        fft_diff = fft_v * np.conj(fft_v)
        diff_vec = np.fft.ifft(fft_diff).real
        diff_vec = np.round(diff_vec).astype(int)
        count_diff = np.count_nonzero(diff_vec)
        
        if count_sum == 0:
            return 0.0
        ratio = count_diff / count_sum
        if ratio <= 0:
            return 0.0
            
        try:
            score = 1.0 + np.log(ratio) / np.log(2 * M + 1)
        except:
            score = 1.0
        return score

    # Initial candidates
    candidates = []
    # Small sets
    for k in range(3, 10):
        candidates.append(list(range(k)))
    # Powers of 2 (small ones)
    candidates.append([0, 1, 2, 4, 8])
    candidates.append([0, 1, 3, 7, 15]) # Mersenne?
    
    best_score = -1.0
    best_U = None
    
    for U in candidates:
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    if best_U is None:
        return [0]

    # Simulated Annealing
    current_U = best_U[:]
    current_score = best_score
    
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    # We will try to improve the best_U found
    # We need to handle the set size and max value dynamically
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        for _ in range(100): # Inner steps
            if time.time() > deadline:
                break
                
            # Generate move
            move_type = random.choice(['add', 'remove', 'replace'])
            
            if move_type == 'add':
                if len(current_U) > 50: # Limit size
                    continue
                max_val = max(current_U)
                # Bias towards small values
                new_val = random.randint(0, max_val * 2 + 10)
                if new_val not in current_U:
                    new_U = current_U + [new_val]
                    new_U.sort()
                    new_score = compute_score(new_U)
                    if new_score > current_score:
                        current_U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_U = current_U[:]
                    else:
                        delta = new_score - current_score
                        if delta > 0 or random.random() < np.exp(delta / T):
                            current_U = new_U
                            current_score = new_score

            elif move_type == 'remove':
                if len(current_U) <= 2:
                    continue
                # Don't remove 0
                idx = random.randint(1, len(current_U) - 1)
                new_U = current_U[:idx] + current_U[idx+1:]
                new_score = compute_score(new_U)
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_U = current_U[:]
                else:
                    delta = new_score - current_score
                    if delta > 0 or random.random() < np.exp(delta / T):
                        current_U = new_U
                        current_score = new_score

            elif move_type == 'replace':
                if len(current_U) <= 1:
                    continue
                idx = random.randint(1, len(current_U) - 1)
                max_val = max(current_U)
                new_val = random.randint(0, max_val * 2 + 10)
                if new_val in current_U:
                    continue
                new_U = current_U[:]
                new_U[idx] = new_val
                new_U.sort()
                new_score = compute_score(new_U)
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_U = current_U[:]
                else:
                    delta = new_score - current_score
                    if delta > 0 or random.random() < np.exp(delta / T):
                        current_U = new_U
                        current_score = new_score

        T *= cooling
        
        # If we are stuck, restart with a different random seed or structure
        if random.random() < 0.05:
             # Random restart
             k = random.randint(5, 20)
             M = random.randint(10, 100)
             U_rand = [0] + random.sample(range(1, M+1), min(k-1, M))
             U_rand.sort()
             s = compute_score(U_rand)
             if s > current_score:
                 current_U = U_rand
                 current_score = s
                 if s > best_score:
                     best_score = s
                     best_U = current_U[:]
             # Reset T
             T = 0.5

    return best_U

# EVOLVE-BLOCK-START
def construct():
    return construct_impl()

def construct_impl():
    import numpy as np
    import time
    import random

    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    def compute_score(U):
        if not U or 0 not in U:
            return 0.0
        M = max(U)
        if M == 0:
            return 1.0
        
        n_fft = 1
        while n_fft < 2 * M + 1:
            n_fft *= 2
        
        v = np.zeros(n_fft, dtype=np.float64)
        for u in U:
            v[u] = 1.0
            
        fft_v = np.fft.fft(v)
        
        fft_sum = fft_v * fft_v
        sum_vec = np.fft.ifft(fft_sum).real
        sum_vec = np.round(sum_vec).astype(int)
        count_sum = np.count_nonzero(sum_vec)
        
        fft_diff = fft_v * np.conj(fft_v)
        diff_vec = np.fft.ifft(fft_diff).real
        diff_vec = np.round(diff_vec).astype(int)
        count_diff = np.count_nonzero(diff_vec)
        
        if count_sum == 0:
            return 0.0
        ratio = count_diff / count_sum
        if ratio <= 0:
            return 0.0
            
        try:
            score = 1.0 + np.log(ratio) / np.log(2 * M + 1)
        except:
            score = 1.0
        return score

    candidates = []
    for k in range(3, 15):
        candidates.append(list(range(k)))
    candidates.append([0, 1, 2, 4, 8, 16])
    candidates.append([0, 1, 3, 7, 15, 31])
    candidates.append([0, 1, 4, 10, 20, 40])
    
    best_score = -1.0
    best_U = None
    
    for U in candidates:
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]

    if best_U is None:
        return [0]

    current_U = best_U[:]
    current_score = best_score
    
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    while time.time() < deadline:
        if T < min_T:
            break
            
        for _ in range(200):
            if time.time() > deadline:
                break
                
            move_type = random.choice(['add', 'remove', 'replace'])
            
            if move_type == 'add':
                if len(current_U) > 30:
                    continue
                max_val = max(current_U)
                new_val = random.randint(0, max_val * 2 + 10)
                if new_val not in current_U:
                    new_U = current_U + [new_val]
                    new_U.sort()
                    new_score = compute_score(new_U)
                    if new_score > current_score:
                        current_U = new_U
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_U = current_U[:]
                    else:
                        delta = new_score - current_score
                        if delta > 0 or random.random() < np.exp(delta / T):
                            current_U = new_U
                            current_score = new_score

            elif move_type == 'remove':
                if len(current_U) <= 2:
                    continue
                idx = random.randint(1, len(current_U) - 1)
                new_U = current_U[:idx] + current_U[idx+1:]
                new_score = compute_score(new_U)
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_U = current_U[:]
                else:
                    delta = new_score - current_score
                    if delta > 0 or random.random() < np.exp(delta / T):
                        current_U = new_U
                        current_score = new_score

            elif move_type == 'replace':
                if len(current_U) <= 1:
                    continue
                idx = random.randint(1, len(current_U) - 1)
                max_val = max(current_U)
                new_val = random.randint(0, max_val * 2 + 10)
                if new_val in current_U:
                    continue
                new_U = current_U[:]
                new_U[idx] = new_val
                new_U.sort()
                new_score = compute_score(new_U)
                if new_score > current_score:
                    current_U = new_U
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_U = current_U[:]
                else:
                    delta = new_score - current_score
                    if delta > 0 or random.random() < np.exp(delta / T):
                        current_U = new_U
                        current_score = new_score

        T *= cooling
        
        if random.random() < 0.1:
             k = random.randint(5, 25)
             M = random.randint(10, 200)
             U_rand = [0] + random.sample(range(1, M+1), min(k-1, M))
             U_rand.sort()
             s = compute_score(U_rand)
             if s > current_score:
                 current_U = U_rand
                 current_score = s
                 if s > best_score:
                     best_score = s
                     best_U = current_U[:]
             T = 0.5

    return best_U
# EVOLVE-BLOCK-END
