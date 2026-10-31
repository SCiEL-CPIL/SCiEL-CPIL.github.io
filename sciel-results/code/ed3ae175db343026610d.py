import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = 110
    
    def get_max_abs_autoconv(seq):
        # Direct computation is O(N^2) which is 160k, acceptable for init
        # But FFT is faster for single calc
        f = np.fft.fft(seq, n=2*N)
        power = f * np.conj(f)
        c = np.fft.ifft(power).real
        return np.max(np.abs(c[:N]))

    def local_search(seq, max_iterations=1000):
        # Initialize C
        # Using FFT for initial C
        f = np.fft.fft(seq, n=2*N)
        power = f * np.conj(f)
        C = np.fft.ifft(power).real[:N]
        
        current_max = np.max(np.abs(C))
        best_local_max = current_max
        best_local_seq = seq.copy()
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # We want to find the flip that minimizes the new max
            # We can check all flips
            
            best_flip_idx = -1
            best_new_max = current_max
            
            for i in range(N):
                # Calculate Delta C
                # Delta C(k) = -2 * seq[i] * (seq[i-k] + seq[i+k])
                # We need to handle boundaries for linear convolution
                # k ranges from 0 to N-1
                # i-k and i+k must be within [0, N-1]
                
                # Vectorized calculation of V_i(k) for all k
                # k = 0 .. N-1
                # i-k is valid if i-k >= 0
                # i+k is valid if i+k < N
                
                # Create array for i-k and i+k
                ks = np.arange(N)
                
                # Valid indices for i-k
                idx1 = i - ks
                valid1 = idx1 >= 0
                val1 = np.zeros(N)
                val1[valid1] = seq[idx1[valid1]]
                
                # Valid indices for i+k
                idx2 = i + ks
                valid2 = idx2 < N
                val2 = np.zeros(N)
                val2[valid2] = seq[idx2[valid2]]
                
                V = val1 + val2
                Delta_C = -2 * seq[i] * V
                
                C_new = C + Delta_C
                new_max = np.max(np.abs(C_new))
                
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_flip_idx = i
            
            if best_flip_idx != -1 and best_new_max < current_max:
                # Apply flip
                i = best_flip_idx
                seq[i] = -seq[i]
                # Update C
                ks = np.arange(N)
                idx1 = i - ks
                valid1 = idx1 >= 0
                val1 = np.zeros(N)
                val1[valid1] = seq[idx1[valid1]] # Note: seq[i] is already flipped, so we use new seq?
                # Wait, Delta_C was calculated using OLD seq[i].
                # So C_new is correct.
                C = C_new
                current_max = best_new_max
                best_local_max = current_max
                best_local_seq = seq.copy()
                improved = True
            else:
                break
                
        return best_local_seq, best_local_max

    best_score = np.inf
    best_seq = np.ones(N)
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        seq = np.random.choice([-1, 1], size=N)
        optimized_seq, current_max = local_search(seq, max_iterations=500)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = optimized_seq
            
    return best_seq
