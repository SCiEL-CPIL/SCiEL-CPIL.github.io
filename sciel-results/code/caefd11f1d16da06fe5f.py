import numpy as np
import time

N = 400

def construct():
    """
    Returns a sequence of signed step heights (length N) that minimizes 
    the upper bound on C3 (interpreted as peak autocorrelation sidelobe).
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    def compute_score(h):
        """
        Computes the peak autocorrelation sidelobe.
        """
        H = np.fft.fft(h)
        R = np.fft.ifft(np.abs(H) ** 2).real
        return np.max(np.abs(R[1:]))

    def thue_morse(n):
        """
        Generate Thue-Morse sequence of length n.
        Thue-Morse has excellent autocorrelation properties.
        """
        seq = np.ones(n, dtype=np.int8)
        for i in range(n):
            if bin(i).count('1') % 2 == 1:
                seq[i] = -1
        return seq

    def hadamard_row(n, row):
        """
        Generate a row of a Hadamard matrix (Walsh function).
        """
        seq = np.ones(n, dtype=np.int8)
        for i in range(n):
            if bin(i & row).count('1') % 2 == 1:
                seq[i] = -1
        return seq

    best_h = None
    best_score = float('inf')

    # Try multiple deterministic initializations with known good spectral properties
    initializations = []
    
    # Thue-Morse - excellent autocorrelation
    initializations.append(thue_morse(N))
    
    # Walsh/Hadamard rows - orthogonal sequences
    for row in [1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256]:
        if row < N:
            initializations.append(hadamard_row(N, row))
    
    # Random fallback
    initializations.append(np.random.choice([-1, 1], N))

    # For each initialization, run simulated annealing with block-based moves
    T_max = 5.0
    T_min = 0.001
    cooling_rate = 0.9999

    for init_idx, h0 in enumerate(initializations):
        if time.time() > deadline - 10:
            break
            
        h = h0.copy()
        current_score = compute_score(h)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
        
        # Early termination if we found a very good solution
        if best_score < 1.0:
            break
            
        T = T_max
        no_improve_count = 0
        max_iter = 20000
        
        for i in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Block-based move: flip a contiguous block
            # This exploits the structure of autocorrelation
            block_len = np.random.choice([1, 2, 4, 8, 16, 32, 64])
            if block_len >= N:
                block_len = 1
            start_idx = np.random.randint(0, N - block_len + 1)
            
            h[start_idx:start_idx+block_len] *= -1
            new_score = compute_score(h)
            delta = new_score - current_score
            
            if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                h[start_idx:start_idx+block_len] *= -1
                no_improve_count += 1
            
            T *= cooling_rate
            
            # Reheat if stuck
            if no_improve_count > 1000:
                T = T_max * 0.5
                no_improve_count = 0
                
            if T < T_min:
                break

    if best_h is None:
        best_h = thue_morse(N)
        
    return best_h
