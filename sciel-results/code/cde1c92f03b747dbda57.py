import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 40s margin before 150s kill

    def gen_m_seq(m, poly_taps):
        state = 1
        seq = []
        for _ in range(2**m - 1):
            seq.append(state & 1)
            fb = 0
            for tap in poly_taps:
                if tap == 0:
                    fb ^= (state & 1)
                else:
                    fb ^= (state >> tap) & 1
            state = ((state >> 1) | (fb << (m - 1))) & ((1 << m) - 1)
        return np.array(seq, dtype=np.float64) * 2 - 1

    # Generate M-sequence of length 511
    m_seq_511 = gen_m_seq(9, [4, 0])
    h = m_seq_511[:N].copy()

    def compute_acf(h):
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.fft(h, n=n_fft)
        power = np.abs(F) ** 2
        acf = np.fft.ifft(power).real
        return acf[1:N]

    def get_metric(acf):
        return np.max(np.abs(acf))

    # Initial metric
    acf = compute_acf(h)
    best_val = get_metric(acf)
    best_h = h.copy()

    k = np.arange(1, N)

    def local_search_greedy(h, max_passes=500):
        acf = compute_acf(h)
        current_val = get_metric(acf)
        
        improved = True
        passes = 0
        while improved and passes < max_passes:
            if time.time() > deadline - 20.0:
                break
            improved = False
            passes += 1
            for i in range(N):
                if time.time() > deadline - 20.0:
                    return h, current_val
                
                idx1 = i + k
                valid1 = idx1 < N
                term1 = np.zeros(N-1)
                if np.any(valid1):
                    term1[valid1] = h[idx1[valid1]]
                
                idx2 = i - k
                valid2 = idx2 >= 0
                term2 = np.zeros(N-1)
                if np.any(valid2):
                    term2[valid2] = h[idx2[valid2]]
                
                delta = -2.0 * h[i] * (term1 + term2)
                
                new_acf = acf + delta
                new_val = get_metric(new_acf)
                
                if new_val < current_val:
                    h[i] *= -1.0
                    acf = new_acf
                    current_val = new_val
                    improved = True
        
        return h, current_val

    # Run Greedy Local Search
    h, val = local_search_greedy(h, max_passes=200)
    if val < best_val:
        best_val = val
        best_h = h.copy()

    # Simulated Annealing to escape local minima
    def simulated_annealing(h, max_iters=20000):
        acf = compute_acf(h)
        current_val = get_metric(acf)
        
        T = 2.0
        T_min = 0.1
        cooling = 0.9999
        
        for _ in range(max_iters):
            if time.time() > deadline - 10.0:
                break
            
            i = np.random.randint(N)
            
            idx1 = i + k
            valid1 = idx1 < N
            term1 = np.zeros(N-1)
            if np.any(valid1):
                term1[valid1] = h[idx1[valid1]]
            
            idx2 = i - k
            valid2 = idx2 >= 0
            term2 = np.zeros(N-1)
            if np.any(valid2):
                term2[valid2] = h[idx2[valid2]]
            
            delta = -2.0 * h[i] * (term1 + term2)
            
            new_acf = acf + delta
            new_val = get_metric(new_acf)
            
            delta_val = new_val - current_val
            
            if delta_val < 0 or np.random.rand() < np.exp(-delta_val / T):
                h[i] *= -1.0
                acf = new_acf
                current_val = new_val
            
            T *= cooling
            if T < T_min:
                T = T_min
                
        return h, current_val

    # Run SA
    h, val = simulated_annealing(best_h, max_iters=50000)
    if val < best_val:
        best_val = val
        best_h = h.copy()
        
    # Final Polish
    h, val = local_search_greedy(best_h, max_passes=100)
    if val < best_val:
        best_val = val
        best_h = h.copy()

    return best_h
