import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _compute_autocorr(s):
    n = len(s)
    f = np.fft.fft(s, n=2*n)
    acf = np.fft.ifft(f * np.conj(f)).real
    return acf[:n]

def _get_peak(acf):
    if len(acf) <= 1:
        return 0.0
    return np.max(np.abs(acf[1:]))

def construct():
    deadline = time.time() + 140
    
    best_s = None
    best_peak = float('inf')
    
    # Multi-start SA
    num_restarts = 10 # Adjust based on time
    
    # We will run SA for a fixed number of steps or time slice
    # Let's do time-sliced restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        # Try different inits: Random, Thue-Morse, etc.
        if restart == 0:
            # Thue-Morse
            s = np.array([1.0 if (i & 1) == 0 else -1.0 for i in range(N)]) # Not true Thue-Morse
            # True Thue-Morse
            tm = np.zeros(N)
            tm[0] = 1
            for i in range(1, N):
                # popcount mod 2
                # Use int.bit_count() if py3.10+, else bin(i).count('1')
                # Let's use a fast numpy trick or just random for simplicity?
                # Thue-Morse is good start.
                # t(i) = 1 if popcount(i)%2==0 else -1
                # We can compute popcount efficiently?
                # For N=400, loop is fine.
                pc = 0
                x = i
                while x:
                    pc += x & 1
                    x >>= 1
                tm[i] = 1.0 if pc % 2 == 0 else -1.0
            s = tm
        else:
            s = np.random.choice([1.0, -1.0], N)
            
        acf = _compute_autocorr(s)
        current_peak = _get_peak(acf)
        
        if current_peak < best_peak:
            best_peak = current_peak
            best_s = s.copy()
            
        # SA Loop
        T = 5.0
        T_min = 0.01
        alpha = 0.9999
        
        # Run SA for a certain number of steps
        steps_per_restart = 50000
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            j = np.random.randint(0, N)
            old_val = s[j]
            new_val = -old_val
            delta_val = new_val - old_val
            
            # Vectorized update
            # k in 1..N-1
            # Term 1: k < N - j
            # Term 2: k <= j
            
            K1 = N - 1 - j
            K2 = j
            
            # Update acf[1:K1+1] with s[j+1:j+K1+1]
            # Update acf[1:K2+1] with s[j-1:j-K2-1:-1] -> s[j-K2:j-1] reversed?
            # s[j-k] for k=1..K2 is s[j-1], s[j-2]... s[j-K2]
            # Slice: s[j-1:j-K2-1:-1]
            
            # We need to revert if rejected, so we copy acf?
            # Copying 400 floats is fast.
            old_acf = acf.copy()
            
            if K1 > 0:
                acf[1:K1+1] += delta_val * s[j+1:j+K1+1]
            if K2 > 0:
                # s[j-k] for k=1..K2
                # indices: j-1, j-2, ..., j-K2
                # slice: s[j-1:j-K2-1:-1]
                acf[1:K2+1] += delta_val * s[j-1:j-K2-1:-1]
                
            new_peak = _get_peak(acf)
            
            if new_peak < current_peak or np.random.rand() < np.exp((current_peak - new_peak) / T):
                current_peak = new_peak
                s[j] = new_val
                if new_peak < best_peak:
                    best_peak = new_peak
                    best_s = s.copy()
            else:
                acf = old_acf
                # s[j] remains old_val (we didn't change it yet)
                # Wait, in the accepted case we set s[j]=new_val.
                # In rejected case, s[j] is still old_val.
                # But we need to make sure s[j] is consistent.
                # In the code above, I only update s[j] in the accepted branch.
                # This is correct.
                
            T *= alpha
            if T < T_min:
                T = T_min

    return best_s
# EVOLVE-BLOCK-END
