import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    n = 128
    target_low = 128
    target_high = 256
    
    # Initial guess: A subset of [0, 128]
    # To cover [128, 256], we need elements close to 64 on average.
    # If A is [64, 128], sums are [128, 256].
    # Let's start with a random valid-ish set or the interval.
    # A = [64, 128] has size 65.
    # Sums of [64, 128]: min 128, max 256.
    # Representation count for sum s in [128, 256]:
    # If s in [128, 192]: pairs (a, s-a) where 64<=a<=s-a<=128 => 64<=a<=s/2.
    # Count = floor(s/2) - 64 + 1.
    # Max at s=192: 96-64+1 = 33.
    # If s in [193, 256]: pairs (a, s-a) where 64<=a<=128 and s-a>=a => a >= s-128, a <= 128.
    # Count = 128 - (s-128) + 1 = 257 - s.
    # Max at s=193: 257-193 = 64? No, wait.
    # Let's re-evaluate peak for [64, 128].
    # s=128: (64,64) -> 1
    # s=129: (64,65) -> 1
    # ...
    # s=192: (64,128)...(96,96). 128-64+1 = 65? No.
    # a from 64 to 96. 96-64+1 = 33.
    # s=193: a from 65 to 128. 128-65+1 = 64? No, a <= s-a => a <= 96.5. So a in [65, 96]. 32.
    # Wait, for s=193, a can be 65..96 (32 pairs) and 97..128 (31 pairs)?
    # Unordered pairs: a <= b.
    # If a=97, b=96 -> invalid.
    # So for s=193, max a is floor(193/2)=96.
    # Min a is s-128 = 65.
    # Count = 96 - 65 + 1 = 32.
    # So peak is 33 at s=192.
    
    # Can we do better?
    # If we remove 64 and 128?
    # A = [65, 127]. Sums [130, 254]. Doesn't cover 128, 129, 255, 256.
    # We MUST cover 128 and 256.
    # 128 = a+b. Since a,b in [0,128], and we want to cover 128.
    # If we don't have small numbers, we can't make 128?
    # If A contains 0, then 128 = 0+128.
    # If A contains 1, 128 = 1+127, etc.
    # To minimize peak, we want to spread out the sums.
    # A structured set like a "Sidon-like" or specific modular construction might work.
    # But let's try local search.
    
    # State: boolean array A of length 129 (indices 0..128)
    # Start with A = [64, 128]
    A = np.zeros(129, dtype=bool)
    A[64:129] = True
    
    def get_peak(A):
        # Compute representation counts for sums 128..256
        # Use convolution?
        # A is length 129. A+A is length 257.
        # Convolution of A with A gives counts of ordered pairs (i,j) such that i+j=k.
        # We want unordered pairs with i<=j.
        # Let C = conv(A, A).
        # For k, number of ordered pairs is C[k].
        # Number of unordered pairs = (C[k] + diag[k]) / 2, where diag[k] is 1 if k/2 in A and k is even, else 0.
        # Actually, simpler:
        # C[k] = sum_{i=0}^{128} A[i] * A[k-i]
        # This counts (i, k-i).
        # If i != k-i, it counts both (i, k-i) and (k-i, i).
        # If i == k-i (i.e., 2i=k), it counts once.
        # So unordered count U[k] = (C[k] + (1 if 2i=k and A[i] else 0)) // 2 ?
        # No.
        # C[k] = 2 * (number of pairs i<j with i+j=k) + (1 if 2i=k and A[i] else 0)
        # We want N[k] = (number of pairs i<=j with i+j=k)
        # N[k] = (number of pairs i<j) + (1 if 2i=k and A[i] else 0)
        # N[k] = (C[k] - diag[k]) / 2 + diag[k] = (C[k] + diag[k]) / 2
        # where diag[k] is 1 if k is even and A[k/2] is True, else 0.
        
        C = np.convolve(A, A, mode='full')
        # C indices 0..256. We care about 128..256.
        diag = np.zeros(257, dtype=int)
        # k is even, i=k/2
        evens = np.arange(0, 257, 2)
        halves = evens // 2
        # Check if halves are within 0..128
        valid_halves = (halves >= 0) & (halves <= 128)
        diag[evens[valid_halves]] = A[halves[valid_halves]].astype(int)
        
        U = (C + diag) // 2
        return U[128:257].max()

    current_peak = get_peak(A)
    
    # Local search
    # Move: swap a True to False and a False to True.
    # Or add/remove if size is not fixed? 
    # The problem says "subset A of [0,128]". It doesn't fix the size.
    # However, smaller sets generally have lower peaks, but must cover the range.
    # Minimum size to cover [128, 256] with sums?
    # If |A|=k, max sums is k(k+1)/2. We need 129 distinct sums.
    # k(k+1)/2 >= 129 => k ~ 16.
    # But the sums must be exactly [128, 256].
    # This is a very tight constraint.
    # Let's stick to size ~65 initially and see if we can reduce peak.
    
    # Randomize start?
    # Let's try a few random restarts or just one long annealing.
    
    best_A = A.copy()
    best_peak = current_peak
    
    # Annealing parameters
    T = 1.0
    cool = 0.999
    
    # Indices for selection
    idxs = np.arange(129)
    
    while time.time() < deadline:
        # Pick a random 1 to turn off
        ones = np.where(A)[0]
        if len(ones) == 0: break
        i_off = ones[np.random.randint(len(ones))]
        
        # Pick a random 0 to turn on
        zeros = np.where(~A)[0]
        if len(zeros) == 0: break
        i_on = zeros[np.random.randint(len(zeros))]
        
        # Perform swap
        A[i_off] = False
        A[i_on] = True
        
        new_peak = get_peak(A)
        
        if new_peak < current_peak:
            current_peak = new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = A.copy()
        else:
            # Accept with probability exp(-delta/T)
            delta = new_peak - current_peak
            if delta > 0:
                if np.random.random() < np.exp(-delta / T):
                    current_peak = new_peak
                else:
                    # Revert
                    A[i_off] = True
                    A[i_on] = False
            else:
                # delta == 0, accept
                current_peak = new_peak
        
        T *= cool
        if T < 0.01:
            T = 0.01 # Keep some noise to escape local minima
            
        # If we've been stuck, maybe restart?
        # For now, just keep going.
        
    # Check validity of best_A
    # Ensure all sums 128..256 are covered (multiplicity > 0)
    C = np.convolve(best_A, best_A, mode='full')
    diag = np.zeros(257, dtype=int)
    evens = np.arange(0, 257, 2)
    halves = evens // 2
    valid_halves = (halves >= 0) & (halves <= 128)
    diag[evens[valid_halves]] = best_A[halves[valid_halves]].astype(int)
    U = (C + diag) // 2
    if np.any(U[128:257] == 0):
        # Invalid, fall back to interval
        return list(range(64, 129))
        
    return list(np.where(best_A)[0])
