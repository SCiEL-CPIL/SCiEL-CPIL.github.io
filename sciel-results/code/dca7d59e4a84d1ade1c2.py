import numpy as np
import time
import random

N = 128
LO, HI = 128, 256
TARGET_SUMS = np.arange(LO, HI + 1)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # State variables
    # A: set of ints
    # counts: numpy array of length 129, counts[i] corresponds to sum LO+i
    # energy: sum of squares of counts (proxy for flatness)
    # max_count: max of counts
    
    def get_initial_state():
        # Start with a reasonably dense random set
        # Size ~ 45-55
        size = random.randint(45, 55)
        A = set(random.sample(range(N + 1), size))
        return A

    def compute_counts(A):
        counts = np.zeros(HI - LO + 1, dtype=np.int64)
        A_list = list(A)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if LO <= s <= HI:
                    idx = s - LO
                    counts[idx] += 1
        return counts

    def is_valid(counts):
        return np.all(counts > 0)

    def calc_energy(counts):
        return int(np.sum(counts * counts))

    def calc_max(counts):
        return int(np.max(counts))

    # Initialize
    A = get_initial_state()
    counts = compute_counts(A)
    
    # Greedy fix coverage if needed
    # If not covered, add random elements until covered
    attempts = 0
    while not is_valid(counts) and attempts < 100:
        candidates = [i for i in range(N + 1) if i not in A]
        if not candidates:
            break
        a = random.choice(candidates)
        A.add(a)
        # Update counts incrementally
        for b in list(A):
            if b == a:
                continue # Already handled in loop? No, loop is over old A.
            # Wait, incremental update logic:
            # Adding 'a' to A:
            # For each b in OLD A:
            #   if b == a: counts[2a] += 1
            #   if b != a: counts[a+b] += 1
            # My loop above is wrong for incremental. Let's just recompute for init fix, it's fast enough for 100 steps.
            pass 
        counts = compute_counts(A)
        attempts += 1
        
    if not is_valid(counts):
        # Fallback to a known valid structure, e.g., all evens + one odd?
        # Evens cover even sums. We need odd sums.
        # If A has at least one odd, say 1, and all evens, then 1+even = odd.
        # So A = {0, 2, ..., 128} U {1} is valid?
        # Sums: even+even=even, even+1=odd, 1+1=2 (even).
        # Range [128, 256].
        # Even sums: 128, 130, ..., 256.
        # Odd sums: 129, 131, ..., 255.
        # Does {0, 2, ..., 128} U {1} cover all?
        # Even sum S: x+y=S. x,y even. x=2k, y=2m. 2(k+m)=S. k+m=S/2.
        # k,m in [0, 64]. Sum range [0, 128]. S/2 in [64, 128].
        # Can we form any sum in [64, 128] with two numbers in [0, 64]?
        # Max sum 64+64=128. Min sum 0+0=0.
        # Yes, for any S/2 in [64, 128], we can find k,m.
        # e.g., S=128 -> S/2=64. 0+64=64. (0, 128).
        # S=256 -> S/2=128. 64+64=128. (128, 128).
        # Odd sum S: x+y=S. One even, one odd (1).
        # y=1. x=S-1. x must be even and in [0, 128].
        # S in [129, 255] odd.
        # S=129 -> x=128 (in A).
        # S=255 -> x=254 (NOT in A, max is 128).
        # So {evens} U {1} does NOT cover high odd sums.
        # We need larger odds.
        # Let's just use a random restart if init fails, or a denser init.
        # For now, if init fails, return a dense set.
        return list(range(N + 1))

    current_energy = calc_energy(counts)
    current_max = calc_max(counts)
    
    best_A = set(A)
    best_counts = counts.copy()
    best_energy = current_energy
    best_max = current_max
    
    # Simulated Annealing
    T = 100.0
    cooling = 0.99
    min_T = 0.1
    min_size = 30
    
    while T > min_T and time.time() < deadline:
        # Propose move
        move_type = random.random()
        new_A = set(A)
        
        if move_type < 0.5:
            # Swap
            if len(A) == 0:
                continue
            a = random.choice(list(A))
            candidates = [i for i in range(N + 1) if i not in A]
            if not candidates:
                continue
            b = random.choice(candidates)
            new_A.remove(a)
            new_A.add(b)
        elif move_type < 0.8:
            # Remove
            if len(A) <= min_size:
                continue
            a = random.choice(list(A))
            new_A.remove(a)
        else:
            # Add
            candidates = [i for i in range(N + 1) if i not in A]
            if not candidates:
                continue
            b = random.choice(candidates)
            new_A.add(b)
            
        if new_A == A:
            continue
            
        # Evaluate new_A
        # To speed up, we can do incremental update, but for simplicity and correctness in this rewrite,
        # let's use the vectorized convolution check for the NEW state, but only if we expect it to be close.
        # Actually, O(N^2) python loop for counts is ~ 50*50/2 = 1250 ops. 
        # np.convolve is C-optimized. 
        # Let's stick to np.convolve for simplicity in this "new design" unless it's too slow.
        # 1250 python ops vs 1 convolve. Convolve is likely faster.
        
        arr = np.zeros(N + 1, dtype=np.int64)
        for a in new_A:
            arr[a] = 1
        conv = np.convolve(arr, arr, mode='full')
        new_counts = conv[TARGET_SUMS]
        
        # Diagonal correction
        Aset = new_A
        diag = np.zeros(len(TARGET_SUMS), dtype=np.int64)
        for i, s in enumerate(TARGET_SUMS):
            if s % 2 == 0:
                if (s // 2) in Aset:
                    diag[i] = 1
        new_counts = (new_counts + diag) // 2
        
        if not is_valid(new_counts):
            # Reject invalid
            T *= cooling
            continue
            
        new_energy = calc_energy(new_counts)
        new_max = calc_max(new_counts)
        
        delta_energy = new_energy - current_energy
        
        # Acceptance criterion
        if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
            # Accept
            A = new_A
            counts = new_counts
            current_energy = new_energy
            current_max = new_max
            
            # Update best
            # We prefer lower max, then lower energy
            if new_max < best_max or (new_max == best_max and new_energy < best_energy):
                best_max = new_max
                best_energy = new_energy
                best_A = set(A)
                best_counts = counts.copy()
                
        T *= cooling
        
    # Final check: ensure best_A is valid
    # Re-verify best_A just in case
    arr = np.zeros(N + 1, dtype=np.int64)
    for a in best_A:
        arr[a] = 1
    conv = np.convolve(arr, arr, mode='full')
    final_counts = conv[TARGET_SUMS]
    Aset = best_A
    diag = np.zeros(len(TARGET_SUMS), dtype=np.int64)
    for i, s in enumerate(TARGET_SUMS):
        if s % 2 == 0:
            if (s // 2) in Aset:
                diag[i] = 1
    final_counts = (final_counts + diag) // 2
    
    if not is_valid(final_counts):
        # Fallback
        return list(range(N + 1))
        
    return list(best_A)
