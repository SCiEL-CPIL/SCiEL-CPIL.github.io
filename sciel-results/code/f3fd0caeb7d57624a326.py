import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 75
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1
    
    # Precompute: for each x, the self-pair target
    self_target = np.full(n + 1, -1, dtype=np.int32)
    for x in range(n + 1):
        s = 2 * x
        if target_lo <= s <= target_hi:
            self_target[x] = s - target_lo
    
    def init_counts(A):
        A_sorted = sorted(A)
        counts = np.zeros(num_targets, dtype=np.int32)
        for i in range(len(A_sorted)):
            a = A_sorted[i]
            for j in range(i, len(A_sorted)):
                s = a + A_sorted[j]
                if target_lo <= s <= target_hi:
                    counts[s - target_lo] += 1
        return counts
    
    def is_complete(counts):
        return bool(np.all(counts > 0))
    
    def peak_of(counts):
        return int(np.max(counts))
    
    def get_contribution(x, A):
        """Get the contribution vector of x given the current set A (x must be in A)."""
        delta = np.zeros(num_targets, dtype=np.int32)
        st = self_target[x]
        if st >= 0:
            delta[st] = 1
        for a in A:
            if a < x:
                s = a + x
                if target_lo <= s <= target_hi:
                    delta[s - target_lo] = 1
        return delta
    
    # This is still O(k) per contribution. Let me think differently.
    
    # Alternative: represent counts as a 2D structure
    # counts[t] = number of pairs (a,b) in A, a<=b, a+b=t
    # = sum over a in A, a <= t-a, a+b=t of 1
    # = sum over a in A, a <= t/2, (t-a) in A of 1
    
    # For fast updates, I can maintain a frequency array freq[x] = 1 if x in A else 0
    # Then counts[t] = sum over a in [max(0,t-128), min(128,t//2)] of freq[a] * freq[t-a]
    # But this is O(n) per target, O(n^2) total.
    
    # Better: use convolution!
    # The number of representations of t as a+b with a,b in A (ordered, a<=b) is:
    # (conv[freq,freq][t] + (1 if 2t is even and freq[t/2]>0 else 0)) / 2
    # Wait, that's for ordered pairs. For unordered:
    # # unordered pairs = (ordered pairs + diagonal) / 2
    # where diagonal = 1 if 2t is even and t/2 in A
    
    # Actually: ordered pairs (a,b) with a+b=t: conv[freq,freq][t]
    # Unordered pairs (a,b) with a<=b, a+b=t:
    #   = (conv[freq,freq][t] + diag[t]) / 2
    # where diag[t] = 1 if t is even and t/2 in A, else 0
    
    # Let me verify: if a≠b, each unordered pair {a,b} corresponds to 2 ordered pairs.
    # If a=b, it corresponds to 1 ordered pair.
    # So: #unordered = (#ordered - #diagonal) / 2 + #diagonal = (#ordered + #diagonal) / 2
    
    # This is great! I can use FFT-based convolution for fast computation.
    
    def compute_counts_conv(freq):
        """Compute representation counts using convolution."""
        # freq is an array of length n+1
        # conv = freq * freq (convolution)
        conv = np.convolve(freq, freq, mode='full')
        # conv[t] = number of ordered pairs (a,b) with a+b=t
        counts = np.zeros(num_targets, dtype=np.int32)
        for t in range(target_lo, target_hi + 1):
            ordered = conv[t]
            if t % 2 == 0 and t // 2 <= n:
                diag = freq[t // 2]
            else:
                diag = 0
            counts[t - target_lo] = (ordered + diag) // 2
        return counts
    
    # Seed
    best_A = list(range(64, 129))
    best_freq = np.zeros(n + 1, dtype=np.int32)
    for x in best_A:
        best_freq[x] = 1
    best_counts = compute_counts_conv(best_freq)
    best_peak = peak_of(best_counts)
    
    def local_search_conv(A, max_iter=50000):
        A = sorted(set(A))
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A:
            freq[x] = 1
        counts = compute_counts_conv(freq)
        if not is_complete(counts):
            return A, peak_of(counts), counts
        cur_peak = peak_of(counts)
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            improved = False
            A_set = set(A)
            
            # Try removals
            for x in A:
                freq[x] = 0
                new_counts = compute_counts_conv(freq)
                if np.all(new_counts > 0):
                    new_peak = peak_of(new_counts)
                    if new_peak < cur_peak:
                        A = [a for a in A if a != x]
                        counts = new_counts
                        cur_peak = new_peak
                        improved = True
                        break
                    else:
                        freq[x] = 1
                else:
                    freq[x] = 1
            
            if improved:
                continue
            if time.time() > deadline - 3:
                break
            
            # Try additions
            for x in range(n + 1):
                if x in A_set:
                    continue
                freq[x] = 1
                new_counts = compute_counts_conv(freq)
                if np.all(new_counts > 0):
                    new_peak = peak_of(new_counts)
                    if new_peak < cur_peak:
                        A = sorted(A + [x])
                        A_set = set(A)
                        counts = new_counts
                        cur_peak = new_peak
                        improved = True
                        break
                    else:
                        freq[x] = 0
                else:
                    freq[x] = 0
            
            if improved:
                continue
            if time.time() > deadline - 3:
                break
            
            # Try swaps (sampled)
            found_swap = False
            for _ in range(10):
                if time.time() > deadline - 3:
                    break
                i = np.random.randint(len(A))
                old_x = A[i]
                new_x = np.random.randint(n + 1)
                if new_x in A_set or new_x == old_x:
                    continue
                freq[old_x] = 0
                freq[new_x] = 1
                new_counts = compute_counts_conv(freq)
                if np.all(new_counts > 0):
                    new_peak = peak_of(new_counts)
                    if new_peak < cur_peak:
                        A = sorted([a for a in A if a != old_x] + [new_x])
                        A_set = set(A)
                        counts = new_counts
                        cur_peak = new_peak
                        improved = True
                        found_swap = True
                        break
                    else:
                        freq[old_x] = 1
                        freq[new_x] = 0
                else:
                    freq[old_x] = 1
                    freq[new_x] = 0
            
            if not improved:
                break
        
        return A, cur_peak, counts
    
    # Run from seed
    A_ls, peak_ls, counts_ls = local_search_conv(best_A)
    if is_complete(counts_ls) and peak_ls < best_peak:
        best_A = A_ls
        best_peak = peak_ls
        best_counts = counts_ls
    
    # Random restarts
    np.random.seed(42)
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        k = int(np.random.randint(40, 70))
        A_rand = sorted(np.random.choice(n + 1, size=k, replace=False).tolist())
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A_rand:
            freq[x] = 1
        counts_init = compute_counts_conv(freq)
        if not is_complete(counts_init):
            continue
        p_init = peak_of(counts_init)
        if p_init >= best_peak:
            continue
        A_ls, peak_ls, counts_ls = local_search_conv(A_rand, max_iter=10000)
        if is_complete(counts_ls) and peak_ls < best_peak:
            best_A = A_ls
            best_peak = peak_ls
            best_counts = counts_ls
    
    # Final validation
    final_freq = np.zeros(n + 1, dtype=np.int32)
    for x in best_A:
        final_freq[x] = 1
    final_counts = compute_counts_conv(final_freq)
    if not is_complete(final_counts):
        return list(range(64, 129))
    
    return list(best_A)
