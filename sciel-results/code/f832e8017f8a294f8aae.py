import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s buffer for 60s limit
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129
    all_vals = np.arange(n + 1)
    
    # Precompute for each target t in [128, 256] the list of pairs (a, b) with a <= b, a+b=t, 0<=a,b<=128
    # Store as list of (a, b) tuples for each target index
    target_pairs = []
    for i in range(num_targets):
        t = target_lo + i
        pairs = []
        a_min = max(0, t - n)
        a_max = min(t // 2, n)
        for a in range(a_min, a_max + 1):
            b = t - a
            if b >= a and b <= n:
                pairs.append((a, b))
        target_pairs.append(pairs)
    
    def compute_counts(freq):
        """Compute representation counts for all targets [128, 256]."""
        counts = np.zeros(num_targets, dtype=np.int32)
        for i in range(num_targets):
            t = target_lo + i
            c = 0
            for a, b in target_pairs[i]:
                if freq[a] and freq[b]:
                    c += 1
            counts[i] = c
        return counts
    
    def peak_of(counts):
        return int(np.max(counts))
    
    def is_complete(counts):
        return bool(np.min(counts) >= 1)
    
    def local_search(A_init, max_iters=200000):
        A = sorted(set(A_init))
        if not A:
            return A, float('inf')
        
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A:
            freq[x] = 1
        
        counts = compute_counts(freq)
        cur_min = int(np.min(counts))
        cur_peak = int(np.max(counts))
        A_set = set(A)
        
        # If initial set is invalid, try to fix by adding elements
        if cur_min < 1:
            for _ in range(500):
                if time.time() > deadline - 2:
                    break
                gaps = np.where(counts == 0)[0]
                if len(gaps) == 0:
                    break
                g = gaps[random.randint(0, len(gaps) - 1)]
                t = target_lo + g
                # Find a pair that covers this gap
                pairs = target_pairs[g]
                if not pairs:
                    continue
                # Try adding one element from a random pair
                a, b = pairs[random.randint(0, len(pairs) - 1)]
                if a not in A_set:
                    freq[a] = 1
                    new_counts = compute_counts(freq)
                    new_min = int(np.min(new_counts))
                    if new_min > cur_min:
                        A_set.add(a)
                        A.append(a)
                        A.sort()
                        counts = new_counts
                        cur_min = new_min
                        cur_peak = int(np.max(counts))
                    else:
                        freq[a] = 0
                elif b not in A_set and b != a:
                    freq[b] = 1
                    new_counts = compute_counts(freq)
                    new_min = int(np.min(new_counts))
                    if new_min > cur_min:
                        A_set.add(b)
                        A.append(b)
                        A.sort()
                        counts = new_counts
                        cur_min = new_min
                        cur_peak = int(np.max(counts))
                    else:
                        freq[b] = 0
        
        if cur_min < 1:
            return A, float('inf')
        
        # Now we have a valid set. Try to reduce peak.
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # Find target with max count
            max_targets = np.where(counts == cur_peak)[0]
            if len(max_targets) == 0:
                break
            
            t_idx = max_targets[random.randint(0, len(max_targets) - 1)]
            t_val = target_lo + t_idx
            pairs = target_pairs[t_idx]
            
            # Find pairs in A that contribute to this target
            active_pairs = []
            for a, b in pairs:
                if freq[a] and freq[b]:
                    active_pairs.append((a, b))
            
            if not active_pairs:
                continue
            
            # Try to reduce multiplicity at this target
            a, b = active_pairs[random.randint(0, len(active_pairs) - 1)]
            
            tried = False
            
            # Option 1: Remove a (if a != b, removing a removes this pair)
            if a != b:
                freq[a] = 0
                new_counts = compute_counts(freq)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                if new_min >= 1:
                    if new_peak <= cur_peak:
                        A_set.discard(a)
                        A.remove(a)
                        counts = new_counts
                        cur_peak = new_peak
                        cur_min = new_min
                        tried = True
                        continue
                freq[a] = 1
            
            # Option 2: Remove b (if a != b)
            if a != b:
                freq[b] = 0
                new_counts = compute_counts(freq)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                if new_min >= 1:
                    if new_peak <= cur_peak:
                        A_set.discard(b)
                        A.remove(b)
                        counts = new_counts
                        cur_peak = new_peak
                        cur_min = new_min
                        tried = True
                        continue
                freq[b] = 1
            
            # Option 3: Swap a with x
            x = random.randint(0, n)
            if x not in A_set and x != a:
                freq[a] = 0
                freq[x] = 1
                new_counts = compute_counts(freq)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                if new_min >= 1:
                    if new_peak < cur_peak:
                        A_set.discard(a)
                        A_set.add(x)
                        A[A.index(a)] = x
                        A.sort()
                        counts = new_counts
                        cur_peak = new_peak
                        cur_min = new_min
                        tried = True
                        continue
                    elif new_peak == cur_peak and random.random() < 0.05:
                        A_set.discard(a)
                        A_set.add(x)
                        A[A.index(a)] = x
                        A.sort()
                        counts = new_counts
                        cur_min = new_min
                        tried = True
                        continue
                freq[a] = 1
                freq[x] = 0
            
            # Option 4: Swap b with x (if a != b)
            if a != b:
                x = random.randint(0, n)
                if x not in A_set and x != b:
                    freq[b] = 0
                    freq[x] = 1
                    new_counts = compute_counts(freq)
                    new_min = int(np.min(new_counts))
                    new_peak = int(np.max(new_counts))
                    if new_min >= 1:
                        if new_peak < cur_peak:
                            A_set.discard(b)
                            A_set.add(x)
                            A[A.index(b)] = x
                            A.sort()
                            counts = new_counts
                            cur_peak = new_peak
                            cur_min = new_min
                            tried = True
                            continue
                        elif new_peak == cur_peak and random.random() < 0.05:
                            A_set.discard(b)
                            A_set.add(x)
                            A[A.index(b)] = x
                            A.sort()
                            counts = new_counts
                            cur_min = new_min
                            tried = True
                            continue
                    freq[b] = 1
                    freq[x] = 0
            
            # Option 5: Add a new element that doesn't increase peak
            x = random.randint(0, n)
            if x not in A_set:
                freq[x] = 1
                new_counts = compute_counts(freq)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                if new_min >= 1 and new_peak < cur_peak:
                    A_set.add(x)
                    A.append(x)
                    A.sort()
                    counts = new_counts
                    cur_peak = new_peak
                    cur_min = new_min
                    tried = True
                    continue
                freq[x] = 0
        
        return A, cur_peak
    
    best_A = None
    best_peak = float('inf')
    
    random.seed(42)
    
    # Try various sizes and random restarts
    for k in range(20, 50, 2):
        if time.time() > deadline - 5:
            break
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            A_rand = sorted(random.sample(range(n + 1), k))
            A_ls, peak_ls = local_search(A_rand)
            if peak_ls < best_peak:
                # Verify
                freq = np.zeros(n + 1, dtype=np.int32)
                for x in A_ls:
                    freq[x] = 1
                counts = compute_counts(freq)
                if np.min(counts) >= 1:
                    best_A = A_ls
                    best_peak = peak_ls
    
    # If we didn't find a good one, fallback to a dense set
    if best_A is None:
        best_A = list(range(64, 129))
    
    return list(best_A)
