import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s buffer for 60s limit
    
    n = 128
    target_lo = 128
    target_hi = 256
    num_targets = target_hi - target_lo + 1  # 129
    all_vals = np.arange(n + 1)
    
    # Precompute for each target t in [128, 256] the list of pairs (a, b) with a <= b, a+b=t, 0<=a,b<=128
    # We'll store these as lists of (a, b) tuples for each target index
    target_pairs = []
    for i in range(num_targets):
        t = target_lo + i
        pairs = []
        for a in range(t // 2 + 1):  # a <= b means a <= t-a => 2a <= t => a <= t//2
            b = t - a
            if 0 <= a <= n and 0 <= b <= n:
                pairs.append((a, b))
        target_pairs.append(pairs)
    
    # Precompute: for each value v in [0, 128], which targets does it participate in?
    # And for each target, which values can cover it (as part of some pair)?
    # Actually, let's precompute the incidence: for each target i, the set of values that appear in any pair for target i
    target_values = []
    for i in range(num_targets):
        vals = set()
        for a, b in target_pairs[i]:
            vals.add(a)
            vals.add(b)
        target_values.append(vals)
    
    def compute_counts(freq):
        """
        freq: binary array of length n+1
        Returns counts array of length num_targets for targets [128, 256]
        """
        # Direct counting: for each target, count unordered pairs (a,b) with a<=b, a+b=t
        counts = np.zeros(num_targets, dtype=np.int32)
        for i in range(num_targets):
            t = target_lo + i
            c = 0
            for a in range(t // 2 + 1):
                b = t - a
                if 0 <= a <= n and 0 <= b <= n and freq[a] and freq[b]:
                    c += 1
            counts[i] = c
        return counts
    
    def evaluate(A):
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A:
            freq[x] = 1
        counts = compute_counts(freq)
        min_c = int(np.min(counts))
        max_c = int(np.max(counts))
        return min_c, max_c, counts
    
    def try_fix_gaps(A_set, freq, counts, max_fix_attempts=200):
        """
        Try to fix gaps (targets with count 0) by adding elements.
        Returns (A_set, freq, counts, cur_min, cur_peak) or None if failed.
        """
        cur_min = int(np.min(counts))
        cur_peak = int(np.max(counts))
        
        for _ in range(max_fix_attempts):
            if time.time() > deadline - 3:
                return None
            if cur_min >= 1:
                break
            
            gaps = np.where(counts == 0)[0]
            if len(gaps) == 0:
                break
            
            # Pick a random gap
            g_idx = gaps[random.randint(0, len(gaps) - 1)]
            t_val = target_lo + g_idx
            
            # Find all pairs (a, b) with a+b = t_val
            pairs = target_pairs[g_idx]
            if not pairs:
                continue
            
            # Try to add an element from a random pair
            random.shuffle(pairs)
            found = False
            for a, b in pairs:
                # Try adding a
                if a not in A_set:
                    freq[a] = 1
                    new_counts = compute_counts(freq)
                    new_min = int(np.min(new_counts))
                    if new_min > cur_min or (new_min == cur_min and int(np.sum(new_counts > 0)) > int(np.sum(counts > 0))):
                        A_set.add(a)
                        counts = new_counts
                        cur_min = new_min
                        cur_peak = int(np.max(counts))
                        found = True
                        break
                    freq[a] = 0
                
                # Try adding b
                if b not in A_set:
                    freq[b] = 1
                    new_counts = compute_counts(freq)
                    new_min = int(np.min(new_counts))
                    if new_min > cur_min or (new_min == cur_min and int(np.sum(new_counts > 0)) > int(np.sum(counts > 0))):
                        A_set.add(b)
                        counts = new_counts
                        cur_min = new_min
                        cur_peak = int(np.max(counts))
                        found = True
                        break
                    freq[b] = 0
                
                if found:
                    break
            
            if not found:
                # Try a random swap
                if A_set:
                    a_out = random.choice(list(A_set))
                    a_in = random.randint(0, n)
                    if a_in not in A_set:
                        freq[a_out] = 0
                        freq[a_in] = 1
                        new_counts = compute_counts(freq)
                        new_min = int(np.min(new_counts))
                        if new_min > cur_min:
                            A_set.discard(a_out)
                            A_set.add(a_in)
                            counts = new_counts
                            cur_min = new_min
                            cur_peak = int(np.max(counts))
                        else:
                            freq[a_out] = 1
                            freq[a_in] = 0
        
        if cur_min < 1:
            return None
        return A_set, freq, counts, cur_min, cur_peak
    
    def local_search(A_init, max_iters=30000):
        A_set = set(A_init)
        if not A_set:
            return None, float('inf')
        
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A_set:
            freq[x] = 1
        
        min_c, max_c, counts = evaluate(A_set)
        
        # Try to fix gaps if any
        if min_c < 1:
            result = try_fix_gaps(A_set, freq, counts)
            if result is None:
                return None, float('inf')
            A_set, freq, counts, min_c, max_c = result
        
        cur_min = min_c
        cur_peak = max_c
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            if cur_min < 1:
                result = try_fix_gaps(A_set, freq, counts)
                if result is None:
                    break
                A_set, freq, counts, cur_min, cur_peak = result
                continue
            
            # Find target with max count
            max_targets = np.where(counts == cur_peak)[0]
            if len(max_targets) == 0:
                break
            
            t_idx = max_targets[random.randint(0, len(max_targets) - 1)]
            t_val = target_lo + t_idx
            
            # List pairs in A with a+b = t_val
            pairs = []
            for a, b in target_pairs[t_idx]:
                if a in A_set and b in A_set:
                    pairs.append((a, b))
            
            if not pairs:
                continue
            
            # Pick a random pair
            a, b = pairs[random.randint(0, len(pairs) - 1)]
            
            # Try removing a (if a != b)
            if a != b:
                freq[a] = 0
                new_counts = compute_counts(freq)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                if new_min >= 1:
                    if new_peak < cur_peak:
                        A_set.discard(a)
                        counts = new_counts
                        cur_peak = new_peak
                        cur_min = new_min
                        continue
                    elif new_peak == cur_peak:
                        A_set.discard(a)
                        counts = new_counts
                        cur_min = new_min
                        continue
                else:
                    freq[a] = 1
            
            # Try removing b (if a != b)
            if a != b:
                freq[b] = 0
                new_counts = compute_counts(freq)
                new_min = int(np.min(new_counts))
                new_peak = int(np.max(new_counts))
                if new_min >= 1:
                    if new_peak < cur_peak:
                        A_set.discard(b)
                        counts = new_counts
                        cur_peak = new_peak
                        cur_min = new_min
                        continue
                    elif new_peak == cur_peak:
                        A_set.discard(b)
                        counts = new_counts
                        cur_min = new_min
                        continue
                else:
                    freq[b] = 1
            
            # Try 2-swap: remove a and b, add x and y
            if a != b and random.random() < 0.2:
                x = random.randint(0, n)
                y = random.randint(0, n)
                if x not in A_set and y not in A_set and x != y and x != a and x != b and y != a and y != b:
                    freq[a] = 0
                    freq[b] = 0
                    freq[x] = 1
                    freq[y] = 1
                    new_counts = compute_counts(freq)
                    new_min = int(np.min(new_counts))
                    new_peak = int(np.max(new_counts))
                    if new_min >= 1:
                        if new_peak < cur_peak:
                            A_set.discard(a)
                            A_set.discard(b)
                            A_set.add(x)
                            A_set.add(y)
                            counts = new_counts
                            cur_peak = new_peak
                            cur_min = new_min
                            continue
                        elif new_peak == cur_peak and random.random() < 0.1:
                            A_set.discard(a)
                            A_set.discard(b)
                            A_set.add(x)
                            A_set.add(y)
                            counts = new_counts
                            cur_min = new_min
                            continue
                    else:
                        freq[a] = 1
                        freq[b] = 1
                        freq[x] = 0
                        freq[y] = 0
            
            # Try swapping a with random x
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
                        counts = new_counts
                        cur_peak = new_peak
                        cur_min = new_min
                        continue
                    elif new_peak == cur_peak and random.random() < 0.15:
                        A_set.discard(a)
                        A_set.add(x)
                        counts = new_counts
                        cur_min = new_min
                        continue
                else:
                    freq[a] = 1
                    freq[x] = 0
            
            # Try swapping b with random x
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
                            counts = new_counts
                            cur_peak = new_peak
                            cur_min = new_min
                            continue
                        elif new_peak == cur_peak and random.random() < 0.15:
                            A_set.discard(b)
                            A_set.add(x)
                            counts = new_counts
                            cur_min = new_min
                            continue
                    else:
                        freq[b] = 1
                        freq[x] = 0
            
            # Try a random move: add or remove a random element
            if random.random() < 0.3:
                if len(A_set) > 1:
                    a_rand = random.choice(list(A_set))
                    freq[a_rand] = 0
                    new_counts = compute_counts(freq)
                    new_min = int(np.min(new_counts))
                    new_peak = int(np.max(new_counts))
                    if new_min >= 1 and new_peak <= cur_peak:
                        A_set.discard(a_rand)
                        counts = new_counts
                        cur_peak = new_peak
                        cur_min = new_min
                    else:
                        freq[a_rand] = 1
                else:
                    a_rand = random.randint(0, n)
                    if a_rand not in A_set:
                        freq[a_rand] = 1
                        new_counts = compute_counts(freq)
                        new_min = int(np.min(new_counts))
                        new_peak = int(np.max(new_counts))
                        if new_min >= cur_min and new_peak <= cur_peak:
                            A_set.add(a_rand)
                            counts = new_counts
                            cur_peak = new_peak
                            cur_min = new_min
                        else:
                            freq[a_rand] = 0
        
        return sorted(A_set), cur_peak
    
    best_A = None
    best_peak = float('inf')
    
    random.seed(42)
    
    # Strategy 1: Start with structured sets
    # A = all even numbers in [0, 128]
    A_even = list(range(0, 129, 2))
    A_ls, peak_ls = local_search(A_even)
    if A_ls is not None and peak_ls < best_peak:
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A_ls:
            freq[x] = 1
        counts = compute_counts(freq)
        if int(np.min(counts)) >= 1:
            best_A = A_ls
            best_peak = peak_ls
    
    # A = all odd numbers in [0, 128]
    A_odd = list(range(1, 129, 2))
    A_ls, peak_ls = local_search(A_odd)
    if A_ls is not None and peak_ls < best_peak:
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A_ls:
            freq[x] = 1
        counts = compute_counts(freq)
        if int(np.min(counts)) >= 1:
            best_A = A_ls
            best_peak = peak_ls
    
    # A = [64, 128]
    A_mid = [64, 128]
    A_ls, peak_ls = local_search(A_mid)
    if A_ls is not None and peak_ls < best_peak:
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A_ls:
            freq[x] = 1
        counts = compute_counts(freq)
        if int(np.min(counts)) >= 1:
            best_A = A_ls
            best_peak = peak_ls
    
    # A = [0, 128]
    A_ends = [0, 128]
    A_ls, peak_ls = local_search(A_ends)
    if A_ls is not None and peak_ls < best_peak:
        freq = np.zeros(n + 1, dtype=np.int32)
        for x in A_ls:
            freq[x] = 1
        counts = compute_counts(freq)
        if int(np.min(counts)) >= 1:
            best_A = A_ls
            best_peak = peak_ls
    
    # Strategy 2: Random starts with various sizes
    for k in [10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48]:
        if time.time() > deadline - 8:
            break
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            A_rand = sorted(random.sample(range(n + 1), k))
            A_ls, peak_ls = local_search(A_rand)
            if A_ls is not None and peak_ls < best_peak:
                freq = np.zeros(n + 1, dtype=np.int32)
                for x in A_ls:
                    freq[x] = 1
                counts = compute_counts(freq)
                if int(np.min(counts)) >= 1:
                    best_A = A_ls
                    best_peak = peak_ls
    
    # Strategy 3: Multiple refinement passes on best solution
    if best_A is not None and time.time() < deadline - 10:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            A_ls, peak_ls = local_search(best_A)
            if A_ls is not None and peak_ls < best_peak:
                freq = np.zeros(n + 1, dtype=np.int32)
                for x in A_ls:
                    freq[x] = 1
                counts = compute_counts(freq)
                if int(np.min(counts)) >= 1:
                    best_A = A_ls
                    best_peak = peak_ls
            else:
                break
    
    # Fallback
    if best_A is None:
        best_A = list(range(64, 129))
    
    return list(best_A)
