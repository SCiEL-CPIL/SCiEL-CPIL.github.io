import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    n = 128
    target_min = 128
    target_max = 256
    target_size = target_max - target_min + 1
    
    best_peak = float('inf')
    best_A = None

    def get_counts(A):
        counts = np.zeros(target_size, dtype=int)
        A_list = sorted(A)
        for i in range(len(A_list)):
            for j in range(i, len(A_list)):
                s = A_list[i] + A_list[j]
                if target_min <= s <= target_max:
                    counts[s - target_min] += 1
        return counts

    def is_valid(counts):
        return np.min(counts) >= 1

    def search_for_k(k):
        nonlocal best_peak, best_A
        if time.time() > deadline:
            return
        
        max_restarts = 10
        for r in range(max_restarts):
            if time.time() > deadline:
                break
            
            A = set(random.sample(range(n + 1), k))
            counts = get_counts(A)
            
            if not is_valid(counts):
                continue
                
            current_peak = np.max(counts)
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = set(A)
                
            # Local Search with incremental updates
            A_list = list(A)
            for step in range(10000):
                if time.time() > deadline:
                    break
                    
                u_idx = random.randint(0, k - 1)
                u = A_list[u_idx]
                
                # Pick v not in A
                # Efficiently pick v
                while True:
                    v = random.randint(0, n)
                    if v not in A:
                        break
                        
                # Simulate swap u -> v
                # Remove u, Add v
                
                # Check validity and peak change
                # We need to know new counts.
                # Let's apply changes to a copy of counts? 
                # Or track delta?
                # Since k is small, copying counts (size 129) is fast.
                
                new_counts = counts.copy()
                
                # Remove pairs with u
                for w in A_list:
                    if w == u:
                        s = 2 * u
                        if target_min <= s <= target_max:
                            new_counts[s - target_min] -= 1
                    else:
                        s = u + w
                        if target_min <= s <= target_max:
                            new_counts[s - target_min] -= 1
                            
                # Add pairs with v
                # Note: w iterates over A_list which still contains u.
                # But in the new set, u is gone.
                # So for w in A_list:
                #   if w == u: skip (u is not in new set)
                #   else: pair (v, w) exists in new set.
                # Also pair (v, v) exists.
                
                for w in A_list:
                    if w == u:
                        continue
                    s = v + w
                    if target_min <= s <= target_max:
                        new_counts[s - target_min] += 1
                        
                s = 2 * v
                if target_min <= s <= target_max:
                    new_counts[s - target_min] += 1

                if not is_valid(new_counts):
                    continue
                    
                new_peak = np.max(new_counts)
                
                if new_peak < current_peak:
                    # Accept
                    A.remove(u)
                    A.add(v)
                    A_list[u_idx] = v
                    counts = new_counts
                    current_peak = new_peak
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = set(A)
                elif new_peak == current_peak and random.random() < 0.3:
                    A.remove(u)
                    A.add(v)
                    A_list[u_idx] = v
                    counts = new_counts

    for k in range(20, 35):
        search_for_k(k)
        
    if best_A is None:
        return list(range(64, 129))
        
    return list(best_A)
