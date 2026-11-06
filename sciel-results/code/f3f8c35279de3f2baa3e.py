import time
import bisect
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    weights = [L // d for d in denominators]

    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    n1 = len(w1)
    n2 = len(w2)

    def gen_sums(weights_list, n_items):
        """
        Generate all subset sums < L with their masks.
        Returns a list of (sum, mask) tuples.
        """
        sums = [(0, 0)]
        for i in range(n_items):
            if time.time() > deadline:
                break
            w = weights_list[i]
            bit = 1 << i
            current_len = len(sums)
            # Local variable for speed
            append = sums.append
            for j in range(current_len):
                s, m = sums[j]
                ns = s + w
                if ns < L:
                    append((ns, m | bit))
        return sums

    # Generate sums for first half
    sums1 = gen_sums(w1, n1)
    if time.time() > deadline:
        # Fallback: 1/2 + 1/3 + 1/7 + 1/43
        # Indices: 0, 1, 5, 41
        return [0, 1, 5, 41]
        
    # Generate sums for second half
    sums2 = gen_sums(w2, n2)
    if time.time() > deadline:
        return [0, 1, 5, 41]

    # Sort sums1 by value
    sums1.sort(key=lambda x: x[0])
    sums1_vals = [s[0] for s in sums1]

    best_sum = -1
    best_m1 = 0
    best_m2 = 0

    # Iterate through sums2
    # To optimize, we can sort sums2 in descending order and break early if possible?
    # No, because a small s2 might pair with a large s1 to make a better total.
    # However, if we sort sums2 descending, the first valid pair might be good, but not necessarily optimal.
    # We must check all.
    
    # One optimization: if s2 is already close to L, the target for s1 is small.
    # If s2 is small, target is large.
    
    for s2, m2 in sums2:
        if time.time() > deadline:
            break
        # We need s1 such that s1 + s2 < L
        # s1 < L - s2
        # Max s1 is the largest value in sums1_vals < L - s2
        target = L - 1 - s2
        if target < 0:
            continue
            
        idx = bisect.bisect_right(sums1_vals, target)
        if idx > 0:
            s1, m1 = sums1[idx - 1]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_m1 = m1
                best_m2 = m2

    if best_sum > 0:
        # Reconstruct indices
        indices = []
        for i in range(n1):
            if best_m1 & (1 << i):
                indices.append(i)
        for i in range(n2):
            if best_m2 & (1 << i):
                indices.append(mid + i)
        if indices:
            # Verify
            val = sum(weights[i] for i in indices)
            if val < L:
                # Local search polish: try to improve by swapping or adding/removing elements
                current_sum = val
                current_indices = set(indices)
                improved = True
                max_polish_time = start_time + 58.0
                
                while improved and time.time() < max_polish_time:
                    improved = False
                    # Try adding each non-selected element
                    for i in range(n):
                        if time.time() > max_polish_time:
                            break
                        if i not in current_indices:
                            new_sum = current_sum + weights[i]
                            if new_sum < L:
                                current_sum = new_sum
                                current_indices.add(i)
                                improved = True
                                break
                    # Try removing one element and adding another (swap)
                    if not improved:
                        for i in list(current_indices):
                            if time.time() > max_polish_time:
                                break
                            for j in range(n):
                                if time.time() > max_polish_time:
                                    break
                                if j not in current_indices:
                                    new_sum = current_sum - weights[i] + weights[j]
                                    if new_sum < L and new_sum > current_sum:
                                        current_sum = new_sum
                                        current_indices.remove(i)
                                        current_indices.add(j)
                                        improved = True
                                        break
                            if improved:
                                break
                    # Try removing one element and adding two others
                    if not improved:
                        for i in list(current_indices):
                            if time.time() > max_polish_time:
                                break
                            for j in range(n):
                                if time.time() > max_polish_time:
                                    break
                                if j not in current_indices:
                                    for k in range(n):
                                        if time.time() > max_polish_time:
                                            break
                                        if k not in current_indices and k != j:
                                            new_sum = current_sum - weights[i] + weights[j] + weights[k]
                                            if new_sum < L and new_sum > current_sum:
                                                current_sum = new_sum
                                                current_indices.remove(i)
                                                current_indices.add(j)
                                                current_indices.add(k)
                                                improved = True
                                                break
                                        if improved:
                                            break
                                if improved:
                                    break
                            if improved:
                                break
                    # Try removing two elements and adding three others
                    if not improved:
                        idx_list = list(current_indices)
                        for a in range(len(idx_list)):
                            if time.time() > max_polish_time:
                                break
                            for b in range(a+1, len(idx_list)):
                                if time.time() > max_polish_time:
                                    break
                                i1 = idx_list[a]
                                i2 = idx_list[b]
                                for j in range(n):
                                    if time.time() > max_polish_time:
                                        break
                                    if j not in current_indices:
                                        for k in range(n):
                                            if time.time() > max_polish_time:
                                                break
                                            if k not in current_indices and k != j:
                                                for m in range(n):
                                                    if time.time() > max_polish_time:
                                                        break
                                                    if m not in current_indices and m != j and m != k:
                                                        new_sum = current_sum - weights[i1] - weights[i2] + weights[j] + weights[k] + weights[m]
                                                        if new_sum < L and new_sum > current_sum:
                                                            current_sum = new_sum
                                                            current_indices.remove(i1)
                                                            current_indices.remove(i2)
                                                            current_indices.add(j)
                                                            current_indices.add(k)
                                                            current_indices.add(m)
                                                            improved = True
                                                            break
                                                    if improved:
                                                        break
                                            if improved:
                                                break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                
                final_indices = sorted(current_indices)
                if final_indices:
                    return final_indices
                else:
                    return [0, 1, 5, 41]
            else:
                return [0, 1, 5, 41]
    else:
        return [0, 1, 5, 41]
