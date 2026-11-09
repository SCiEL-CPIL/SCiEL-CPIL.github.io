import time
import numpy as np
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline to stay under 90s limit

    n = 30
    # LCM of 1..30
    L = 1
    for i in range(1, 31):
        L = L * i // gcd(L, i)
    
    W = [L // (i+1) for i in range(n)]
    mid = n // 2  # 15
    W1 = W[:mid]
    W2 = W[mid:]
    
    def time_left():
        return deadline - time.time()
    
    # Generate all 3^15 (sum, coeffs) pairs for a half
    # Store as list of (sum, coeff_tuple) to allow recovery
    def generate_half_with_coeffs(weights):
        m = len(weights)
        # Start with [(0, (0,)*m)]
        # For memory efficiency, we'll store sums in a numpy array and coeffs in a parallel structure
        # But 3^15 = 14348907, which is large. Let's use a different approach.
        # We'll generate iteratively and store in lists.
        
        sums = [0]
        coeffs = [()]
        
        for idx, w in enumerate(weights):
            new_sums = []
            new_coeffs = []
            for s, c in zip(sums, coeffs):
                # c = -1
                new_sums.append(s - w)
                new_coeffs.append(c + (-1,))
                # c = 0
                new_sums.append(s)
                new_coeffs.append(c + (0,))
                # c = 1
                new_sums.append(s + w)
                new_coeffs.append(c + (1,))
            
            if time_left() < 25:
                # Too slow, truncate
                break
            
            sums = new_sums
            coeffs = new_coeffs
        
        return sums, coeffs
    
    # Alternative: use numpy for sums only, then reconstruct via target search
    # But the DFS backtrack was the bottleneck. Let's try storing coefficients as a flat array.
    
    def generate_half_numpy(weights):
        m = len(weights)
        sums = np.array([0], dtype=np.int64)
        # Store coefficient choices as base-3 digits: 0->-1, 1->0, 2->1
        # We'll store the index in the generation tree
        # Actually, let's just store sums and use a smarter backtrack
        for w in weights:
            new_sums = np.empty(len(sums) * 3, dtype=np.int64)
            new_sums[0::3] = sums - w
            new_sums[1::3] = sums
            new_sums[2::3] = sums + w
            sums = new_sums
            if time_left() < 30:
                break
        return sums
    
    # Use numpy generation for speed, then backtrack with memoization
    sums1 = generate_half_numpy(W1)
    if time_left() < 20:
        return [0] * 28 + [1, -1]
    
    sums2 = generate_half_numpy(W2)
    if time_left() < 10:
        return [0] * 28 + [1, -1]
    
    # Sort both arrays with original indices
    order1 = np.argsort(sums1)
    sums1_sorted = sums1[order1]
    order2 = np.argsort(sums2)
    sums2_sorted = sums2[order2]
    
    # Two-pointer scan to find min |s1 + s2| where s1 + s2 != 0
    n1 = len(sums1_sorted)
    n2 = len(sums2_sorted)
    i = 0
    j = n2 - 1
    best_val = float('inf')
    best_i = -1
    best_j = -1
    
    while i < n1 and j >= 0:
        s = sums1_sorted[i] + sums2_sorted[j]
        if s != 0:
            abs_s = abs(s)
            if abs_s < best_val:
                best_val = abs_s
                best_i = i
                best_j = j
                if best_val == 1:
                    break
        if s > 0:
            j -= 1
        elif s < 0:
            i += 1
        else:
            j -= 1
    
    if best_val == float('inf'):
        return [0] * 28 + [1, -1]
    
    # Now backtrack to find coefficients
    # We need to find which original index in sums1 gives sums1_sorted[best_i]
    # and which original index in sums2 gives sums2_sorted[best_j]
    orig_idx1 = order1[best_i]
    orig_idx2 = order2[best_j]
    
    # Convert original index to base-3 representation to get coefficients
    # The generation order: for each weight, we append -1, 0, +1 variants
    # So index in the flat array maps to a sequence of choices
    def index_to_coeffs(idx, m):
        """Convert flat index to coefficient tuple for m weights"""
        coeffs = [0] * m
        for pos in range(m):
            # At each level, the index is divided by 3^(m-1-pos)
            # The remainder gives the choice at this position
            # Actually, let's think about it differently.
            # After processing k weights, we have 3^k entries.
            # Entry idx at level k corresponds to choice = idx % 3, and parent = idx // 3
            # But we build forward, so:
            # Level 0: [0]
            # Level 1: [0-w, 0, 0+w] -> indices 0,1,2 map to choices -1,0,1
            # Level 2: for each of 3, we add 3 more
            # So the index encodes the sequence of choices in base 3, with most significant first
            # Let's verify: after m weights, total = 3^m
            # Index = c0 * 3^(m-1) + c1 * 3^(m-2) + ... + c_{m-1} * 3^0
            # where c_i in {0,1,2} maps to {-1,0,1}
            pass
        
        # Better: decode from the generation process
        # At each step, new_sums[0::3] = old - w, new_sums[1::3] = old, new_sums[2::3] = old + w
        # So for a given index in the final array, we can trace back
        coeffs = [0] * m
        for pos in range(m - 1, -1, -1):
            choice = idx % 3
            idx //= 3
            coeffs[pos] = choice - 1  # 0->-1, 1->0, 2->1
        return coeffs
    
    coeffs1 = index_to_coeffs(int(orig_idx1), len(W1))
    coeffs2 = index_to_coeffs(int(orig_idx2), len(W2))
    
    coeffs = coeffs1 + coeffs2
    
    # Verify the sum
    total = 0
    for c, w in zip(coeffs, W):
        total += c * w
    
    # If total is 0, we need to find a different pair
    if total == 0:
        # Try nearby pairs
        found = False
        for di in range(-5, 6):
            for dj in range(-5, 6):
                ni = best_i + di
                nj = best_j + dj
                if 0 <= ni < n1 and 0 <= nj < n2:
                    s = sums1_sorted[ni] + sums2_sorted[nj]
                    if s != 0:
                        oi1 = order1[ni]
                        oi2 = order2[nj]
                        c1 = index_to_coeffs(int(oi1), len(W1))
                        c2 = index_to_coeffs(int(oi2), len(W2))
                        c_all = c1 + c2
                        t = sum(c * w for c, w in zip(c_all, W))
                        if t != 0 and abs(t) < abs(total) if total != 0 else True:
                            coeffs = c_all
                            found = True
                            break
            if found:
                break
    
    # Local search: try flipping individual coefficients to reduce |sum|
    if time_left() > 5:
        current_sum = sum(c * w for c, w in zip(coeffs, W))
        for _ in range(3):  # A few rounds of local search
            improved = False
            for pos in range(n):
                for delta in [-1, 1]:
                    if coeffs[pos] + delta in (-1, 0, 1):
                        new_coeffs = coeffs[:]
                        new_coeffs[pos] += delta
                        new_sum = current_sum + delta * W[pos]
                        if new_sum != 0 and abs(new_sum) < abs(current_sum):
                            coeffs = new_coeffs
                            current_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
    
    return coeffs
