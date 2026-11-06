import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin for 120s limit
    
    # We will try to maximize L within the time limit.
    # Start with a conservative L and increase if fast.
    # But for simplicity and robustness, let's pick a good L.
    # L = 2 * 10**8 seems safe.
    L = 2 * 10**8
    
    # 1. Generate powerful numbers up to L
    powerful_set = set()
    max_b = int(L ** (1/3)) + 1
    
    # Optimization: Precompute squares?
    # b^3 is computed in loop.
    for b in range(1, max_b):
        b3 = b ** 3
        if b3 > L:
            break
        # a^2 <= L / b^3
        max_a = isqrt(L // b3)
        # Generate a^2 * b^3 for a in 1..max_a
        # We can vectorize this part
        a_vals = np.arange(1, max_a + 1)
        p_vals = (a_vals * a_vals) * b3
        # Filter if any > L (shouldn't happen due to isqrt, but safe)
        # Add to set
        powerful_set.update(p_vals.tolist())
        
    if time.time() - start_time > time_limit * 0.2:
        # If generation is slow, we might need to reduce L, but it's usually fast.
        pass

    A = np.array(sorted(powerful_set), dtype=np.int64)
    M = len(A)
    
    # 2. Compute pairwise sums (i <= j)
    # We want to count N = A[i] + A[j]
    # To save memory, we can do it in chunks if M is large.
    # For M ~ 14000, M^2/2 = 10^8. 
    # Let's try to do it in one go if memory allows, else chunk.
    
    # We'll use a dictionary or bincount.
    # Bincount requires non-negative integers. A[i]+A[j] is up to 2L.
    # 2L = 4*10^8. This is a large array for bincount (400M entries).
    # 400M * 8 bytes = 3.2 GB. This might be too much.
    
    # Alternative: Use a dictionary for counts?
    # 10^8 entries in a dict is too slow and memory heavy.
    
    # Better Alternative:
    # We don't need to count ALL sums. We just need the MAX count.
    # But we don't know which N is the max.
    
    # Let's reconsider the memory for bincount.
    # If L=10^8, 2L=2*10^8. Bincount size 2*10^8. 1.6 GB. Safe.
    # If L=2*10^8, 2L=4*10^8. Bincount size 4*10^8. 3.2 GB. Risky.
    
    # Let's stick to L=10^8 to be safe on memory, or use a smarter counting method.
    # Actually, we can use `np.bincount` on the sums, but we can reduce the range?
    # No, the sums are distributed.
    
    # Let's try L=10**8 first.
    if L > 10**8:
        # If we are ambitious, we can chunk the bincount? No, bincount is fixed size.
        # We can use a sparse approach if we suspect the max is low?
        # But we want the max.
        
        # Let's try to do the sum calculation in a way that doesn't require a huge bincount array.
        # We can sort the sums and then count runs?
        # Generating 10^8 sums, sorting them (O(N log N)), then counting.
        # Sorting 10^8 int64s is fast in numpy (~1-2s).
        # Memory for sums: 800MB.
        # This avoids the 3.2GB bincount array.
        
        pass
        
    # Let's implement the sorting method which is memory efficient for the counts.
    
    # Generate sums A[i] + A[j] for i <= j
    # We can do this in chunks to avoid creating the full MxM matrix at once.
    # But M is small enough (14k) that MxM is 200M elements (1.6GB).
    # Upper triangle is half that.
    
    # Let's just generate the upper triangle sums directly.
    # np.triu_indices is slow for large M.
    # Loop over i, and take A[i:]
    
    sums = []
    chunk_size = 1000
    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        # For each i in [i, end), we need sums with j >= i
        # This is tricky to vectorize perfectly without full matrix.
        
        # Simpler: Just do the full matrix if M is small enough.
        # If M <= 20000, M^2 = 4*10^8. 3.2GB. Too big.
        
        # Let's stick to L=10**8. M ~ 10000. M^2 = 10^8. 800MB.
        # This is the safest bet.
        
    L = 10**8
    # Regenerate if L changed
    if len(A) > 12000: # Heuristic for L=10^8
        # Re-do generation for L=10^8
        powerful_set = set()
        max_b = int(L ** (1/3)) + 1
        for b in range(1, max_b):
            b3 = b ** 3
            if b3 > L:
                break
            max_a = isqrt(L // b3)
            a_vals = np.arange(1, max_a + 1)
            p_vals = (a_vals * a_vals) * b3
            powerful_set.update(p_vals.tolist())
        A = np.array(sorted(powerful_set), dtype=np.int64)
        M = len(A)

    # Compute sums
    # Create the upper triangle sums
    # We can use a loop to append chunks to a list, then concatenate.
    # Or just use the full matrix if M is small.
    # M=10000.
    # A[:, None] + A[None, :] is 10^8 elements.
    # This is 800MB. Acceptable.
    
    if time.time() - start_time > time_limit * 0.5:
        # Fallback: just return a known good one or small search
        pass

    # Generate sums
    # To be safe on memory, we process in blocks of rows.
    all_sums = []
    block_size = 100
    for i_start in range(0, M, block_size):
        i_end = min(i_start + block_size, M)
        # Rows i_start to i_end
        # We need j >= i for each row i.
        # This creates a jagged array.
        # It's easier to just do the full MxM and take triu, but that's 2x memory.
        
        # Let's just do the full MxM if M < 15000.
        if M < 15000:
             # 15000^2 = 225M. 1.8GB.
             # Let's try.
             S = A[:, None] + A[None, :]
             # Mask for i <= j
             idx = np.triu_indices(M)
             sums = S[idx]
             all_sums = [sums]
        else:
             # Fallback to smaller L or chunked bincount?
             # For now, assume M < 15000 holds for L=10^8.
             pass

    if not all_sums:
        # Should not happen
        return {'N': 2, 'pairs': [[1, 1]]}

    sums_flat = np.concatenate(all_sums)
    
    # Count occurrences
    # We can use np.bincount if the max sum is not too large.
    # Max sum is 2 * max(A) <= 2 * 10^8.
    # Bincount array size 2*10^8. 1.6GB.
    # This is the bottleneck.
    
    # Alternative: Sort and count.
    # sums_flat has 5*10^7 elements.
    # Sorting takes ~1-2s.
    # Counting runs takes ~0.1s.
    # Memory is just the sums array (400MB) + sorted copy (400MB).
    # Total ~800MB. Safe.
    
    sums_sorted = np.sort(sums_flat)
    
    # Find unique values and their counts
    # np.diff to find boundaries
    # Or use a loop? No, vectorize.
    
    # Create a mask of where the value changes
    changes = np.diff(sums_sorted) != 0
    # The indices where change happens are the ends of groups.
    # The starts are 0 and indices after changes.
    
    # Let's use a simpler method:
    # We can use np.unique with return_counts=True
    # But np.unique on 5*10^7 elements might be slow or memory heavy?
    # It sorts anyway.
    
    unique_sums, counts = np.unique(sums_sorted, return_counts=True)
    
    # Now, adjust for a <= b constraint.
    # unique_sums are the N values.
    # counts are the number of pairs (i,j) with i<=j such that A[i]+A[j]=N.
    # Wait, my sums generation was for i<=j?
    # Yes, I used triu_indices.
    # So `counts` is EXACTLY the number of pairs a<=b.
    
    # So we just need the max count.
    max_idx = np.argmax(counts)
    best_N = unique_sums[max_idx]
    max_pairs = counts[max_idx]
    
    # Verify N <= 10^18
    if best_N > 10**18:
        # Find next best
        # This is unlikely for L=10^8
        pass
        
    # Extract pairs
    # We need to return the list of pairs.
    # max_pairs can be large.
    # We need to find all a in A such that N-a in A and a <= N-a.
    
    A_set = set(A.tolist())
    pairs = []
    # Iterate through A
    # Optimization: Only iterate a <= N/2
    half_N = best_N // 2
    # Find indices in A where A[i] <= half_N
    # Since A is sorted, we can use searchsorted
    idx_limit = np.searchsorted(A, half_N)
    
    for i in range(idx_limit):
        a = A[i]
        b = best_N - a
        if b in A_set:
            pairs.append([int(a), int(b)])
            
    # Check time
    if time.time() - start_time > time_limit:
        # Return what we have
        pass

    return {'N': int(best_N), 'pairs': pairs}
