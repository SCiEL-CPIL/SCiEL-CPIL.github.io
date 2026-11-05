import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    MAX_N = 1000000
    # Smooth parts for n+1 to n+24, so we need indices up to MAX_N + 24
    # We will compute smooth parts for 0 to MAX_N + 24
    max_val = MAX_N + 24
    smooth = np.ones(max_val + 1, dtype=np.int64)
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    for p in primes:
        power = p
        while power <= max_val:
            smooth[power::power] *= p
            power *= p
            
    # We are interested in n in [1000, 1000000]
    # Window size K = 24
    # For a start index n, the window is smooth[n+1 : n+25]
    # Let's shift indices to make it 0-based for the window content.
    # Let A = smooth[1 : MAX_N + 25]
    # Then window for n is A[n : n+24]
    A = smooth[1 : MAX_N + 25]
    
    min_distinct = 25
    best_n = 1000
    
    # We scan n from 1000 to MAX_N
    # To avoid huge memory usage, we process in chunks
    chunk_size = 50000
    start_idx = 1000
    end_idx = MAX_N + 1
    
    while start_idx < end_idx:
        if time.time() > deadline:
            break
            
        current_end = min(start_idx + chunk_size, end_idx)
        # n range: [start_idx, current_end)
        num_n = current_end - start_idx
        
        # We need A[start_idx : current_end + 24]
        # Let's extract the relevant slice of A
        # A_slice = A[start_idx : current_end + 24]
        # We want to count distincts in windows of size 24 starting at relative offset 0 to num_n-1
        
        # Vectorized distinct count for this chunk
        # redundant_counts = zeros(num_n)
        redundant_counts = np.zeros(num_n, dtype=np.int32)
        
        # For each offset i in 1..23
        for i in range(1, 24):
            # X = A[start_idx + i : current_end + i]  (values at offset i for each n)
            # Y = A[start_idx : current_end + i] ... wait.
            # For n in [start_idx, current_end), the window is A[n:n+24].
            # Element at offset i is A[n+i].
            # Previous elements are A[n], ..., A[n+i-1].
            
            # X corresponds to A[start_idx+i : current_end+i]
            X = A[start_idx + i : current_end + i]
            
            # Y is a matrix of shape (num_n, i)
            # Y[k, j] = A[start_idx + k + j]
            # We can create Y by stacking slices
            # slices = [A[start_idx + j : current_end + j] for j in range(i)]
            # Y = np.column_stack(slices)
            
            # To save memory, we can do the comparison in a loop over j?
            # No, vectorized is better.
            # Let's try column_stack
            if i == 1:
                Y = A[start_idx : current_end]
                matches = (X == Y)
                redundant_counts += matches.astype(np.int32)
            else:
                # Create Y
                # This creates a temporary array of size num_n * i
                # For chunk_size=50000 and i=23, size is ~1.1M, which is small.
                slices = [A[start_idx + j : current_end + j] for j in range(i)]
                Y = np.column_stack(slices)
                matches = (X[:, None] == Y)
                redundant_counts += matches.any(axis=1).astype(np.int32)
        
        distinct_counts = 24 - redundant_counts
        min_in_chunk = np.min(distinct_counts)
        
        if min_in_chunk < min_distinct:
            min_distinct = min_in_chunk
            # Find the index of the first occurrence of this min
            idx = np.argmin(distinct_counts)
            best_n = int(start_idx + idx)
            
        start_idx = current_end
        
    return int(best_n)
