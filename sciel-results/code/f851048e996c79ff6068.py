import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin

    limit = 10000
    # Sieve of Eratosthenes
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            is_prime[p*p : limit+1 : p] = False
    
    primes = np.where(is_prime)[0]
    N = len(primes)
    
    if N == 0:
        return []
        
    # Map prime value to index, -1 if not prime
    p_to_idx = np.full(limit + 1, -1, dtype=np.int32)
    p_to_idx[primes] = np.arange(N)
    
    # DP tables
    # best[i, g] = max length of valid seq ending at primes[i] with last gap <= g
    # dtype uint8 is sufficient (max length is small)
    best = np.zeros((N, limit + 1), dtype=np.uint8)
    # argmax_gap[i, g] = the specific gap k <= g that achieved best[i, g]
    argmax_gap = np.zeros((N, limit + 1), dtype=np.uint16)
    # parent_idx[i, g] = index j of previous prime in the optimal path for state (i, g)
    # -1 if this is the start of the sequence (length 1)
    parent_idx = np.full((N, limit + 1), -1, dtype=np.int32)
    
    global_max_len = 1
    best_end_i = 0
    best_end_g = 0
    
    # Base case: i=0
    # Sequence of length 1 ending at p0.
    # "Last gap" is conceptually 0.
    # For any g >= 0, best length is 1.
    # We set exact[0] = 1, so accumulate will make best[:, g] >= 1.
    # But we need to initialize the tables properly.
    # Let's just run the loop for i=0 as well, or initialize manually.
    # Manual init:
    # best[0, :] = 1
    # argmax_gap[0, :] = 0
    # parent_idx[0, :] = -1
    best[0, :] = 1
    argmax_gap[0, :] = 0
    # parent_idx[0, :] is already -1
    
    for i in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
            
        pi = primes[i]
        
        # We want to compute exact[g] for g in 1..pi-2
        # exact[g] = best[j, g] + 1 if pi-g is prime
        # Initialize exact to 0
        # Note: exact[0] = 1 (base case for this prime)
        
        # Create array of potential previous primes
        # g ranges from 1 to pi - 2 (smallest prime is 2)
        # Actually, pi - g must be >= 2.
        max_g = pi - 2
        if max_g < 1:
            # No previous prime can form a gap >= 1 with pi if pi is small?
            # If pi=3, max_g=1. prev=2.
            pass
        
        # Vectorized calculation
        # g_vals = np.arange(1, max_g + 1)
        # prev_vals = pi - g_vals
        # mask = is_prime[prev_vals]
        # if np.any(mask):
        #     valid_g = g_vals[mask]
        #     valid_prev = prev_vals[mask]
        #     valid_j = p_to_idx[valid_prev]
        #     # best[valid_j, valid_g] gives the lengths
        #     lengths = best[valid_j, valid_g] + 1
        #     # Update exact array
        #     # We need to store these in a temporary array 'exact' of size limit+1
        #     # initialized to 0, with exact[0]=1
        
        exact = np.zeros(limit + 1, dtype=np.uint8)
        exact[0] = 1 # Base case: sequence of length 1
        
        if max_g >= 1:
            g_vals = np.arange(1, max_g + 1)
            prev_vals = pi - g_vals
            mask = is_prime[prev_vals]
            if np.any(mask):
                valid_g = g_vals[mask]
                valid_prev = prev_vals[mask]
                valid_j = p_to_idx[valid_prev]
                
                # Get best lengths for these specific states
                # valid_j is array of indices, valid_g is array of gaps
                # best[valid_j, valid_g] works with advanced indexing
                prev_lengths = best[valid_j, valid_g]
                new_lengths = prev_lengths + 1
                
                # Update exact
                exact[valid_g] = new_lengths
                
                # Update parent_idx for these specific gaps
                # parent_idx[i, valid_g] = valid_j
                # We will fill parent_idx[i, :] after computing argmax
                # But we need to know which j contributed to the max.
                # Since exact[g] is uniquely determined by j (as pi-g is unique),
                # the parent for exact[g] is exactly valid_j corresponding to g.
                # So we can store parent_exact[g] = j
                parent_exact = np.full(limit + 1, -1, dtype=np.int32)
                parent_exact[valid_g] = valid_j
                
                # Now compute prefix max and argmax
                # best[i, g] = max(exact[0..g])
                # We can use np.maximum.accumulate
                best_i = np.maximum.accumulate(exact)
                
                # To get argmax (index of max), we need to be careful.
                # If there are ties, which one to pick? Any is fine for reconstruction as long as consistent.
                # np.argmax returns the first occurrence.
                # But we need the index relative to the array 'exact'.
                # We can compute cumulative max and then find where it increased?
                # Or simply:
                # argmax[i, g] = the index k <= g such that exact[k] == best_i[g]
                
                # Efficient way to get argmax for prefix max:
                # Iterate? No, vectorize.
                # Since exact is sparse (mostly 0), and best_i is non-decreasing.
                # We can just loop g? 10000 iterations per i. 1229 * 10000 = 12M.
                # Python loop is slow.
                # Let's try to vectorize argmax.
                
                # Trick: 
                # We want indices of the max values in the prefix.
                # Let M = best_i.
                # We can create an array of indices 0..limit.
                # idxs = np.arange(limit + 1)
                # We want for each g, the last index k <= g where exact[k] == M[g]? 
                # No, the first index is fine.
                
                # Actually, we only need to store the parent for the *chosen* argmax.
                # Let's just compute argmax_gap[i, :] in a loop if it's faster than complex vectorization, 
                # or use a simple trick.
                
                # Since N is small and limit is 10000, a Python loop for argmax might be acceptable 
                # if the rest is vectorized. 12M iterations in Python is ~1-2 seconds.
                # Total time limit 90s. This is safe.
                
                # However, we can do better.
                # We know exact[valid_g] are the non-zero (mostly) values.
                # We can just fill best_i and argmax_i.
                
                best_i = np.maximum.accumulate(exact)
                argmax_i = np.zeros(limit + 1, dtype=np.uint16)
                parent_i = np.full(limit + 1, -1, dtype=np.int32)
                
                # Loop to find argmax and parent
                # This is O(limit) per i.
                current_max = 0
                current_arg = 0
                current_parent = -1
                for g in range(limit + 1):
                    if exact[g] > current_max:
                        current_max = exact[g]
                        current_arg = g
                        current_parent = parent_exact[g]
                    argmax_i[g] = current_arg
                    parent_i[g] = current_parent
                    # best_i[g] is already computed by accumulate, but we can verify
                    # best_i[g] should equal current_max
                
                best[i, :] = best_i
                argmax_gap[i, :] = argmax_i
                parent_idx[i, :] = parent_i
                
                # Update global max
                if best_i[-1] > global_max_len:
                    global_max_len = best_i[-1]
                    best_end_i = i
                    best_end_g = argmax_i[-1] # The gap that achieved the max for this i
                    # Note: best_end_g should be the argmax for the max value.
                    # argmax_i[-1] is the argmax for the last element, which is the global max for this i.
        
    # Reconstruct
    if global_max_len == 1:
        return [int(primes[0])]
        
    seq = []
    cur_i = best_end_i
    cur_g = best_end_g
    
    while cur_i >= 0:
        seq.append(int(primes[cur_i]))
        if cur_i == 0:
            break
        next_i = parent_idx[cur_i, cur_g]
        if next_i == -1:
            break
        # The next state's gap is the argmax for the parent at the current gap?
        # No.
        # State (i, g) came from (j, g_prev).
        # We stored parent_idx[i, g] = j.
        # But which gap g_prev did j have?
        # It was the gap that achieved best[j, g].
        # We stored argmax_gap[j, g] as the specific gap k such that exact[j, k] = best[j, g].
        # So the next state is (j, argmax_gap[j, g]).
        
        next_g = argmax_gap[cur_i, cur_g] # Wait.
        
        # Let's re-verify the reconstruction logic.
        # We are at state (i, g_i).
        # Value is L.
        # If L == 1, stop.
        # Else, it came from j = parent_idx[i, g_i].
        # The value at j was best[j, g_i] (because exact[i, g_i] = best[j, g_i] + 1).
        # Wait, exact[i, g] is defined for the *current* gap g.
        # The parent state is (j, g_prev) where g_prev is the gap that achieved the max at j.
        # Specifically, best[j, g] = max_{k<=g} exact[j, k].
        # Let k* be the argmax.
        # Then the path goes through (j, k*).
        # So the next state in the sequence is (j, k*).
        # k* is stored in argmax_gap[j, g].
        
        # So:
        # next_i = parent_idx[cur_i, cur_g]
        # next_g = argmax_gap[next_i, cur_g]
        
        # Let's trace:
        # cur_i, cur_g
        # next_i = parent_idx[cur_i, cur_g]
        # next_g = argmax_gap[next_i, cur_g]
        
        # Is this correct?
        # exact[cur_i, cur_g] = best[next_i, cur_g] + 1.
        # best[next_i, cur_g] is the max length ending at next_i with last gap <= cur_g.
        # This max is achieved at gap next_g = argmax_gap[next_i, cur_g].
        # So the previous prime is next_i, and its "effective" last gap in the optimal chain is next_g.
        # So yes, next state is (next_i, next_g).
        
        cur_i = next_i
        cur_g = next_g
        
    seq.reverse()
    return seq
