import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    # We try to add numbers from 512 down to 1
    # This heuristic often works better for mean-free sets
    candidates = list(range(512, 0, -1))
    
    S = []
    bits_S = [1]  # bits_S[0] = 1 (sum 0 with 0 elements)
    max_sum = 0
    
    for x in candidates:
        if time.time() > deadline:
            break
            
        # Tentatively add x
        S_new = S + [x]
        n = len(S_new)
        
        # Compute bits for S_new
        # bits[k] is a bitmask of sums achievable with k elements
        bits_new = [0] * (n + 1)
        bits_new[0] = 1
        for i in range(n):
            val = S_new[i]
            # Update in reverse order to avoid using x multiple times
            for k in range(i, -1, -1):
                if bits_new[k]:
                    bits_new[k+1] |= (bits_new[k] << val)
        
        # Check validity
        valid = True
        # For each element a in S_new, check if it's the mean of a subset of others
        for a in S_new:
            # Create set of others
            others = [y for y in S_new if y != a]
            # Compute bits for others
            bits_others = [0] * (n) # Max size n-1
            bits_others[0] = 1
            for j in range(len(others)):
                val = others[j]
                for k in range(j, -1, -1):
                    if bits_others[k]:
                        bits_others[k+1] |= (bits_others[k] << val)
            
            # Check all k >= 2
            for k in range(2, n):
                target = k * a
                # If target is too large to be represented in bits_others[k], skip
                # Max sum is roughly sum(others)
                # We just check the bit
                if (bits_others[k] >> target) & 1:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S = S_new
            bits_S = bits_new
            # Update max_sum if needed, but not strictly necessary for bit check
    
    return S
