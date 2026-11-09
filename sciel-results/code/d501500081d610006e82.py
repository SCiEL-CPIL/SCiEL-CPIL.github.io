import time

N = 17

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Generate Initial Network: Odd-Even Transposition
    # Phases 0 to N-1
    # Phase p: comparators (i, i+1) where i % 2 == p % 2
    initial_network = []
    for p in range(N):
        for i in range(p % 2, N - 1, 2):
            initial_network.append((i, i + 1))
    
    # 2. Precompute Target Bitmasks for Verification
    # M = 2^N inputs
    M = 1 << N
    # T[w] is the bitmask of all inputs k such that in the sorted output, wire w is 1.
    # Sorted output for input k (with c ones) has 1s in wires N-c to N-1.
    # So wire w is 1 iff N-c <= w  => c >= N-w.
    # So T[w] has bit k set iff popcount(k) >= N - w.
    
    # We can compute popcount for all k in a list or use bit tricks.
    # Since M=131072, we can just iterate.
    targets = [0] * N
    for k in range(M):
        c = bin(k).count('1')
        # Wire w is 1 if w >= N - c
        # So for w from N-c to N-1, set bit k in targets[w]
        # This loop is O(N) per k, total O(N*M) = 17 * 131072 ~ 2.2M ops. Fast enough.
        # Optimization: Precompute popcounts? Or just do it.
        start_w = N - c
        for w in range(start_w, N):
            targets[w] |= (1 << k)
            
    # 3. Define Fast Verifier
    def verify(network):
        # Initialize state S[w] for each wire w
        # S[w] has bit k set iff wire w of input k is 1.
        # Input k is an integer. Wire w is bit w of k.
        # So S[w] is the bitmask of all k where bit w is 1.
        S = [0] * N
        for w in range(N):
            # Bit w is 1 for k in [2^w, 2^(w+1)-1], [3*2^w, 4*2^w-1], etc.
            # We can build this pattern.
            # Or simply:
            # S[w] = sum(1 << k for k in range(M) if (k >> w) & 1)
            # This is M/2 iterations per wire. 17 * 65536 ~ 1.1M ops.
            # Let's do it efficiently.
            mask = 0
            step = 1 << (w + 1)
            half = 1 << w
            for start in range(0, M, step):
                # Bits from start+half to start+step-1 are 1
                # Create a mask for this block
                # This is getting complex. Let's just use the simple loop for initialization once.
                pass
            
            # Simpler initialization:
            # We can compute S[w] using bit operations on the index range.
            # Actually, let's just build S once outside the verify loop if possible?
            # No, S depends on the network? No, S is the INITIAL state of inputs.
            # It is constant!
            pass
        
        # Let's precompute initial S outside verify.
        # S_init[w] has bit k set iff (k >> w) & 1 == 1.
        
        # Copy S_init to S
        S = S_init[:]
        
        # Apply comparators
        for i, j in network:
            # min goes to i, max goes to j
            # min(bit_i, bit_j) = bit_i AND bit_j
            # max(bit_i, bit_j) = bit_i OR bit_j
            si = S[i]
            sj = S[j]
            S[i] = si & sj
            S[j] = si | sj
            
        # Check against targets
        for w in range(N):
            if S[w] != targets[w]:
                return False
        return True

    # Precompute S_init
    S_init = [0] * N
    for w in range(N):
        mask = 0
        # For each block of size 2^(w+1), the second half has bit w set
        step = 1 << (w + 1)
        half = 1 << w
        for start in range(0, M, step):
            # Set bits from start+half to start+step-1
            # This range is 'half' bits long.
            # We can create a mask of 'half' ones and shift it.
            # But M is up to 2^17.
            # Let's just loop k. It's safer and fast enough for one-time init.
            pass
        
    # Let's do the simple loop for S_init, it's only 17 * 131072 = 2.2M iterations.
    for w in range(N):
        mask = 0
        for k in range(M):
            if (k >> w) & 1:
                mask |= (1 << k)
        S_init[w] = mask

    # Now verify is fast: N big-int ops per comparator.
    # Number of comparators ~136. 136 * 17 * (big-int ops) ~ 2300 big-int ops.
    # Big-int ops on 131k-bit integers are fast.
    
    # 4. Greedy Removal
    current_network = initial_network[:]
    improved = True
    while improved:
        improved = False
        i = 0
        while i < len(current_network):
            # Try removing current_network[i]
            if time.time() > deadline:
                break
            
            new_network = current_network[:i] + current_network[i+1:]
            if verify(new_network):
                # Removal successful
                current_network = new_network
                improved = True
                # Do not increment i, check next index (which is now the one after removed)
            else:
                i += 1
        
        if time.time() > deadline:
            break

    return current_network

# EVOLVE-BLOCK-END
