import time
import random
from itertools import product

N = 12
SIZE = 1 << N

def compute_reset_length(trans_a, trans_b):
    """
    Computes the exact shortest reset word length for a deterministic binary automaton.
    Returns -1 if not synchronizing.
    Uses BFS on the subset automaton represented by bitmasks.
    """
    size = SIZE
    # Precompute transitions for all subsets
    trans_a_sub = [0] * size
    trans_b_sub = [0] * size
    
    for subset in range(size):
        if subset == 0:
            continue
        sa = 0
        sb = 0
        temp = subset
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            sa |= (1 << trans_a[i])
            sb |= (1 << trans_b[i])
            temp ^= lsb
        trans_a_sub[subset] = sa
        trans_b_sub[subset] = sb

    start = (1 << N) - 1
    if start == 1:
        return 0
        
    visited = [False] * size
    queue = [start]
    visited[start] = True
    dist = 0
    
    while queue:
        new_queue = []
        for subset in queue:
            # Check if singleton
            if subset & (subset - 1) == 0:
                return dist
            
            next_a = trans_a_sub[subset]
            if not visited[next_a]:
                visited[next_a] = True
                new_queue.append(next_a)
            
            next_b = trans_b_sub[subset]
            if not visited[next_b]:
                visited[next_b] = True
                new_queue.append(next_b)
        
        queue = new_queue
        dist += 1
        
    return -1

def construct():
    deadline = time.time() + 55  # Safety margin
    
    best_a = [0] * N
    best_b = [0] * N
    best_len = -1
    
    # Strategy: Exhaustive search over the "Pinwheel" family.
    # The Pinwheel automaton is a standard construction for high reset thresholds.
    # Structure:
    #   Letter A: A cycle. Usually (i+1) % N or (i-1) % N.
    #   Letter B: A "step-down" map with a break.
    #     Typically: b[i] = (i-1) % N for i != break, and b[break] = 0 (or similar).
    #     Or more generally: b[i] = i - 1 for i > 0, b[0] = 0 is the "worst case" for N=2? No, that's not it.
    #     The standard Pinwheel for N states often has:
    #       a: i -> (i+1) % N
    #       b: i -> i-1 for i>0, b[0] = 0 ? No, that's not synchronizing efficiently.
    #     
    #     Let's look at the known optimal for N=12.
    #     The reset threshold for N=12 is known to be 112? No, the current score is 112.
    #     Wait, the current score is 112.0. The theoretical max for N=12 is likely around 112-120.
    #     Actually, for N=12, the max reset length is 112? 
    #     Let's check the Pinwheel construction:
    #     a = (0 1 2 ... N-1)
    #     b = (0 0 1 2 ... N-2) ? 
    #     Let's generate a broad set of Pinwheel-like variants.
    
    # Family 1: a is a cycle, b is a "mostly decrement" with one break.
    # a[i] = (i + 1) % N
    # b[i] = (i - 1) % N for i != k, b[k] = target
    
    # We will iterate over:
    # 1. Direction of a (forward or backward)
    # 2. Structure of b:
    #    - b is mostly (i-1) mod N, with one state k mapped to some target t.
    #    - b is mostly (i+1) mod N, with one state k mapped to some target t.
    #    - b is a "shift" with a break.
    
    # Let's be more systematic.
    # The "Pinwheel" automaton P_n is defined by:
    #   x: (0 1 ... n-1)
    #   y: (0 0 1 ... n-2)  -- this is not the standard one.
    
    # A known high-threshold family is:
    #   a: i -> (i+1) % N
    #   b: i -> i-1 if i>0 else 0  (This is a "star" like structure? No, it's a path to 0).
    #   Actually, the "Pinwheel" often refers to:
    #   a: (0 1 ... N-1)
    #   b: (0 1 ... N-2 0)  -- i.e., b[i] = i-1 for i>0, b[0]=0? No, b[N-1]=0.
    
    # Let's enumerate a broader class:
    # a is a cycle.
    # b is a function that is "almost" a cycle or "almost" a constant map, but specifically designed to slow down synchronization.
    
    # Key insight: The optimal automata for small N are often "Pinwheel" or "Rank-2" variants.
    # We will generate all automata where:
    # 1. a is a single N-cycle.
    # 2. b is a function where b(i) = i-1 (mod N) for all i except one index k, where b(k) = t.
    #    Also consider b(i) = i+1 (mod N) for all i except one index k.
    #    Also consider b(i) = i (identity) except one index? No, identity doesn't help sync.
    
    # Let's also consider "Two-break" variants if time permits.
    
    candidates = []
    
    # Generate all N-cycles for 'a'
    # Since the automaton is relabel-invariant in terms of reset length, we can fix a to be the standard cycle (0->1->...->N-1->0)
    # and vary b. However, the relative position of the break in b matters.
    # So, fix a = [1, 2, ..., N-1, 0].
    
    a_fixed = [(i + 1) % N for i in range(N)]
    
    # Case 1: b is "decrement" with one break
    # b[i] = (i-1) % N for i != k, b[k] = t
    for k in range(N):
        for t in range(N):
            b = [(i - 1) % N for i in range(N)]
            b[k] = t
            candidates.append((a_fixed[:], b))
            
    # Case 2: b is "increment" with one break
    # b[i] = (i+1) % N for i != k, b[k] = t
    for k in range(N):
        for t in range(N):
            b = [(i + 1) % N for i in range(N)]
            b[k] = t
            candidates.append((a_fixed[:], b))
            
    # Case 3: b is "decrement" with two breaks
    # b[i] = (i-1) % N for i not in {k1, k2}, b[k1]=t1, b[k2]=t2
    # This is N*N*N*N = 20736 candidates. Manageable.
    for k1 in range(N):
        for t1 in range(N):
            for k2 in range(k1+1, N):
                for t2 in range(N):
                    b = [(i - 1) % N for i in range(N)]
                    b[k1] = t1
                    b[k2] = t2
                    candidates.append((a_fixed[:], b))
                    
    # Case 4: b is "increment" with two breaks
    for k1 in range(N):
        for t1 in range(N):
            for k2 in range(k1+1, N):
                for t2 in range(N):
                    b = [(i + 1) % N for i in range(N)]
                    b[k1] = t1
                    b[k2] = t2
                    candidates.append((a_fixed[:], b))
                    
    # Case 5: a is "decrement" cycle, b is "decrement" with one break
    a_dec = [(i - 1) % N for i in range(N)]
    for k in range(N):
        for t in range(N):
            b = [(i - 1) % N for i in range(N)]
            b[k] = t
            candidates.append((a_dec[:], b))
            
    # Case 6: a is "decrement" cycle, b is "increment" with one break
    for k in range(N):
        for t in range(N):
            b = [(i + 1) % N for i in range(N)]
            b[k] = t
            candidates.append((a_dec[:], b))

    # Deduplicate candidates (optional, but good for speed)
    # We'll just evaluate them. 20k candidates * BFS is fast enough?
    # BFS takes ~0.1ms? 20k * 0.1ms = 2s. Yes.
    
    # To be safe, we'll prioritize the "one break" cases first, then "two break" if time remains.
    
    # Sort candidates by "complexity" or just iterate.
    # Let's iterate through them.
    
    for a, b in candidates:
        if time.time() > deadline:
            break
            
        # Quick check: Does it synchronize?
        # We can do a fast check or just run the BFS.
        # The BFS returns -1 if not synchronizing.
        
        l = compute_reset_length(a, b)
        
        if l > best_len:
            best_len = l
            best_a = a[:]
            best_b = b[:]
            
    # If we haven't found anything (shouldn't happen), fallback to a known good one
    if best_len < 0:
        a_std = [(i + 1) % N for i in range(N)]
        b_std = [0] * N
        for i in range(1, N):
            b_std[i] = i - 1
        best_a = a_std
        best_b = b_std
        best_len = compute_reset_length(best_a, best_b)
        
    return [best_a, best_b]
