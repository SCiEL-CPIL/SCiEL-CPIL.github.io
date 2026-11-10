import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    N = 128
    TARGETS_MIN = 128
    TARGETS_MAX = 256
    NUM_TARGETS = TARGETS_MAX - TARGETS_MIN + 1
    # Index mapping: target t -> index t - TARGETS_MIN
    # We only care about multiplicities in [128, 256]
    
    # Precompute valid pairs for each target to speed up delta calculations
    # pairs[t_idx] = list of (a, b) such that a+b = TARGETS_MIN + t_idx and a <= b
    # But for delta, we just need to know which elements are involved.
    # Actually, for delta, if we add x, it affects target t if t-x is in A.
    # So we need to iterate over A for each x? No, iterate over A for each t?
    
    # Let's use a boolean array for A for O(1) lookup
    # A_bool[x] = 1 if x in A
    
    best_A = None
    best_score = float('inf')
    
    # Theoretical lower bound:
    # |A| = k. Total pairs = k(k+1)/2.
    # Targets = 129.
    # Avg mult = k(k+1)/2 / 129.
    # If k=20, avg ~ 1.55. Max likely 2 or 3.
    # If k=18, avg ~ 1.27. Max likely 2.
    # If k=17, avg ~ 1.14. Max likely 2.
    # Let's try k from 15 to 25.
    
    for k in range(15, 26):
        if time.time() > deadline - 5:
            break
            
        # Try multiple initializations for this k
        for init_iter in range(5):
            if time.time() > deadline - 5:
                break
                
            # Initialize A
            # Strategy: Random sample, but maybe bias towards uniform distribution?
            # Simple random is usually fine for small N.
            A_set = set(random.sample(range(0, N + 1), k))
            
            # Check if valid (coverage)
            # Compute initial multiplicities
            # We will maintain mults array incrementally
            
            def compute_mults_from_set(A_set):
                mults = np.zeros(NUM_TARGETS, dtype=int)
                # Use convolution for initial setup? No, N is small, direct is fast enough for init
                # Or use the incremental method to build it up?
                # Direct:
                A_list = list(A_set)
                # Convolution is fast
                A_bool = np.zeros(2 * N + 1, dtype=int)
                for a in A_set:
                    A_bool[a] = 1
                conv = np.convolve(A_bool, A_bool)
                for i in range(NUM_TARGETS):
                    t = TARGETS_MIN + i
                    delta = 0
                    if t % 2 == 0:
                        half = t // 2
                        if half in A_set:
                            delta = 1
                    mults[i] = (conv[t] + delta) // 2
                return mults

            mults = compute_mults_from_set(A_set)
            
            if np.min(mults) == 0:
                # Try to repair coverage greedily
                # If k is too small, it might be impossible to cover with k elements?
                # Actually, for k=15, max pairs 120 < 129 targets. Impossible to cover all!
                # Wait. 15*16/2 = 120. 129 targets. Pigeonhole: impossible to cover all 129 with 15 elements.
                # So k must be at least such that k(k+1)/2 >= 129.
                # 16*17/2 = 136. So k=16 is the absolute minimum.
                # Let's restart loop for k < 16.
                if k < 16:
                    continue
                # For k >= 16, it's possible but hard.
                # Greedy repair:
                for _ in range(20):
                    uncovered = np.where(mults == 0)[0]
                    if len(uncovered) == 0:
                        break
                    # Pick a random uncovered target
                    t_idx = uncovered[random.randint(0, len(uncovered)-1)]
                    t = TARGETS_MIN + t_idx
                    # Find pair (a,b) not in A? No, we can only add elements to A.
                    # We need to add an element x such that t-x is in A.
                    # So x = t - a for some a in A.
                    # Let's try all a in A, see if x is valid and not in A.
                    candidates = []
                    for a in A_set:
                        x = t - a
                        if 0 <= x <= N and x not in A_set:
                            candidates.append(x)
                    if candidates:
                        x = random.choice(candidates)
                        A_set.add(x)
                        # Update mults incrementally?
                        # For simplicity in repair phase, recompute. It's fast.
                        mults = compute_mults_from_set(A_set)
                    else:
                        # Stuck? Add random element?
                        # If we can't cover this t with existing A, we might need to change A more drastically.
                        # But we are in a fixed k loop? No, repair adds elements.
                        # If we add elements, k increases.
                        # Let's just break and try next init if we can't cover.
                        break
                
                if np.min(mults) == 0:
                    continue # Failed to cover
            
            current_max = np.max(mults)
            if current_max < best_score:
                best_score = current_max
                best_A = set(A_set)
            
            # Local Search: Swap
            # We want to minimize max(mults).
            # Move: Remove a from A, Add b to A.
            # Delta in mults:
            # For each t, mult(t) changes by:
            # -1 for each pair (a, y) in A\{a} such that a+y=t
            # -1 for pair (a,a) if 2a=t
            # +1 for each pair (b, y) in A\{a} such that b+y=t
            # +1 for pair (b,b) if 2b=t
            # Note: if a=b, no change.
            
            # To make this fast, we can precompute for each element x in A, the vector of targets it contributes to.
            # contrib[x] is a vector of length NUM_TARGETS, where contrib[x][t_idx] = 1 if x + (t-x) is a valid pair in A?
            # No, contrib[x] should represent the pairs involving x.
            # If we remove x, we lose pairs (x, y) for all y in A.
            # If we add z, we gain pairs (z, y) for all y in A\{x}.
            
            # Let's maintain a "degree" or "contribution" matrix?
            # Since N=128, NUM_TARGETS=129.
            # We can compute the delta vector for a swap (remove a, add b) in O(|A|).
            
            steps = 1000
            temp = 2.0
            cooling = 0.99
            
            for step in range(steps):
                if time.time() > deadline - 2:
                    break
                    
                # Pick a random element a to remove
                a = random.choice(list(A_set))
                # Pick a random element b to add (not in A)
                # Candidates: 0..N not in A
                # To speed up, we can precompute list of non-A? Or just random guess?
                # N=128, |A|~20. 108 candidates.
                b = random.randint(0, N)
                if b in A_set:
                    # Retry or pick from list
                    # Simple retry loop
                    while b in A_set:
                        b = random.randint(0, N)
                
                if a == b:
                    continue
                
                # Calculate delta
                # new_mults = mults - contrib(a) + contrib(b) - correction for (a,b) pair?
                # Let's define contrib(x, A_set) as the vector of multiplicities contributed by x to targets, 
                # assuming x is in A and all other elements in A\{x} are present.
                # Actually, it's easier to compute the change directly.
                
                # Change in mults for target t:
                # old: count of pairs in A summing to t
                # new: count of pairs in A\{a} U {b} summing to t
                
                # Pairs involving a in old A:
                # (a, y) where y in A, y != a. And (a, a).
                # Pairs involving b in new A:
                # (b, y) where y in A\{a}. And (b, b).
                
                # Let's compute delta vector
                delta = np.zeros(NUM_TARGETS, dtype=int)
                
                # Subtract contributions of a
                # For each y in A:
                #   if y == a: pair (a,a) sums to 2a. mults[2a] -= 1
                #   if y != a: pair (a,y) sums to a+y. mults[a+y] -= 1
                # But we must be careful with unordered pairs.
                # My mults definition: m(t) = # unordered pairs {u,v} in A with u+v=t.
                
                # When we remove a:
                # We lose the pair {a, a} if 2a is in target range.
                # We lose the pair {a, y} for each y in A, y != a.
                
                # When we add b:
                # We gain the pair {b, b} if 2b is in target range.
                # We gain the pair {b, y} for each y in A\{a}.
                
                # Note: if b == a, no change (handled above).
                # Note: if b is already in A, we can't add it (handled above).
                
                # Iterate through A to compute delta
                # This is O(|A|) per swap. |A| ~ 20.
                # 1000 steps * 20 ops = 20,000 ops. Very fast.
                
                for y in A_set:
                    # Remove a
                    if y == a:
                        t = 2 * a
                        if TARGETS_MIN <= t <= TARGETS_MAX:
                            delta[t - TARGETS_MIN] -= 1
                    else:
                        t = a + y
                        if TARGETS_MIN <= t <= TARGETS_MAX:
                            delta[t - TARGETS_MIN] -= 1
                    
                    # Add b
                    # We add b to A\{a}. So y is in A\{a} (since y iterates A, and y!=a is handled? No, y iterates all A)
                    # If y == a, the pair is {b, a}.
                    # If y != a, the pair is {b, y}.
                    # In both cases, t = b + y.
                    t = b + y
                    if TARGETS_MIN <= t <= TARGETS_MAX:
                        delta[t - TARGETS_MIN] += 1
                
                # Add pair {b, b}
                t = 2 * b
                if TARGETS_MIN <= t <= TARGETS_MAX:
                    delta[t - TARGETS_MIN] += 1
                
                new_mults = mults + delta
                new_max = np.max(new_mults)
                new_min = np.min(new_mults)
                
                # Acceptance
                if new_min < 0:
                    continue # Invalid (coverage broken)
                
                if new_max < current_max:
                    # Accept
                    A_set.remove(a)
                    A_set.add(b)
                    mults = new_mults
                    current_max = new_max
                    if current_max < best_score:
                        best_score = current_max
                        best_A = set(A_set)
                elif new_max == current_max:
                    # Accept with prob 0.5 to explore
                    if random.random() < 0.5:
                        A_set.remove(a)
                        A_set.add(b)
                        mults = new_mults
                else:
                    # Worse
                    delta_val = new_max - current_max
                    prob = np.exp(-delta_val / temp)
                    if random.random() < prob:
                        A_set.remove(a)
                        A_set.add(b)
                        mults = new_mults
                        current_max = new_max
                
                temp *= cooling
                
                if current_max == 1:
                    break # Perfect

    if best_A is None:
        # Fallback: Return a set that at least covers something?
        # The task requires coverage. If we failed, we return a random set, but it might be invalid.
        # Better to return a known valid set.
        # A = [0..128] covers everything with high mult.
        return list(range(0, 129))
    
    return list(best_A)

