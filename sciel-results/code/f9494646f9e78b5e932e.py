def construct():
    import time
    import math

    deadline = time.time() + 55

    denoms = list(range(2, 49))
    n = len(denoms)

    # Calculate LCM
    def lcm(a, b):
        return a * b // math.gcd(a, b)

    L = 1
    for d in denoms:
        L = lcm(L, d)

    weights = [L // d for d in denoms]
    
    # Suffix sums of weights
    suf = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suf[i] = suf[i + 1] + weights[i]

    # Greedy initialization
    best_sum = 0
    best_idxs = []
    current_sum = 0
    for i, w in enumerate(weights):
        if current_sum + w < L:
            current_sum += w
            best_idxs.append(i)
    best_sum = current_sum

    # DP with integer sums
    # State: dict sum -> list of indices
    # To save memory and time, we might just store the sum and reconstruct later?
    # No, storing indices is O(N) per state.
    # If number of states is K, memory is K*N.
    # If K is ~10^5, N=47, memory is ~5*10^6 ints. Fine.
    
    achievable = {0: []}

    for i, w in enumerate(weights):
        new_achievable = {}
        max_rem = suf[i + 1]
        
        for s, idxs in achievable.items():
            # Prune existing state if it can't beat best
            if s + max_rem > best_sum:
                if s not in new_achievable:
                    new_achievable[s] = idxs[:]
            
            # Add current weight
            new_s = s + w
            if new_s < L:
                if new_s > best_sum:
                    best_sum = new_s
                    best_idxs = idxs + [i]
                
                # Prune new state
                if new_s + max_rem > best_sum:
                    if new_s not in new_achievable:
                        new_achievable[new_s] = idxs + [i]
        
        achievable = new_achievable
        
        if time.time() > deadline:
            break

    if not best_idxs:
        return [0]

    return best_idxs
