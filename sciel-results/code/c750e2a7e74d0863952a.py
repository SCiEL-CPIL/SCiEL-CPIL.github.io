import time

def construct():
    start = time.time()
    deadline = start + 85
    
    # Numbers 2..256
    # L candidates: 2..128 (indices 0..126 in L-space)
    # H candidates: 129..256 (indices 0..126 in H-space)
    
    L_nums = list(range(2, 129))  # 2..128, 127 numbers
    H_nums = list(range(129, 257))  # 129..256, 128 numbers
    nL = len(L_nums)
    nH = len(H_nums)
    
    from math import gcd
    
    # Conflict graph on L: conflictL[i] is bitset of j where lcm(L_nums[i], L_nums[j]) <= 256
    conflictL = [0] * nL
    for i in range(nL):
        a = L_nums[i]
        bits = 0
        for j in range(nL):
            if i == j:
                continue
            b = L_nums[j]
            l = a * b // gcd(a, b)
            if l <= 256:
                bits |= (1 << j)
        conflictL[i] = bits
    
    # C(x) for x in L: bitset over H where lcm(x, h) <= 256
    C = [0] * nL
    for i in range(nL):
        a = L_nums[i]
        bits = 0
        for j in range(nH):
            h = H_nums[j]
            l = a * h // gcd(a, h)
            if l <= 256:
                bits |= (1 << j)
        C[i] = bits
    
    # Weights
    wL = [1.0 / x for x in L_nums]
    wH = [1.0 / x for x in H_nums]
    
    # H(L) = all H bits minus union of C(x) for x in L
    # Represent L as bitset over nL
    # H(L) as bitset over nH
    
    def compute_H(L_bits):
        """Return H bitset given L bitset."""
        blocked = 0
        x = L_bits
        while x:
            lsb = x & (-x)
            i = lsb.bit_length() - 1
            blocked |= C[i]
            x ^= lsb
        return ((1 << nH) - 1) ^ blocked
    
    def objective(L_bits):
        """Compute f(L) = sum_{x in L} wL[x] + sum_{h in H(L)} wH[h]."""
        s = 0.0
        x = L_bits
        while x:
            lsb = x & (-x)
            i = lsb.bit_length() - 1
            s += wL[i]
            x ^= lsb
        H_bits = compute_H(L_bits)
        x = H_bits
        while x:
            lsb = x & (-x)
            j = lsb.bit_length() - 1
            s += wH[j]
            x ^= lsb
        return s
    
    # Initial solution: L = empty, H = all
    best_L = 0
    best_w = objective(best_L)
    
    # Local search
    def local_search(L_bits, w, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_gain = 1e-12
            best_new_L = None
            H_bits = compute_H(L_bits)
            
            # 1-add: add x not in L, not in conflict with L
            for i in range(nL):
                if L_bits & (1 << i):
                    continue
                if L_bits & conflictL[i]:
                    continue
                # Gain = wL[i] - sum_{h in C[i] & H_bits} wH[h]
                lost = 0.0
                x = C[i] & H_bits
                while x:
                    lsb = x & (-x)
                    j = lsb.bit_length() - 1
                    lost += wH[j]
                    x ^= lsb
                gain = wL[i] - lost
                if gain > best_gain:
                    best_gain = gain
                    best_new_L = L_bits | (1 << i)
            
            # 1-remove: remove x from L
            for i in range(nL):
                if not (L_bits & (1 << i)):
                    continue
                # H after removal: H_new = all H minus union of C[y] for y in L\{i}
                L_new = L_bits ^ (1 << i)
                H_new = compute_H(L_new)
                # Gained H: H_new & ~H_bits
                gained = 0.0
                x = H_new & ~H_bits
                while x:
                    lsb = x & (-x)
                    j = lsb.bit_length() - 1
                    gained += wH[j]
                    x ^= lsb
                gain = gained - wL[i]
                if gain > best_gain:
                    best_gain = gain
                    best_new_L = L_new
            
            # 2-swap: remove i, add j
            for i in range(nL):
                if not (L_bits & (1 << i)):
                    continue
                L_minus_i = L_bits ^ (1 << i)
                for j in range(nL):
                    if L_bits & (1 << j):
                        continue
                    if L_minus_i & conflictL[j]:
                        continue
                    L_new = L_minus_i | (1 << j)
                    H_new = compute_H(L_new)
                    # Compute gain
                    # Old: wL[i] + sum_{h in H_bits} wH[h]
                    # New: wL[j] + sum_{h in H_new} wH[h]
                    # Gain = (wL[j] - wL[i]) + (sum_{h in H_new} wH[h] - sum_{h in H_bits} wH[h])
                    # = (wL[j] - wL[i]) + sum_{h in H_new\H_bits} wH[h] - sum_{h in H_bits\H_new} wH[h]
                    gained = 0.0
                    x = H_new & ~H_bits
                    while x:
                        lsb = x & (-x)
                        k = lsb.bit_length() - 1
                        gained += wH[k]
                        x ^= lsb
                    lost = 0.0
                    x = H_bits & ~H_new
                    while x:
                        lsb = x & (-x)
                        k = lsb.bit_length() - 1
                        lost += wH[k]
                        x ^= lsb
                    gain = (wL[j] - wL[i]) + gained - lost
                    if gain > best_gain:
                        best_gain = gain
                        best_new_L = L_new
            
            if best_new_L is not None:
                L_bits = best_new_L
                w += best_gain
                improved = True
        
        return L_bits, w
    
    best_L, best_w = local_search(best_L, best_w, deadline)
    
    # Also try greedy initializations
    import random
    random.seed(42)
    
    # Greedy by weight
    order = sorted(range(nL), key=lambda i: -wL[i])
    L_greedy = 0
    for i in order:
        if not (L_greedy & (1 << i)):
            if not (L_greedy & conflictL[i]):
                L_greedy |= (1 << i)
    w_greedy = objective(L_greedy)
    L_greedy, w_greedy = local_search(L_greedy, w_greedy, deadline)
    if w_greedy > best_w:
        best_w = w_greedy
        best_L = L_greedy
    
    # Random restarts
    for _ in range(20):
        if time.time() > deadline - 2:
            break
        L_rand = 0
        for i in range(nL):
            if random.random() < 0.3:
                if not (L_rand & conflictL[i]):
                    L_rand |= (1 << i)
        w_rand = objective(L_rand)
        L_rand, w_rand = local_search(L_rand, w_rand, deadline)
        if w_rand > best_w:
            best_w = w_rand
            best_L = L_rand
    
    # Build final solution
    H_final = compute_H(best_L)
    result = []
    x = best_L
    while x:
        lsb = x & (-x)
        i = lsb.bit_length() - 1
        result.append(L_nums[i])
        x ^= lsb
    x = H_final
    while x:
        lsb = x & (-x)
        j = lsb.bit_length() - 1
        result.append(H_nums[j])
        x ^= lsb
    
    return result
