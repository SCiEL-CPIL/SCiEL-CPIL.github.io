import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    # 255 = 3 * 5 * 17
    # Every element in the set must share a factor with 255, so must be divisible by 3, 5, or 17.
    # Additionally, for any pair (a, b) in the set, gcd(a,b) > 1.
    # Since every element shares a factor with 255, they are all in the union of multiples of 3, 5, 17 in [1,255].
    # We want the largest clique in the graph where vertices are these candidates and edges connect pairs with gcd > 1.
    # But we must include 255.
    
    # Build candidate set: all i in [1,255] with gcd(i, 255) > 1
    candidates = [i for i in range(1, 256) if math.gcd(i, 255) > 1]
    if 255 not in candidates:
        candidates.append(255)
    
    n = len(candidates)
    idx = {v: i for i, v in enumerate(candidates)}
    idx_255 = idx[255]
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    def popcount(x):
        return bin(x).count("1")
    
    # Bron-Kerbosch with pivot, starting from R={255}
    best_size = 1
    best_mask = (1 << idx_255)
    
    def bron_kerbosch(R, P, X):
        nonlocal best_size, best_mask
        if time.time() - start_time > DEADLINE:
            return
        
        r_size = popcount(R)
        if r_size > best_size:
            best_size = r_size
            best_mask = R
        
        if P == 0 and X == 0:
            return
        
        # Pivot: choose from P|X with max degree in P
        U = P | X
        if U == 0:
            return
        pivot = -1
        max_deg = -1
        temp = U
        while temp:
            lb = temp & (-temp)
            v = lb.bit_length() - 1
            deg = popcount(P & adj[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
            temp ^= lb
        
        # Candidates: P vertices not adjacent to pivot
        cand = P & ~adj[pivot]
        
        while cand:
            if time.time() - start_time > DEADLINE:
                return
            lb = cand & (-cand)
            v = lb.bit_length() - 1
            bron_kerbosch(R | lb, P & adj[v], X & adj[v])
            P ^= lb
            X |= lb
            cand ^= lb
    
    # Start: R={255}, P=neighbors of 255, X=empty
    R = (1 << idx_255)
    P = adj[idx_255]
    X = 0
    
    bron_kerbosch(R, P, X)
    
    result = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        result.append(candidates[v])
        m ^= lb
    result.sort()
    
    if 255 not in result:
        result.append(255)
        result.sort()
    
    return result
