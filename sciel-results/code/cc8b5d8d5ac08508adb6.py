import math
import time

def construct():
    """
    Construct a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Build candidate set: all x in [1,255] with gcd(x, 255) > 1
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)
    
    n = len(candidates)
    
    # Build adjacency as bitmasks for speed
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Find index of 255
    idx_255 = candidates.index(255)
    
    # We must include 255, so start with R={255}, P=N(255), X=empty
    best_clique_size = 1
    best_clique_mask = 1 << idx_255
    
    def popcount(x):
        return bin(x).count('1')
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique_size, best_clique_mask
        if time.time() > deadline:
            return
        
        # Prune: if |R| + |P| <= best, no improvement possible
        if popcount(R) + popcount(P) <= best_clique_size:
            return
        
        if P == 0 and X == 0:
            sz = popcount(R)
            if sz > best_clique_size:
                best_clique_size = sz
                best_clique_mask = R
            return
        
        # Choose pivot u from P ∪ X to maximize |P ∩ N(u)|
        U = P | X
        if U == 0:
            return
        
        # Find pivot with max |P & adj[u]|
        best_pivot = -1
        best_count = -1
        temp = U
        while temp:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            temp ^= lsb
            cnt = popcount(P & adj[u])
            if cnt > best_count:
                best_count = cnt
                best_pivot = u
        
        if best_pivot == -1:
            return
        
        # Iterate over P \ N(pivot)
        candidates_bitmask = P & ~(adj[best_pivot])
        
        temp = candidates_bitmask
        while temp:
            if time.time() > deadline:
                return
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            
            new_R = R | lsb
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X)
    
    # Start with 255 forced in the clique
    R = 1 << idx_255
    P = adj[idx_255]
    X = 0
    bron_kerbosch(R, P, X)
    
    # Convert back to values
    result_set = set()
    for i in range(n):
        if (best_clique_mask >> i) & 1:
            result_set.add(candidates[i])
    
    # Ensure 255 is in result
    if 255 not in result_set:
        result_set.add(255)
    
    # Verify validity
    res_list = list(result_set)
    for i in range(len(res_list)):
        for j in range(i + 1, len(res_list)):
            if math.gcd(res_list[i], res_list[j]) == 1:
                return list(range(3, 256, 3))
    
    if 255 not in res_list:
        return list(range(3, 256, 3))
    
    return res_list

# For quick testing
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Contains 255: {255 in result}")
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
    print(f"Valid: {valid}")
