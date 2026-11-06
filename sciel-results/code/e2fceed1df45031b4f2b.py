import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 512
    # Build adjacency: node i (0-indexed) represents value i+1
    # adj[i] is a bitmask of neighbors
    adj = [0] * n
    
    # Precompute LCMs or check on the fly?
    # On the fly is O(N^2) which is 512^2 = 262k operations. Very fast.
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            l = a * b // gcd(a, b)
            if l <= 512:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Initial best clique: Divisors of 360 (size 24)
    # 360 = 2^3 * 3^2 * 5
    divs_360 = [d for d in range(1, 361) if 360 % d == 0]
    # Convert to 0-indexed mask
    best_clique_mask = 0
    for d in divs_360:
        best_clique_mask |= (1 << (d - 1))
    best_size = len(divs_360)
    best_clique = divs_360[:]
    
    # Try a greedy heuristic to find a larger initial clique
    def greedy_clique():
        # Start with all nodes
        P = (1 << n) - 1
        R = 0
        while P:
            # Pick node with max degree in P
            best_node = -1
            max_deg = -1
            temp = P
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = (P & adj[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_node = v
            if best_node == -1:
                break
            R |= (1 << best_node)
            P &= adj[best_node]
        
        if R.bit_count() > best_size:
            return [i + 1 for i in range(n) if (R >> i) & 1]
        return None
    
    g_clique = greedy_clique()
    if g_clique:
        best_clique = g_clique
        best_size = len(best_clique)
        # Update mask
        best_clique_mask = 0
        for v in best_clique:
            best_clique_mask |= (1 << (v - 1))

    def bronk(R, P, X):
        nonlocal best_clique, best_size, best_clique_mask
        if time.time() > deadline:
            return
        if not P and not X:
            sz = R.bit_count()
            if sz > best_size:
                best_size = sz
                best_clique = [i + 1 for i in range(n) if (R >> i) & 1]
                best_clique_mask = R
            return
        if not P:
            return
        
        # Upper bound: if |R| + |P| <= best_size, we can't improve
        if R.bit_count() + P.bit_count() <= best_size:
            return
        
        # Choose pivot u from P ∪ X to maximize |P ∩ N(u)|
        union = P | X
        if union == 0:
            return
        
        pivot = -1
        max_count = -1
        temp = union
        # Iterate through nodes in union
        # Optimization: If we find a node where P is a subset of N(u), 
        # then P \ N(u) is empty, and we don't need to branch.
        # But we still need to check if it's a maximal clique.
        # Actually, if P \ N(u) is empty, the loop below runs 0 times.
        # So we just return (since X is not empty? No, if X is not empty, it's not maximal).
        # Wait, Bron-Kerbosch condition:
        # If P \ N(u) is empty, there are no candidates to branch on.
        # The function returns without updating best_clique.
        # This is correct because if X is not empty, R is not maximal.
        # If X is empty, R is maximal, but we would have caught it in `if not P and not X`.
        # Wait, if P is not empty, but P \ N(u) is empty, it means P is a subset of N(u).
        # Then we iterate 0 times.
        
        while temp:
            lsb = temp & (-temp)
            u = lsb.bit_length() - 1
            temp ^= lsb
            count = (P & adj[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
                if count == P.bit_count():
                    # Perfect pivot, no need to check others
                    break
        
        if pivot == -1:
            # Fallback
            lsb = union & (-union)
            pivot = lsb.bit_length() - 1
        
        # Iterate over P \ N(pivot)
        candidates = P & ~adj[pivot]
        while candidates:
            lsb = candidates & (-candidates)
            v = lsb.bit_length() - 1
            candidates ^= lsb
            
            bronk(R | lsb, P & adj[v], X & adj[v])
            P ^= lsb
            X |= lsb
    
    all_nodes = (1 << n) - 1
    bronk(0, all_nodes, 0)
    
    if not best_clique:
        return [1]
    
    return best_clique
