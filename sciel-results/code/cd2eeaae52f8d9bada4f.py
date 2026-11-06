import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 58  # Safety margin (90s kill, but be conservative)
    
    N = 512
    
    # Precompute LCM compatibility using prime factorization for speed
    # For each number 1..512, compute its prime factorization
    # Two numbers a,b have lcm(a,b) <= 512 iff for every prime p,
    # max(v_p(a), v_p(b)) <= v_p(512) where 512 = 2^9
    # But more generally, lcm(a,b) = a*b/gcd(a,b) <= 512
    
    # Precompute gcd-based LCM check
    # neighbors[i] is a bitmask of all j where lcm(i,j) <= 512
    neighbors = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if (i // g) * j <= 512:
                mask |= (1 << (j - 1))
        neighbors[i] = mask
    
    initial_P = (1 << N) - 1
    
    # Greedy clique for initial lower bound
    def greedy_clique():
        P = initial_P
        R = []
        while P:
            best_v = -1
            max_deg = -1
            temp = P
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()  # 1-based
                deg = (P & neighbors[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            R.append(best_v)
            P &= ~(1 << (best_v - 1))
            P &= neighbors[best_v]
        return R
    
    best_clique = greedy_clique()
    
    # Try multiple greedy orderings
    for _ in range(10):
        alt_clique = greedy_clique()
        if len(alt_clique) > len(best_clique):
            best_clique = alt_clique
    
    # Greedy coloring bound - optimized
    def greedy_color_bound(P):
        if P == 0:
            return 0
        color_count = 0
        uncolored = P
        while uncolored:
            color_count += 1
            independent = 0
            temp = uncolored
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()  # 1-based
                if independent & neighbors[v] == 0:
                    independent |= lsb
                    uncolored ^= lsb
                    temp ^= lsb
                else:
                    temp ^= lsb
        return color_count
    
    def bron_kerbosch(R, P, X):
        nonlocal best_clique
        
        if time.time() > deadline:
            return
            
        if P == 0 and X == 0:
            if len(R) > len(best_clique):
                best_clique = R[:]
            return
            
        # Simple bound
        if len(R) + P.bit_count() <= len(best_clique):
            return
            
        # Coloring bound - key pruning mechanism
        c_bound = greedy_color_bound(P)
        if len(R) + c_bound <= len(best_clique):
            return
            
        PX = P | X
        if PX == 0:
            return
            
        # Find pivot: choose vertex in PX with max overlap with P
        max_overlap = -1
        pivot = -1
        temp = PX
        while temp:
            lsb = temp & -temp
            u = lsb.bit_length()
            overlap = P & neighbors[u]
            count = overlap.bit_count()
            if count > max_overlap:
                max_overlap = count
                pivot = u
            temp ^= lsb
            
        if pivot == -1:
            return

        candidates = P & ~neighbors[pivot]
        
        temp = candidates
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length()
            temp ^= lsb
            
            bron_kerbosch(R + [v], P & neighbors[v], X & neighbors[v])
            
            P &= ~lsb
            X |= lsb
            
            if time.time() > deadline:
                return

    bron_kerbosch([], initial_P, 0)
    
    return best_clique
