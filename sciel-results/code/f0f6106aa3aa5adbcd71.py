import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 512
    # 1. Precompute LCM compatibility
    neighbors = [0] * (N + 1)
    
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if (i // g) * j <= 512:
                mask |= (1 << (j - 1))
        neighbors[i] = mask
    
    # Initial P = all vertices
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
                v = lsb.bit_length() - 1 + 1
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
    
    # Try multiple greedy orderings to get a better initial lower bound
    for _ in range(5):
        alt_clique = greedy_clique()
        if len(alt_clique) > len(best_clique):
            best_clique = alt_clique
    
    # Divisor-core strategy: for each M, start with divisors of M, then greedily expand
    def expand_from_core(core_set):
        """Greedily expand a core set by adding compatible vertices."""
        R = list(core_set)
        R_mask = 0
        for v in R:
            R_mask |= (1 << (v - 1))
        
        # Find all vertices compatible with all in R
        compatible = initial_P
        for v in R:
            compatible &= neighbors[v]
        compatible &= ~R_mask
        
        # Greedy expansion
        while compatible:
            best_v = -1
            max_deg = -1
            temp = compatible
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length()
                deg = (compatible & neighbors[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp ^= lsb
            if best_v == -1:
                break
            R.append(best_v)
            R_mask |= (1 << (best_v - 1))
            compatible &= neighbors[best_v]
            compatible &= ~(1 << (best_v - 1))
        
        return R
    
    # Try divisor cores for all M from 1 to 512
    for M in range(1, N + 1):
        if time.time() > deadline - 5:
            break
        
        # Get all divisors of M
        divisors = []
        for d in range(1, M + 1):
            if M % d == 0:
                divisors.append(d)
        
        if not divisors:
            continue
        
        # Try expanding from this core
        expanded = expand_from_core(divisors)
        if len(expanded) > len(best_clique):
            best_clique = expanded
    
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
            
        # Always apply coloring bound - it's the key pruning mechanism
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
