"""
R_4(C6): maximize n with a 4-colouring of K_n with no monochromatic C6.
Uses Z_17 cyclic construction plus local search fallback.
"""
import time
import random
import numpy as np

DEADLINE_MARGIN = 5.0

def _count_mono_c6(colors, n):
    """
    Count the number of monochromatic C6s in the coloring.
    """
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    
    total_c6 = 0
    for c in range(4):
        for v0 in range(n):
            m1 = adj[c][v0]
            while m1:
                lsb = m1 & -m1
                v1 = lsb.bit_length() - 1
                m1 ^= lsb
                
                m2 = adj[c][v1] & ~(1 << v0)
                while m2:
                    lsb2 = m2 & -m2
                    v2 = lsb2.bit_length() - 1
                    m2 ^= lsb2
                    
                    m3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                    while m3:
                        lsb3 = m3 & -m3
                        v3 = lsb3.bit_length() - 1
                        m3 ^= lsb3
                        
                        m4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                        while m4:
                            lsb4 = m4 & -m4
                            v4 = lsb4.bit_length() - 1
                            m4 ^= lsb4
                            
                            m5 = adj[c][v4] & adj[c][v0]
                            m5 &= ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                            
                            if m5:
                                total_c6 += m5.bit_count()
                                
    return total_c6

def _has_c6_in_color(adj, n, c):
    """Check if color c's adjacency contains any C6."""
    for v0 in range(n):
        m1 = adj[c][v0]
        while m1:
            lsb = m1 & -m1
            v1 = lsb.bit_length() - 1
            m1 ^= lsb
            
            m2 = adj[c][v1] & ~(1 << v0)
            while m2:
                lsb2 = m2 & -m2
                v2 = lsb2.bit_length() - 1
                m2 ^= lsb2
                
                m3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                while m3:
                    lsb3 = m3 & -m3
                    v3 = lsb3.bit_length() - 1
                    m3 ^= lsb3
                    
                    m4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while m4:
                        lsb4 = m4 & -m4
                        v4 = lsb4.bit_length() - 1
                        m4 ^= lsb4
                        
                        m5 = adj[c][v4] & adj[c][v0]
                        m5 &= ~(1 << v0) & ~(1 << v1) & ~(1 << v2) & ~(1 << v3)
                        
                        if m5:
                            return True
    return False

def _is_valid_coloring(colors, n):
    """Check if coloring has no monochromatic C6."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    for c in range(4):
        if _has_c6_in_color(adj, n, c):
            return False
    return True

def _construct_z17():
    """
    Construct a 4-coloring of K_17 using Z_17 cyclic structure.
    Color edge (i,j) by the difference d = (j - i) mod 17.
    Partition differences {1,...,8} into 4 groups such that no group
    contains a C6 in the Cayley graph.
    
    The known valid partition for C6-avoidance uses:
    Group 0: {1, 2, 4, 8}  (powers of 2 mod 17)
    Group 1: {3, 6, 12, 7}
    Group 2: {5, 10, 13, 9}
    Group 3: {11, 15, 16, 14}
    
    But we need to verify. Let's use a different approach:
    The standard Schur-type result: color by quadratic residues.
    QR mod 17: {1, 2, 4, 8, 9, 13, 15, 16}
    NQR mod 17: {3, 5, 6, 7, 10, 11, 12, 14}
    
    We need 4 colors. Let's split QR and NQR into 2 each.
    """
    # Try all partitions of {1..8} into 4 sets of size 2 (since diff and -diff are same edge)
    # Actually differences are 1..8 (each represents {d, 17-d})
    # We need to assign each of 8 differences to one of 4 colors
    
    # Use a known valid coloring: based on the fact that Z_17 is a prime field
    # Let's try: color d by floor((d-1)/2) % 4 won't work...
    # 
    # Better: use the multiplicative structure. 
    # Z_17* has order 16. Split into 4 cosets of the subgroup of order 4.
    # Subgroup H = {1, 4, 13, 16} (since 4^1=4, 4^2=16, 4^3=64=13, 4^4=52=1)
    # Wait: 4^2 = 16, 4^3 = 64 mod 17 = 64-51=13, 4^4=52 mod 17=52-34=18 mod 17=1. Yes.
    # Cosets: H*1={1,4,13,16}, H*2={2,8,10,15}, H*3={3,12,5,14}, H*4 already in H...
    # H*2: 2*{1,4,13,16}={2,8,26=9,32=15} -> {2,8,9,15}
    # H*3: 3*{1,4,13,16}={3,12,39=5,48=14} -> {3,5,12,14}
    # H*5: 5*{1,4,13,16}={5,20=3,65=14,80=12} -> same as H*3
    # Let me list all:
    # H = {1, 4, 13, 16}
    # 2H = {2, 8, 9, 15}
    # 3H = {3, 12, 5, 14}
    # 6H = {6, 24=7, 78=10, 96=11} -> {6,7,10,11}
    
    # So the 4 cosets are:
    # C0 = {1, 4, 13, 16}
    # C1 = {2, 8, 9, 15}
    # C2 = {3, 5, 12, 14}
    # C3 = {6, 7, 10, 11}
    
    # For edge (i,j) with i<j, diff = (j-i) % 17, which is in 1..16
    # The effective difference is min(diff, 17-diff) in 1..8
    # But for the Cayley graph, we use the full difference.
    # Edge (i,j): color by which coset contains (j-i) mod 17
    
    cosets = [
        {1, 4, 13, 16},
        {2, 8, 9, 15},
        {3, 5, 12, 14},
        {6, 7, 10, 11}
    ]
    
    # Build color map for differences 1..16
    diff_to_color = {}
    for c, coset in enumerate(cosets):
        for d in coset:
            diff_to_color[d] = c
    
    n = 17
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (j - i) % n
            colors.append(diff_to_color[d])
    
    # Verify
    if _is_valid_coloring(colors, n):
        return colors
    
    # Try other subgroup-based partitions
    # Try with generator g=3: 3^1=3, 3^2=9, 3^3=27=10, 3^4=30=13
    # H = {1, 3, 9, 10}
    # 2H = {2, 6, 18=1, 20=3} -- wait that's not right
    # 2*{1,3,9,10} = {2,6,18 mod 17=1, 20 mod 17=3} = {1,2,3,6} -- overlaps with H!
    # That's wrong. 2 is not in H, but 2*9=18=1 which IS in H. So 2H is not a coset.
    # The issue: H must be a subgroup. {1,3,9,10}: 3*3=9, 3*9=27=10, 3*10=30=13. Not closed!
    # 3^4 = 81 mod 17 = 81-68=13, not 1. So {1,3,9,10,13,5,15,11,16,14,12,4,8,2,6,7} is the full group.
    # Order of 3 mod 17: 3^1=3, 3^2=9, 3^3=10, 3^4=13, 3^5=5, 3^6=15, 3^7=11, 3^8=16, 3^9=14, 3^10=8, 3^11=7, 3^12=4, 3^13=12, 3^14=2, 3^15=6, 3^16=1. Order 16.
    # So no proper subgroup generated by 3.
    
    # Generator 4: 4^1=4, 4^2=16, 4^3=13, 4^4=1. Order 4. Good.
    # We already tried that.
    
    # Let's just try all 4-colorings of the 8 difference-classes (each class is {d, 17-d})
    # There are 4^8 = 65536 possibilities. Check each.
    diffs = list(range(1, 9))
    
    for mask in range(4 ** 8):
        assignment = [(mask >> (i * 2)) & 3 for i in range(8)]
        
        # Build full diff_to_color for 1..16
        dtc = {}
        for i, d in enumerate(diffs):
            dtc[d] = assignment[i]
            dtc[17 - d] = assignment[i]
        
        colors_try = []
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                colors_try.append(dtc[d])
        
        if _is_valid_coloring(colors_try, n):
            return colors_try
    
    return None

def _local_search(n, deadline):
    """
    Local search to find a 4-coloring of K_n with 0 monochromatic C6s.
    """
    num_edges = n * (n - 1) // 2
    best_colors = None
    best_count = float('inf')
    
    while time.time() < deadline:
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        current_count = _count_mono_c6(colors, n)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors[:]
            
        if current_count == 0:
            return colors
        
        current = colors[:]
        curr_count = current_count
        temp = 1.0
        step_size = 0.999
        
        while time.time() < deadline:
            if curr_count == 0:
                return current
            
            e_idx = random.randint(0, num_edges - 1)
            old_c = current[e_idx]
            new_c = (old_c + random.randint(1, 3)) % 4
            current[e_idx] = new_c
            
            new_count = _count_mono_c6(current, n)
            
            delta = new_count - curr_count
            if delta < 0 or (random.random() < np.exp(-delta / temp)):
                curr_count = new_count
                if curr_count < best_count:
                    best_count = curr_count
                    best_colors = current[:]
            else:
                current[e_idx] = old_c
            
            temp *= step_size
            
            if temp < 0.01:
                break
                
    return best_colors

def construct():
    start_time = time.time()
    deadline = start_time + (120.0 - DEADLINE_MARGIN)
    
    # First, try the Z_17 construction
    z17 = _construct_z17()
    if z17 is not None and _is_valid_coloring(z17, 17):
        return {'n': 17, 'colors': z17}
    
    # If Z_17 didn't work, try local search for n=17, then smaller
    for n in [17, 16, 15, 14]:
        if time.time() > deadline - 10.0:
            break
            
        result = _local_search(n, deadline)
        
        if result is not None:
            if _count_mono_c6(result, n) == 0:
                return {'n': n, 'colors': result}
        
    if time.time() < deadline - 5.0:
        result = _local_search(12, deadline)
        if result is not None and _count_mono_c6(result, 12) == 0:
            return {'n': 12, 'colors': result}
            
    random.seed(42)
    colors = [random.randint(0, 3) for _ in range(6)]
    return {'n': 4, 'colors': colors}
