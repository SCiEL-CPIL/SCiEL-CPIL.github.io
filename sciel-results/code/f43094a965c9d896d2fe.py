import math
import random

def min_span(m):
    # Find least L such that (m, m+L] has distinct b_i div by i
    # Max L is 144
    for L in range(1, 145):
        # Candidates for each i are offsets k in 1..L such that (m+k) % i == 0
        # We need a matching from i in 1..12 to distinct k in 1..L
        
        # match[k] = i
        match = [-1] * (L + 1)
        
        def dfs(i, visited):
            for k in range(1, L + 1):
                if visited[k]:
                    continue
                if (m + k) % i != 0:
                    continue
                visited[k] = True
                if match[k] == -1 or dfs(match[k], visited):
                    match[k] = i
                    return True
            return False
        
        # Try to match all i in 1..12
        success = True
        for i in range(1, 13):
            visited = [False] * (L + 1)
            if not dfs(i, visited):
                success = False
                break
        
        if success:
            return L
            
    return 144 # Fallback

def construct():
    best_m = 0
    best_val = 0
    candidates = set()
    
    # Period is lcm(1..12) = 27720
    PERIOD = 27720
    
    # 1. CRT-based generation for large spans
    # We try to construct m that supports a matching for span L
    # by finding consistent residues r_i
    
    def find_residue_set(L, rand_seed=None):
        if rand_seed is not None:
            random.seed(rand_seed)
            
        r = [0] * 13 # 1-indexed
        
        def backtrack(i):
            if i == 13:
                return r[1:13]
            
            # Get valid values for r_i
            # Must be distinct from r_1..r_{i-1}
            # Must satisfy r_i == r_j mod gcd(i, j) for j < i
            
            used_vals = set(r[1:i])
            # To speed up, we can pre-filter values?
            # Just iterate 1..L
            
            # Randomize order to find different solutions
            vals = list(range(1, L + 1))
            random.shuffle(vals)
            
            for val in vals:
                if val in used_vals:
                    continue
                
                valid = True
                for j in range(1, i):
                    g = math.gcd(i, j)
                    if (val - r[j]) % g != 0:
                        valid = False
                        break
                
                if valid:
                    r[i] = val
                    res = backtrack(i + 1)
                    if res:
                        return res
                    r[i] = 0
            return None
        
        return backtrack(1)

    # Generate candidates for decreasing L
    # We want m that has a matching at high L.
    # If we find such m, f(12, m) <= L.
    # But we hope f(12, m) is close to L.
    
    for L in range(144, 60, -1):
        # Try to find a few valid residue sets
        for seed in range(5):
            res_set = find_residue_set(L, seed)
            if res_set:
                # Solve for m in [0, PERIOD)
                # m + r_i == 0 mod i => m == -r_i mod i
                # We can just scan m since PERIOD is small
                # Or use CRT solver
                # Scanning is 27720 * 12 ops ~ 300k ops.
                # 5 seeds * 84 L values * 300k = 126M ops. Might be slow.
                
                # Better: Use incremental CRT or just check consistency and solve
                # Since we already checked consistency in find_residue_set,
                # a solution exists.
                
                # Let's just pick m = 0 and adjust?
                # No, we need specific m.
                
                # We can solve the system of congruences.
                # m = x
                # x == -r_1 mod 1 (0)
                # x == -r_2 mod 2
                # ...
                
                # We can use the fact that we checked consistency.
                # Let's just find one m by scanning a small range?
                # No, m is unique mod 27720.
                
                # Efficient way:
                # Start with m=0, mod=1
                # For i=2..12:
                #   Find m' in [0, lcm(mod, i)) such that m' == m mod mod and m' == -r_i mod i
                #   Update m, mod
                
                m_val = 0
                m_mod = 1
                for i in range(1, 13):
                    target = (-res_set[i-1]) % i
                    # Solve m_val + m_mod * k == target (mod i)
                    # m_mod * k == target - m_val (mod i)
                    # Let g = gcd(m_mod, i)
                    # Solution exists if (target - m_val) % g == 0 (guaranteed by consistency)
                    g = math.gcd(m_mod, i)
                    # Divide by g
                    m_mod_g = m_mod // g
                    i_g = i // g
                    rhs = (target - m_val) // g
                    
                    # Find k such that m_mod_g * k == rhs (mod i_g)
                    # Inverse of m_mod_g mod i_g
                    # Since gcd(m_mod_g, i_g) == 1, inverse exists
                    # Use pow(m_mod_g, -1, i_g)
                    inv = pow(m_mod_g, -1, i_g)
                    k = (rhs * inv) % i_g
                    
                    new_m = m_val + m_mod * k
                    new_mod = m_mod * i_g
                    m_val = new_m % new_mod
                    m_mod = new_mod
                    
                candidates.add(m_val)
        
        # Add some random m
        for _ in range(5):
            candidates.add(random.randint(0, PERIOD - 1))
            
    # Evaluate all candidates
    for m in candidates:
        val = min_span(m)
        if val > best_val:
            best_val = val
            best_m = m
            
    return best_m
