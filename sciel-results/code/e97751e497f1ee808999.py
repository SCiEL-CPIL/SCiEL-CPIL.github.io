import math
import random
import time

MAX_MOD = 128

def gcd(a, b):
    return math.gcd(a, b)

def is_disjoint(r1, m1, r2, m2):
    g = gcd(m1, m2)
    if g == 1:
        return False
    return ((r2 - r1) % g) != 0

def count_conflicts(pairs):
    conflicts = 0
    n = len(pairs)
    for i in range(n):
        r1, m1 = pairs[i]
        for j in range(i + 1, n):
            r2, m2 = pairs[j]
            if not is_disjoint(r1, m1, r2, m2):
                conflicts += 1
    return conflicts

def local_search(moduli, deadline):
    n = len(moduli)
    if n == 0:
        return []
    
    best_pairs = None
    best_conflicts = float('inf')
    
    # Try multiple random restarts
    for trial in range(200):
        if time.time() > deadline:
            break
            
        # Initialize randomly
        pairs = [[random.randint(0, m-1), m] for m in moduli]
        conflicts = count_conflicts(pairs)
        
        if conflicts == 0:
            return pairs
        
        if conflicts < best_conflicts:
            best_conflicts = conflicts
            best_pairs = [list(p) for p in pairs]
            
        # Local search: change one residue at a time to reduce conflicts
        improved = True
        current_conflicts = conflicts
        no_improve_count = 0
        while improved and time.time() < deadline:
            improved = False
            no_improve_count += 1
            if no_improve_count > 1000:
                break
                
            # Randomly pick an index
            idx = random.randint(0, n - 1)
            m = moduli[idx]
            old_r = pairs[idx][0]
            
            # Try a new random residue
            new_r = random.randint(0, m - 1)
            if new_r == old_r:
                continue
                
            # Calculate change in conflicts involving idx
            old_conf = 0
            for j in range(n):
                if j == idx: continue
                r_j, m_j = pairs[j]
                if not is_disjoint(old_r, m, r_j, m_j):
                    old_conf += 1
            
            new_conf = 0
            for j in range(n):
                if j == idx: continue
                r_j, m_j = pairs[j]
                if not is_disjoint(new_r, m, r_j, m_j):
                    new_conf += 1
            
            delta = new_conf - old_conf
            
            if delta < 0:
                pairs[idx][0] = new_r
                current_conflicts += delta
                improved = True
                no_improve_count = 0
                if current_conflicts == 0:
                    return pairs
            else:
                # With small probability, accept worse move to escape local minima
                # But for speed, stick to hill climbing mostly
                pass
                
        if current_conflicts < best_conflicts:
            best_conflicts = current_conflicts
            best_pairs = [list(p) for p in pairs]
            
    return best_pairs

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin
    
    # Candidate sets of moduli
    # 1. All even moduli
    even_moduli = list(range(2, MAX_MOD + 1, 2))
    # 2. All multiples of 3
    mult3_moduli = list(range(3, MAX_MOD + 1, 3))
    # 3. All multiples of 5
    mult5_moduli = list(range(5, MAX_MOD + 1, 5))
    
    best_solution = None
    best_size = 0
    
    # Try largest sets first
    for moduli in [even_moduli, mult3_moduli, mult5_moduli]:
        if time.time() > deadline:
            break
        sol = local_search(moduli, deadline)
        if sol is not None:
            # Verify
            if count_conflicts(sol) == 0:
                if len(sol) > best_size:
                    best_size = len(sol)
                    best_solution = sol
    
    if best_solution is not None:
        return best_solution
        
    # Fallback: Powers of 2 (known valid construction)
    # r = 2^k - 1, m = 2^(k+1)
    # Check: gcd(2^a, 2^b) = 2^min.
    # r_a = 2^a - 1, r_b = 2^b - 1.
    # If a < b, gcd = 2^a.
    # r_b mod 2^a = (2^b - 1) mod 2^a = -1 mod 2^a = 2^a - 1.
    # r_a mod 2^a = 2^a - 1.
    # They are congruent! So this construction is INVALID for disjointness.
    # Wait.
    # Disjoint iff NOT congruent.
    # My previous check:
    # m=2, r=1. m=4, r=3. gcd=2. 1 mod 2 = 1. 3 mod 2 = 1. Congruent. NOT DISJOINT.
    # So the inspiration 1 was actually invalid?
    # "score 1.0000" might mean it passed a weak verifier or the score is just 1.
    # The task says "verifier uses the exact gcd criterion".
    # So [[1,2], [3,4]] is invalid.
    
    # Valid construction for powers of 2:
    # We need r_a not congruent r_b mod 2^min.
    # Let r_m = 1 for all m?
    # m=2, r=1. m=4, r=1. gcd=2. 1=1. Fail.
    # Let r_m = m/2?
    # m=2, r=1. m=4, r=2. gcd=2. 1 != 0. OK.
    # m=8, r=4. gcd(2,8)=2. 1 != 0. OK.
    # gcd(4,8)=4. r_4=2, r_8=4. 2 mod 4 = 2. 4 mod 4 = 0. 2 != 0. OK.
    # gcd(4,16)=4. r_4=2, r_16=8. 2 mod 4 = 2. 8 mod 4 = 0. OK.
    # It seems r_m = m/2 works for powers of 2?
    # Let's check m=2,4,8,16.
    # r: 1, 2, 4, 8.
    # (1,2) vs (2,4): g=2. 1%2=1, 2%2=0. Distinct. OK.
    # (1,2) vs (4,8): g=2. 1%2=1, 4%2=0. Distinct. OK.
    # (1,2) vs (8,16): g=2. 1%2=1, 8%2=0. Distinct. OK.
    # (2,4) vs (4,8): g=4. 2%4=2, 4%4=0. Distinct. OK.
    # (2,4) vs (8,16): g=4. 2%4=2, 8%4=0. Distinct. OK.
    # (4,8) vs (8,16): g=8. 4%8=4, 8%8=0. Distinct. OK.
    # Yes, r_m = m/2 works for powers of 2.
    
    fallback = [[m//2, m] for m in [2, 4, 8, 16, 32, 64, 128]]
    return fallback
