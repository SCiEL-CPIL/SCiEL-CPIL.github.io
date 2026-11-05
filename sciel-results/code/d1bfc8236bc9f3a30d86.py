import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin
    
    def check_pair(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def check_all(classes):
        n = len(classes)
        for i in range(n):
            for j in range(i + 1, n):
                if not check_pair(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                    return False
        return True
    
    # Seed solution: powers of 2
    seed = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    best = seed
    best_count = len(seed)
    
    # Strategy: try to find a larger set using all even moduli in [2, 128]
    # Even moduli: 2,4,6,8,10,...,128 (64 values)
    # But we need distinct moduli and pairwise disjoint classes.
    # Use backtracking with MRV (most constrained variable first)
    
    even_moduli = [m for m in range(2, 129) if m % 2 == 0]
    n_mod = len(even_moduli)
    
    # Precompute gcd matrix
    gcd_mat = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            gcd_mat[i][j] = math.gcd(even_moduli[i], even_moduli[j])
            gcd_mat[j][i] = gcd_mat[i][ j]
    
    # Greedy approach: try to build up a set by adding moduli one at a time
    # Order moduli by "difficulty" - those with more divisors / higher gcds first
    # Actually, let's try multiple orderings
    
    def greedy_build(moduli_order, time_limit=10.0):
        """Build a set of disjoint classes using greedy with backtracking."""
        t0 = time.time()
        selected = []  # list of (residue, modulus)
        selected_indices = []
        
        for idx in moduli_order:
            m = even_moduli[idx]
            # Try all residues for this modulus
            found = False
            # Try residues in a smart order: start with 0, then 1, etc.
            # Or try to find any valid residue
            for r in range(m):
                ok = True
                for (r_s, m_s) in selected:
                    g = math.gcd(m, m_s)
                    if (r - r_s) % g == 0:
                        ok = False
                        break
                if ok:
                    selected.append([r, m])
                    found = True
                    break
            if not found:
                # Try with backtracking: remove last and try different
                # Simple approach: just skip this modulus
                pass
            if time.time() - t0 > time_limit:
                break
        
        return selected
    
    # Try different orderings
    # Order 1: ascending modulus
    order_asc = list(range(n_mod))
    # Order 2: descending modulus
    order_desc = list(range(n_mod - 1, -1, -1))
    # Order 3: by number of divisors (more divisors = more constraints = harder)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 2
                if i * i == n:
                    count -= 1
        return count
    
    order_by_divs = sorted(range(n_mod), key=lambda i: num_divisors(even_moduli[i]), reverse=True)
    # Order 4: by smallest prime factor (2 is best since all are even)
    # Actually all even, so SPF is 2. Sort by m/2 descending (larger odd part first)
    order_by_odd_part = sorted(range(n_mod), key=lambda i: even_moduli[i] // 2, reverse=True)
    
    for order in [order_asc, order_desc, order_by_divs, order_by_odd_part]:
        if time.time() > deadline:
            break
        result = greedy_build(order, time_limit=min(5.0, deadline - time.time()))
        if len(result) > best_count:
            best = result
            best_count = len(result)
    
    # Try randomized restarts
    random.seed(42)
    for attempt in range(200):
        if time.time() > deadline:
            break
        # Random ordering with bias toward certain moduli
        shuffled = list(range(n_mod))
        random.shuffle(shuffled)
        # Bias: put smaller moduli earlier (they have fewer residue choices)
        # Actually, put moduli with fewer valid residues first
        # For now, just try random
        result = greedy_build(shuffled, time_limit=1.0)
        if len(result) > best_count:
            best = result
            best_count = len(result)
    
    # Try a more systematic backtracking for the top candidates
    # Focus on a subset: try to include as many as possible
    # Use the insight that we want moduli that are "compatible"
    
    # Let's try: use all powers of 2 plus some other even numbers
    # The powers of 2 give 7. Can we add more?
    # For modulus 6: gcd(6,2)=2, gcd(6,4)=2, gcd(6,8)=2, gcd(6,16)=2, etc.
    # Need r_6 such that r_6 != r_2 mod 2, r_6 != r_4 mod 2, etc.
    # With seed residues: r_2=0, r_4=1, r_8=3, r_16=7, r_32=15, r_64=31, r_128=63
    # All r_2^k are odd except r_2=0. So for gcd=2, we need r_6 != 0 mod 2 (i.e., r_6 odd) 
    # AND r_6 != 1 mod 2 (i.e., r_6 even). Contradiction! Can't add 6.
    
    # So the seed solution might be locally optimal for that specific residue assignment.
    # But with different residues, we might do better.
    
    # Let's do a proper backtracking search on a curated set of moduli
    # Try: all even moduli, but use a smarter search
    
    def backtrack_search(moduli_subset, time_limit=20.0):
        """Backtracking search for maximum set."""
        t0 = time.time()
        n = len(moduli_subset)
        if n == 0:
            return []
        
        # Precompute gcds
        gcds = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                gcds[i][j] = math.gcd(moduli_subset[i], moduli_subset[j])
                gcds[j][i] = gcds[i][j]
        
        # Order by constraint: moduli with higher max gcd first
        def max_gcd(idx):
            return max(gcds[idx][j] for j in range(n) if j != idx) if n > 1 else 0
        
        order = sorted(range(n), key=max_gcd, reverse=True)
        
        assignment = [None] * n
        count = [0]
        
        def bt(pos):
            if time.time() - t0 > time_limit:
                return False
            if pos == n:
                count[0] = sum(1 for a in assignment if a is not None)
                return True
            idx = order[pos]
            m = moduli_subset[idx]
            # Try all residues
            for r in range(m):
                ok = True
                for prev_pos in range(pos):
                    pidx = order[prev_pos]
                    if assignment[pidx] is not None:
                        g = gcds[idx][pidx]
                        if (r - assignment[pidx]) % g == 0:
                            ok = False
                            break
                if ok:
                    assignment[idx] = r
                    if bt(pos + 1):
                        return True
                    assignment[idx] = None
            # Also try skipping this modulus
            assignment[idx] = None
            return bt(pos + 1)
        
        bt(0)
        # Collect results
        result = []
        for i in range(n):
            if assignment[i] is not None:
                result.append([assignment[i], moduli_subset[i]])
        return result
    
    # Try backtracking on a subset of promising moduli
    # Start with powers of 2 and add some others
    powers_of_2 = [2, 4, 8, 16, 32, 64, 128]
    other_even = [m for m in even_moduli if m not in powers_of_2]
    
    # Try subsets: powers of 2 + some others
    # First, try all even moduli with a time-limited search
    if time.time() < deadline - 10:
        # Try a curated set: all even moduli up to a certain point
        # Use the first 20 even moduli
        subset = even_moduli[:20]
        result = backtrack_search(subset, time_limit=min(10.0, deadline - time.time() - 2))
        if len(result) > best_count:
            best = result
            best_count = len(result)
    
    # Try another subset: powers of 2 + multiples of 6
    subset2 = powers_of_2 + [m for m in even_moduli if m % 6 == 0 and m not in powers_of_2]
    if len(subset2) <= 20 and time.time() < deadline - 10:
        result = backtrack_search(subset2, time_limit=min(10.0, deadline - time.time() - 2))
        if len(result) > best_count:
            best = result
            best_count = len(result)
    
    # Final: try all even moduli with a very fast greedy + local search
    if time.time() < deadline - 5:
        # Multi-start greedy
        for seed_val in range(100):
            if time.time() > deadline - 2:
                break
            random.seed(seed_val)
            shuffled = list(range(n_mod))
            random.shuffle(shuffled)
            selected = []
            for idx in shuffled:
                m = even_moduli[idx]
                for r in range(m):
                    ok = True
                    for (r_s, m_s) in selected:
                        g = math.gcd(m, m_s)
                        if (r - r_s) % g == 0:
                            ok = False
                            break
                    if ok:
                        selected.append([r, m])
                        break
            if len(selected) > best_count:
                best = selected
                best_count = len(selected)
    
    return best
