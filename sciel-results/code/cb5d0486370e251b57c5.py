import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # Precompute gcds
    max_m = 128
    gcd_table = {}
    for i in range(2, max_m + 1):
        for j in range(i + 1, max_m + 1):
            gcd_table[(i, j)] = gcd(i, j)
    
    # For each modulus, compute which other moduli it's coprime with
    # A modulus m can coexist with n only if gcd(m,n) > 1
    # So we want a large subset where no two are coprime
    
    # Greedy: try to add moduli in order, checking if we can assign a residue
    # Process moduli in order of decreasing number of coprime conflicts (most constrained first)
    
    # Count for each m, how many other moduli in 2..128 are coprime to m
    coprime_count = {}
    for m in range(2, max_m + 1):
        cnt = 0
        for n in range(2, max_m + 1):
            if m != n and gcd(m, n) == 1:
                cnt += 1
        coprime_count[m] = cnt
    
    # Sort moduli: most constrained first (most coprime partners)
    moduli_sorted = sorted(range(2, max_m + 1), key=lambda m: (-coprime_count[m], m))
    
    # Greedy assignment
    assigned = {}  # modulus -> residue
    for m in moduli_sorted:
        if time.time() > deadline:
            break
        found = False
        for r in range(m):
            ok = True
            for m2, r2 in assigned.items():
                g = gcd(m, m2)
                if g == 1:
                    ok = False
                    break
                if (r2 - r) % g == 0:
                    ok = False
                    break
            if ok:
                assigned[m] = r
                found = True
                break
        if not found:
            continue
    
    if not assigned:
        # Fallback to the seed solution
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    result = [[r, m] for m, r in sorted(assigned.items())]
    return result
