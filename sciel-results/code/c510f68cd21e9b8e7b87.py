import time
import numpy as np
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    N = 10_000_000
    
    # Step 1: Compute sigma(n) for n=1..N
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i::i] = i
            
    h = np.zeros(N + 1, dtype=np.int64)
    g = np.zeros(N + 1, dtype=np.int64)
    r = np.zeros(N + 1, dtype=np.int64)
    sigma_arr = np.zeros(N + 1, dtype=np.int64)
    
    h[1] = 1
    g[1] = 1
    r[1] = 1
    sigma_arr[1] = 1
    
    spf_list = spf.tolist()
    h_list = h.tolist()
    g_list = g.tolist()
    r_list = r.tolist()
    sigma_list = sigma_arr.tolist()
    
    for n in range(2, N + 1):
        if time.time() - start_time > 20:
            break
        p = spf_list[n]
        m = n // p
        if m % p != 0:
            h_list[n] = p
            g_list[n] = 1 + p
            r_list[n] = m
        else:
            h_list[n] = h_list[m] * p
            g_list[n] = g_list[m] + h_list[n]
            r_list[n] = r_list[m]
        
        sigma_list[n] = sigma_list[r_list[n]] * g_list[n]
    
    if time.time() - start_time > 30:
        return [11, 6]

    # Build map: sigma_val -> list of indices
    sigma_to_indices = defaultdict(list)
    for i in range(1, N + 1):
        sigma_to_indices[sigma_list[i]].append(i)
        
    # Identify primes
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    target_log = np.log(1.5)
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    atoms = []
    
    for p in primes:
        if time.time() - start_time > 50:
            break
        S = p + 1
        if S in sigma_to_indices:
            candidates = sigma_to_indices[S]
            for u in candidates:
                if u == p:
                    continue
                # Check if u and p are coprime? 
                # u is composed of primes. p is prime.
                # If p divides u, then gcd(u,p) != 1.
                # But sigma(u) = p+1. If p|u, then p | sigma(u)?
                # Not necessarily.
                # However, for the product of atoms to work, we need disjoint primes.
                # For a single atom (u, p), we just need u, p <= 10^7.
                
                ratio = u / p
                err = abs(ratio - 1.5)
                if err < best_error:
                    best_error = err
                    best_a = u
                    best_b = p
                    
                ratio2 = p / u
                err2 = abs(ratio2 - 1.5)
                if err2 < best_error:
                    best_error = err2
                    best_a = p
                    best_b = u
                    
                # Store atom for combination
                # L = log(u/p)
                L = np.log(u) - np.log(p)
                atoms.append((L, u, p))
                
    # Try combining two atoms
    # Sort atoms by L
    atoms.sort(key=lambda x: x[0])
    
    # We want L1 + L2 ~ target_log
    # Iterate over atoms, and for each, find best match in others
    
    # To avoid O(M^2), we can use binary search on L
    # But M might be small enough?
    # Number of primes is ~6e5.
    # Number of atoms is likely small (only if p+1 has specific sigma preimages).
    
    if len(atoms) > 1:
        # Simple nested loop with pruning
        # Or binary search
        Ls = [a[0] for a in atoms]
        
        for i in range(len(atoms)):
            if time.time() - start_time > 55:
                break
            L1, u1, p1 = atoms[i]
            target_L2 = target_log - L1
            
            # Find j such that Ls[j] is closest to target_L2
            # Using bisect
            import bisect
            pos = bisect.bisect_left(Ls, target_L2)
            
            for j in [pos-1, pos]:
                if 0 <= j < len(atoms) and j != i:
                    L2, u2, p2 = atoms[j]
                    # Check disjointness
                    # Primes in u1, u2, p1, p2 must be distinct?
                    # u1, p1 are from atom 1. u2, p2 from atom 2.
                    # We need primes in u1 disjoint from primes in u2.
                    # And p1 != p2.
                    # And p1 not in u2, p2 not in u1.
                    
                    # Factorization check is expensive.
                    # Approximation: check gcd(u1, u2) == 1 and gcd(u1, p2)==1 etc.
                    import math
                    if math.gcd(u1, u2) != 1:
                        continue
                    if math.gcd(u1, p2) != 1:
                        continue
                    if math.gcd(p1, u2) != 1:
                        continue
                    if p1 == p2:
                        continue
                    if math.gcd(p1, p2) != 1:
                        continue
                        
                    a = u1 * u2
                    b = p1 * p2
                    if a > N or b > N:
                        continue
                        
                    ratio = a / b
                    err = abs(ratio - 1.5)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        
    return [int(best_a), int(best_b)]
