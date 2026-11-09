import time
import random
import numpy as np
from math import gcd

_DEAD = 145 # Safety margin

def construct():
    t0 = time.time()
    MAX_L = 100000
    N = 1000
    
    # Primes
    isprime = np.ones(N + 1, dtype=bool)
    isprime[0] = isprime[1] = False
    for i in range(2, int(N ** 0.5) + 1):
        if isprime[i]:
            isprime[i*i:N+1:i] = False
    primes = np.array([i for i in range(2, N + 1) if isprime[i]])
    prime_set = set(primes)
    
    # State
    # residues: dict p -> r
    residues = {}
    primes_used = set()
    current_sum = 0.0
    
    # NumPy arrays for coverage
    # cover_count[i] = number of classes covering i
    cover_count = np.zeros(MAX_L + 1, dtype=np.int32)
    
    def get_L():
        # Find first i >= 1 where cover_count[i] == 0
        # cover_count[0] is not used
        # We check 1..MAX_L
        # If all > 0, return MAX_L
        # np.where returns indices
        # We want the first one
        # To speed up, we can check in chunks or just use np.argmax on (cover_count[1:] == 0)
        # If no zero, argmax returns 0, but we need to handle the case where all are covered.
        # Let's just do a simple loop in Python? No, 100k is too slow in Python.
        # Use NumPy.
        # Create a boolean array: is_uncovered = cover_count[1:] == 0
        # If any is True, find first index.
        # Else return MAX_L.
        
        # Optimization: We only need to check from current_L onwards if we know it was covered before.
        # But for simplicity and correctness, let's do full check.
        # 100k check is ~10-20 microseconds in NumPy.
        
        # We can maintain a variable current_L and only scan forward.
        # But if we break coverage, we need to scan from 1.
        # Let's stick to full scan for safety, it's fast enough.
        
        # Actually, we can just return MAX_L if the sum of cover_count[1:] is MAX_L.
        # But checking for zero is better.
        
        # Let's use a helper that takes a start index.
        pass

    def get_L_from(start):
        # Scan from start to MAX_L
        # Return first index i >= start where cover_count[i] == 0
        # If none, return MAX_L
        # cover_count is 0-indexed. Integers 1..MAX_L map to indices 1..MAX_L.
        if start > MAX_L:
            return MAX_L
        # Check slice [start:MAX_L+1]
        # If all > 0, return MAX_L
        # Else return start + index of first 0
        segment = cover_count[start:MAX_L+1]
        if np.all(segment > 0):
            return MAX_L
        else:
            idx = np.argmax(segment == 0)
            return start + idx

    def update(p, r, sign):
        # sign = 1 add, -1 remove
        # indices: r, r+p, r+2p, ... < MAX_L
        # Note: r is in 0..p-1.
        # Integers are 1..MAX_L.
        # If r == 0, indices are p, 2p, ...
        # If r > 0, indices are r, r+p, ...
        # np.arange(r, MAX_L+1, p) generates 0-indexed array values.
        # We need to be careful: np.arange returns values.
        # If r=0, it returns 0, p, 2p...
        # But index 0 corresponds to integer 0, which is not in 1..MAX_L.
        # So we should skip 0 if present.
        # Or just use indices = np.arange(r, MAX_L+1, p)
        # and then mask indices > 0.
        
        indices = np.arange(r, MAX_L + 1, p)
        if r == 0:
            indices = indices[1:] # Skip 0
        # Ensure indices are within bounds (they are by construction)
        cover_count[indices] += sign

    def add_class(p, r):
        update(p, r, 1)
        residues[p] = r
        primes_used.add(p)
        return 1.0 / p

    def remove_class(p):
        r = residues[p]
        update(p, r, -1)
        del residues[p]
        primes_used.discard(p)
        return 1.0 / p

    def change_residue(p, r_new):
        r_old = residues[p]
        if r_old == r_new:
            return
        update(p, r_old, -1)
        update(p, r_new, 1)
        residues[p] = r_new

    # Greedy Initialization
    L = 0
    # Try to add primes to cover L+1
    # We want to maximize L.
    # Iterate: find best p to add that extends L.
    
    # Sort primes by size? 
    # Smaller primes cover more frequently, but cost more budget.
    # Let's just try all available primes for the current gap.
    
    while time.time() < t0 + 20:
        gap = L + 1
        best_p = None
        best_r = None
        best_new_L = L
        
        for p in primes:
            if p in primes_used:
                continue
            if current_sum + 1.0/p > 1.0:
                continue
            r = gap % p
            # Simulate add
            update(p, r, 1)
            new_L = get_L_from(L) # Since we only added coverage, L can only stay same or increase
            # Wait, if we add a class, we cover more. So L >= old L.
            # So get_L_from(L) is correct.
            
            if new_L > best_new_L:
                best_new_L = new_L
                best_p = p
                best_r = r
            
            # Rollback
            update(p, r, -1)
            
        if best_p is not None:
            cost = add_class(best_p, best_r)
            current_sum += cost
            L = best_new_L
        else:
            break
            
    best_L = L
    best_residues = dict(residues)
    best_primes = set(primes_used)
    best_sum = current_sum

    # Local Search
    # Try to improve L by changing residues or swapping primes.
    
    deadline = t0 + 110.0
    no_improve_count = 0
    
    while time.time() < deadline:
        # Current L
        # We maintain L, but let's verify it's correct
        # L = get_L_from(1) # Expensive if L is large? No, get_L_from(1) scans 1..MAX_L.
        # It's better to maintain L and update it.
        # But after changes, L might decrease.
        # So we must recalculate L from 1 if we do a move that might break coverage.
        
        # Strategy:
        # 1. Try to extend L by covering L+1.
        # 2. If stuck, perturb.
        
        improved = False
        
        # Phase 1: Greedy Extension
        # Try to find a move that increases L.
        
        # Option A: Change residue of existing prime to cover L+1
        # Option B: Add new prime to cover L+1
        # Option C: Swap prime to cover L+1 (if budget tight)
        
        candidates = []
        
        # A: Change residue
        for p in list(residues.keys()):
            r_new = (L + 1) % p
            if residues[p] == r_new:
                continue
            # Simulate
            r_old = residues[p]
            change_residue(p, r_new)
            new_L = get_L_from(1) # Must scan from 1 because we might have broken coverage elsewhere
            change_residue(p, r_old) # Rollback
            
            if new_L > L:
                candidates.append((new_L, 'change', p, r_new))
                
        # B: Add new prime
        for p in primes:
            if p in primes_used:
                continue
            if current_sum + 1.0/p > 1.0:
                continue
            r_new = (L + 1) % p
            # Simulate
            update(p, r_new, 1)
            new_L = get_L_from(L) # Adding coverage only extends L
            update(p, r_new, -1) # Rollback
            
            if new_L > L:
                candidates.append((new_L, 'add', p, r_new))
                
        # C: Swap (remove one, add one)
        # This is expensive. Only do if A and B fail.
        
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_L_new, op, p, r = candidates[0]
            
            if op == 'change':
                change_residue(p, r)
            elif op == 'add':
                cost = add_class(p, r)
                current_sum += cost
            
            L = best_L_new
            improved = True
            
            if L > best_L:
                best_L = L
                best_residues = dict(residues)
                best_primes = set(primes_used)
                best_sum = current_sum
            no_improve_count = 0
        else:
            # No greedy improvement found
            no_improve_count += 1
            
            if no_improve_count > 10:
                # Perturb
                # Randomly change a residue
                if residues:
                    p_rand = random.choice(list(residues.keys()))
                    r_rand = random.randrange(p_rand)
                    change_residue(p_rand, r_rand)
                    # Recalculate L
                    L = get_L_from(1)
                    no_improve_count = 0
                else:
                    break

    # Final cleanup
    res = []
    for p, r in best_residues.items():
        res.append([p, r])
        
    if not res:
        res = [[2, 1]]
        
    return res

if __name__ == "__main__":
    r = construct()
    from math import gcd
    n_, d_ = 0, 1
    for p, rr in r:
        g = gcd(d_, p)
        n_ = n_ * (p // g) + (d_ // g)
        d_ = d_ * (p // g)
        d_ //= g
    print("num classes", len(r), "sum", n_, "/", d_)
