import time
import random
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    MAX_L = 100000
    covered = np.zeros(MAX_L + 1, dtype=bool)
    
    def get_L(covered_arr):
        # Find first False
        # np.where returns indices of True. We want first False.
        # Invert: ~covered_arr
        # Find first index where ~covered_arr is True
        # If all True, return MAX_L
        if not (~covered_arr).any():
            return MAX_L
        return int(np.where(~covered_arr)[0][0])
        
    def apply_residue(covered_arr, p, r):
        # Set covered_arr[r::p] = True
        # Note: r is 0-indexed in numpy, but our residues are 0 <= r < p.
        # If r == 0, it covers p, 2p, ... (indices p, 2p...)
        # Wait, the problem says "integer 1..L".
        # Class [p, r] covers { k | k % p == r }.
        # If r=0, it covers p, 2p, ...
        # If r=1, it covers 1, 1+p, ...
        # In numpy, indices are 0..MAX_L.
        # Integer n corresponds to index n.
        # So we need to set index n to True if n % p == r.
        # This is exactly covered_arr[r::p] = True.
        # BUT, if r=0, index 0 is set. We care about 1..L.
        # Setting index 0 to True doesn't hurt, but we must ensure 1..L are checked.
        # get_L checks ~covered_arr. If index 0 is False, it returns 0.
        # But L is defined as "largest L such that every integer 1..L lies in some class".
        # So index 0 is irrelevant. We should ensure index 0 is True or ignored.
        # Let's set covered_arr[0] = True always to avoid L=0.
        covered_arr[0] = True
        covered_arr[r::p] = True
        
    def revert_residue(covered_arr, p, r):
        # This is tricky because we don't know if OTHER primes cover those spots.
        # So we cannot simply set them to False.
        # We must recalculate coverage from scratch or maintain counts.
        # Given the complexity, "Revert" is expensive.
        # Better strategy:
        # Keep a list of (p, r) pairs.
        # To evaluate a move, we can't easily simulate without full rebuild.
        
        # Alternative Local Search:
        # 1. Start with a valid configuration.
        # 2. Identify first gap n.
        # 3. For each p, calculate r_new = n % p.
        # 4. Create a TEMPORARY covered array? Too slow.
        
        # Let's use a different approach:
        # Maintain `counts` array: counts[i] = number of primes covering i.
        # Then `covered` is counts > 0.
        # Applying residue:
        #   counts[r::p] += 1
        # Reverting residue:
        #   counts[r::p] -= 1
        # This is O(N/p) per move.
        # N=100,000. p=2 -> 50,000 ops. p=997 -> 100 ops.
        # Average p is small? We have many small primes.
        # If we have 20 primes, average p ~ 50.
        # 20 * 2000 = 40,000 ops per move.
        # 1000 moves -> 40,000,000 ops.
        # In Python/NumPy, this is fast.
        
        pass

    # Refined Local Search with Counts
    def optimize(prime_set):
        # Check budget
        s = sum(Fraction(1, p) for p in prime_set)
        if s > 1:
            return 0, []
            
        P_sorted = sorted(prime_set)
        residues = {p: 0 for p in P_sorted}
        
        counts = np.zeros(MAX_L + 1, dtype=np.int32)
        
        # Initial application
        for p in P_sorted:
            r = residues[p]
            counts[r::p] += 1
            
        counts[0] = 10 # Ensure 0 is covered, though irrelevant
        
        def get_L_from_counts(c):
            # Find first index i >= 1 where c[i] == 0
            # c is numpy array
            # We want min(i) for i in 1..MAX_L such that c[i]==0
            # If none, MAX_L
            # Use np.where(c[1:] == 0)[0]
            # If empty, return MAX_L
            # Else return index + 1
            zero_indices = np.where(c[1:] == 0)[0]
            if len(zero_indices) == 0:
                return MAX_L
            return int(zero_indices[0]) + 1

        current_L = get_L_from_counts(counts)
        
        # Local Search Loop
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < deadline - 5:
            if current_L == MAX_L:
                break
                
            n = current_L # First uncovered
            best_move = None
            best_L = current_L
            
            # Try changing each prime's residue to cover n
            for p in P_sorted:
                r_new = n % p
                if residues[p] == r_new:
                    continue
                
                r_old = residues[p]
                
                # Apply move
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                
                new_L = get_L_from_counts(counts)
                
                # Revert move
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                
                if new_L > best_L:
                    best_L = new_L
                    best_move = (p, r_old, r_new)
                    
            if best_move:
                p, r_old, r_new = best_move
                residues[p] = r_new
                counts[r_old::p] -= 1
                counts[r_new::p] += 1
                current_L = best_L
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
                    
        classes = [[p, residues[p]] for p in P_sorted]
        return current_L, classes

    # Generate Candidates
    candidates = []
    
    # 1. Greedy Small Primes
    def build_greedy_small():
        s = Fraction(0)
        res = []
        for p in primes:
            if s + Fraction(1, p) <= 1:
                res.append(p)
                s += Fraction(1, p)
        return res
        
    # 2. Greedy Large Primes
    def build_greedy_large():
        s = Fraction(0)
        res = []
        for p in reversed(primes):
            if s + Fraction(1, p) <= 1:
                res.append(p)
                s += Fraction(1, p)
        return res
        
    # 3. Random Sets
    random.seed(42)
    for _ in range(20):
        s = Fraction(0)
        res = []
        # Pick random primes, ensure distinct
        available = list(primes)
        random.shuffle(available)
        for p in available:
            if s + Fraction(1, p) <= 1:
                res.append(p)
                s += Fraction(1, p)
        candidates.append(res)
        
    candidates.append(build_greedy_small())
    candidates.append(build_greedy_large())
    
    best_L = 0
    best_classes = []
    
    for P in candidates:
        if time.time() > deadline - 10:
            break
        if not P:
            continue
        L, classes = optimize(P)
        if L > best_L:
            best_L = L
            best_classes = classes
            
    return best_classes
