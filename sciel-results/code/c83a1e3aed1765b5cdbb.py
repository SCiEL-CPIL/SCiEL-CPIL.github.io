import time
from math import gcd
from functools import reduce
import random

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    lcm_all = reduce(lcm, divisors)  # 30030
    
    # For a given W and L, check if valid assignment exists
    # Returns assignment list or None
    def check_window(L, W):
        # Build candidate positions for each divisor
        # For divisor d, position p in [0, W-1] is valid if (L+p) % d == 0
        # i.e., p ≡ (-L) % d (mod d)
        
        candidates = []
        for d in divisors:
            r = (-L) % d
            pos = []
            p = r
            while p < W:
                pos.append(p)
                p += d
            if not pos:
                return None
            candidates.append(pos)
        
        # Sort divisors by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        # Backtracking with forward checking
        used = [False] * W
        assignment = [None] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for p in candidates[i]:
                if not used[p]:
                    used[p] = True
                    assignment[i] = L + p
                    if backtrack(idx + 1):
                        return True
                    used[p] = False
                    assignment[i] = None
            return False
        
        if backtrack(0):
            return assignment
        return None
    
    # Search W from minimum upward
    for W in range(n - 1, 100):
        if time.time() > deadline:
            break
        
        # First pass: try L values in a deterministic but shuffled order
        # Use a fixed seed for reproducibility
        rng = random.Random(42)
        
        # Try all L from 1 to lcm_all, but in a shuffled order for early termination
        L_values = list(range(1, lcm_all + 1))
        rng.shuffle(L_values)
        
        found = False
        for L in L_values:
            if time.time() > deadline:
                break
            if L + W - 1 > MAX_V:
                continue
            
            result = check_window(L, W)
            if result is not None:
                # Verify
                if len(set(result)) != n:
                    continue
                for i, d in enumerate(divisors):
                    if result[i] % d != 0:
                        continue
                if any(v < 1 or v > MAX_V for v in result):
                    continue
                return [int(v) for v in result]
    
    # Fallback
    return [int(d) for d in divisors]
