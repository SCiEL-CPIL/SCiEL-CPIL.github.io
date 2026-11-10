import numpy as np

def construct():
    n = 256
    signs = [0] * n
    
    for i in range(n):
        # Find all APs ending at i
        # An AP is defined by start a, step d, length k
        # i = a + (k-1)d
        # So d must divide (i-a).
        # Since a >= 0 (0-indexed) and a < i, let's iterate d from 1 to i.
        # If d divides i? No.
        # Let's iterate d from 1 to i.
        # For a given d, a = i - (k-1)d.
        # Since a >= 0, i - (k-1)d >= 0 => (k-1)d <= i.
        # Also a >= 0 implies start index is valid.
        # We consider all k >= 1.
        
        candidates = [-1, 1]
        best_sign = 1
        best_metric = float('inf')
        
        # We need to evaluate the impact on APs ending at i.
        # Let's collect the previous sums for all APs ending at i.
        prev_sums = []
        
        for d in range(1, i + 1):
            # a = i - j*d for j = 1, 2, ... such that a >= 0
            # j is the number of steps from a to i.
            # Length k = j + 1.
            # a = i - j*d
            # We need a >= 0.
            # So j <= i/d.
            for j in range(1, int(i / d) + 1):
                a = i - j * d
                # Calculate sum of signs[a : i : d]
                # This is s[a] + s[a+d] + ... + s[i-d]
                # We can compute this sum.
                s = 0
                idx = a
                while idx < i:
                    s += signs[idx]
                    idx += d
                prev_sums.append(s)
                
        if not prev_sums:
            signs[i] = 1
            continue
            
        for c in candidates:
            max_abs = 0
            for ps in prev_sums:
                val = abs(ps + c)
                if val > max_abs:
                    max_abs = val
            if max_abs < best_metric:
                best_metric = max_abs
                best_sign = c
            elif max_abs == best_metric:
                # Tie-breaker: maybe minimize sum of squares?
                # Or just keep first
                pass
                
        signs[i] = best_sign
        
    return signs
