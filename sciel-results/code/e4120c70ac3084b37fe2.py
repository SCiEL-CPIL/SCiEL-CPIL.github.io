import numpy as np
import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: 110 seconds
    deadline = start_time + 110.0
    
    # The problem specifies radius 1 logic ("at most one position")
    # If radius != 1, this specific neighbor generation is wrong.
    # However, the task contract says "at most one position".
    # I will assume radius == 1.
    
    N = q ** n
    
    # Helper functions
    def to_digits(x, n, q):
        digits = [0] * n
        for i in range(n):
            digits[i] = x % q
            x //= q
        return digits

    def to_int(digits, q):
        x = 0
        for i in range(len(digits)-1, -1, -1):
            x = x * q + digits[i]
        return x

    # Precompute neighbors for radius 1
    # If n=7, q=3, radius=1, ball size is 1 + 7*2 = 15
    # General ball size for R=1 is 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)
    
    neighbors = []
    for i in range(N):
        digits = to_digits(i, n, q)
        neigh = [i]
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_digits = digits.copy()
                new_digits[pos] = new_val
                new_val_int = to_int(new_digits, q)
                neigh.append(new_val_int)
        neighbors.append(neigh)
        
    Neigh = np.array(neighbors, dtype=np.int32)
    
    best_code = None
    best_size = N # Worst case
    
    # Run restarts
    # We can run as many as fit in time
    # Estimate time per restart:
    # ~200 iterations * (matrix mult + overhead)
    # Matrix mult is fast. Overhead is Python loop.
    # Let's try 100 restarts first, then increase if time allows.
    
    max_restarts = 200
    
    for r in range(max_restarts):
        if time.time() > deadline - 5.0:
            break
            
        uncovered = np.ones(N, dtype=np.int32)
        code = []
        
        while np.sum(uncovered) > 0:
            # Calculate marginal gain
            # Neigh is (N, ball_size)
            # uncovered is (N,)
            # gains is (N,)
            gains = Neigh @ uncovered
            
            max_gain = np.max(gains)
            
            if max_gain == 0:
                # Should not happen if uncovered > 0
                idxs = np.where(uncovered > 0)[0]
                if len(idxs) == 0:
                    break
                i = idxs[0]
            else:
                candidates = np.where(gains == max_gain)[0]
                if len(candidates) > 1:
                    # Random tie-breaking to explore different solutions
                    i = candidates[random.randint(0, len(candidates)-1)]
                else:
                    i = candidates[0]
            
            code.append(i)
            
            # Mark neighbors as covered
            uncovered[Neigh[i]] = 0
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        
    final_code = [to_digits(c, n, q) for c in best_code]
    return final_code
