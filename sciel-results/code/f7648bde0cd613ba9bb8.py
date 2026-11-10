import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Construct a ternary covering code of length n with radius r.
    For q=3, n=8, r=1, we aim to find a code with fewer than 486 codewords.
    """
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (shouldn't happen in this task)
        # Simple greedy construction
        N = q ** n
        neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
        for w in range(N):
            idx = 0
            neighbors[w, idx] = w
            idx += 1
            temp = w
            digits = [0] * n
            for i in range(n):
                digits[i] = temp % q
                temp //= q
            for i in range(n):
                d = digits[i]
                for new_d in range(q):
                    if new_d != d:
                        w1 = w + (new_d - d) * (q ** i)
                        neighbors[w, idx] = w1
                        idx += 1
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        while uncovered.sum() > 0:
            gains = uncovered[neighbors].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            w = candidates[np.random.randint(len(candidates))]
            code.append(w)
            uncovered[neighbors[w]] = 0
        return [[int(c) for c in np.base_repr(w, base=q).zfill(n)][::-1] for w in code]

    N = 3 ** 8  # 6561
    deadline = time.time() + 115.0  # Safety margin

    # Precompute neighbors for all N words
    # Shape: (N, 1 + 2*n) = (6561, 17)
    neighbors = np.zeros((N, 1 + 2 * n), dtype=np.int32)
    
    # Precompute powers of 3
    powers = [3 ** i for i in range(n)]
    
    for w in range(N):
        neighbors[w, 0] = w
        idx = 1
        temp = w
        for i in range(n):
            d = temp % 3
            temp //= 3
            # Change digit i to (d+1)%3
            w1 = w + ((d + 1) % 3 - d) * powers[i]
            neighbors[w, idx] = w1
            idx += 1
            # Change digit i to (d+2)%3
            w2 = w + ((d + 2) % 3 - d) * powers[i]
            neighbors[w, idx] = w2
            idx += 1

    # Seed solution: Tetracode direct sum with all 3^4 tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code_ints = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            # Convert to integer representation (LSB first in my neighbor precompute)
            w_int = 0
            for i in range(n):
                w_int += word[i] * powers[i]
            seed_code_ints.append(w_int)
    
    best_code = seed_code_ints[:]
    best_size = len(best_code)
    
    rng = np.random.default_rng(42)
    
    # Multiple random restarts of greedy algorithm
    while time.time() < deadline:
        uncovered = np.ones(N, dtype=np.uint8)
        code = []
        
        # Randomly shuffle the order of candidate selection to break ties differently
        # Or use a randomized greedy approach
        
        while uncovered.sum() > 0:
            # Calculate gain for each candidate
            # gain[w] = number of uncovered neighbors of w
            gains = uncovered[neighbors].sum(axis=1)
            
            max_gain = np.max(gains)
            
            if max_gain == 0:
                break
                
            # Find all words with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Pick one randomly from candidates
            w = rng.choice(candidates)
            
            code.append(w)
            
            # Mark neighbors as covered
            uncovered[neighbors[w]] = 0
            
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
            # If we reach the lower bound, we can stop
            if best_size <= 403:
                break

    # Convert integer codewords back to lists of integers
    final_code = []
    for w in best_code:
        word = []
        temp = w
        for _ in range(n):
            word.append(temp % 3)
            temp //= 3
        # word[0] is LSB (coordinate 0), word[1] is coordinate 1, etc.
        # This matches the order in the seed construction
        final_code.append(word)
        
    return final_code
