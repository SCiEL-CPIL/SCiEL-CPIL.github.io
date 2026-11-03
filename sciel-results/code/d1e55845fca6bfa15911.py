"""Improved ternary covering code for K_3(8,1) using structured init + greedy + local search."""
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters
        return _generic_construct(q, n, radius, start_time, deadline)
    
    # Step 1: Structured initialization - tetracode product
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in product(range(3), repeat=4)]
    
    # Convert to numpy for faster operations
    code_arr = np.array(initial_code, dtype=np.int8)
    best_code = code_arr.copy()
    best_count = len(best_code)
    
    # Precompute: for each codeword, its ball (17 words)
    def compute_ball_indices(cw, n=8, q=3):
        """Return list of all words within distance 1 of cw."""
        ball = [list(cw)]
        for i in range(n):
            for d in range(q):
                if d != cw[i]:
                    w = list(cw)
                    w[i] = d
                    ball.append(w)
        return ball
    
    # Encode words as integers for fast lookup
    def encode_word(w, n=8, q=3):
        val = 0
        for i in range(n):
            val = val * q + w[i]
        return val
    
    def decode_word(val, n=8, q=3):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = val % q
            val //= q
        return w
    
    # Build coverage structure
    def build_coverage(code_arr, n=8, q=3):
        """Return a set of encoded words that are covered."""
        covered = set()
        for cw in code_arr:
            ball = compute_ball_indices(cw, n, q)
            for w in ball:
                covered.add(encode_word(w, n, q))
        return covered
    
    def is_covering(code_arr, n=8, q=3):
        covered = build_coverage(code_arr, n, q)
        return len(covered) == q ** n
    
    # Greedy reduction: remove codewords whose ball is fully covered by others
    def greedy_reduce(code_arr, n=8, q=3):
        if time.time() > deadline:
            return code_arr
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            new_code = list(code_arr)
            for idx in range(len(new_code) - 1, -1, -1):
                if time.time() > deadline:
                    break
                # Check if removing new_code[idx] still covers everything
                candidate = [c for i, c in enumerate(new_code) if i != idx]
                if is_covering(np.array(candidate, dtype=np.int8), n, q):
                    new_code.pop(idx)
                    changed = True
                    best_count = len(new_code)
            code_arr = np.array(new_code, dtype=np.int8)
        return code_arr
    
    # Local search: try swapping one codeword for another, or two for one
    def local_search(code_arr, n=8, q=3, max_iterations=5000):
        nonlocal best_code, best_count
        current = list(code_arr)
        current_count = len(current)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Try removing a codeword if still covered
            if current_count > 1:
                idx = random.randint(0, len(current) - 1)
                candidate = [c for i, c in enumerate(current) if i != idx]
                if is_covering(np.array(candidate, dtype=np.int8), n, q):
                    current = candidate
                    current_count -= 1
                    if current_count < best_count:
                        best_count = current_count
                        best_code = np.array(current, dtype=np.int8)
                    continue
            
            # Try replacing one codeword with a random neighbor
            if current_count >= 1:
                idx = random.randint(0, len(current) - 1)
                old_cw = current[idx]
                # Generate a random word at distance 1 or 2 from old_cw
                new_cw = list(old_cw)
                for _ in range(random.randint(1, 2)):
                    pos = random.randint(0, n - 1)
                    new_val = random.choice([v for v in range(q) if v != new_cw[pos]])
                    new_cw[pos] = new_val
                
                candidate = current.copy()
                candidate[idx] = new_cw
                if is_covering(np.array(candidate, dtype=np.int8), n, q):
                    current = candidate
                    if current_count < best_count:
                        best_count = current_count
                        best_code = np.array(current, dtype=np.int8)
                    continue
            
            # Try replacing two codewords with one new codeword
            if current_count >= 2:
                idx1 = random.randint(0, len(current) - 1)
                idx2 = random.randint(0, len(current) - 1)
                if idx1 == idx2:
                    continue
                
                # Generate a new codeword that might cover both balls
                # Try the midpoint or a random word
                new_cw = list(current[idx1])
                # Perturb it
                for _ in range(random.randint(0, 2)):
                    pos = random.randint(0, n - 1)
                    new_val = random.choice([v for v in range(q) if v != new_cw[pos]])
                    new_cw[pos] = new_val
                
                candidate = [c for i, c in enumerate(current) if i != idx1 and i != idx2]
                candidate.append(new_cw)
                if is_covering(np.array(candidate, dtype=np.int8), n, q):
                    current = candidate
                    current_count -= 1
                    if current_count < best_count:
                        best_count = current_count
                        best_code = np.array(current, dtype=np.int8)
        
        return best_code
    
    # Run greedy reduction first
    best_code = greedy_reduce(best_code, n, q)
    
    # Run local search
    best_code = local_search(best_code, n, q)
    
    # Final check and return
    result = [list(row) for row in best_code]
    return result


def _generic_construct(q, n, radius, start_time, deadline):
    """Fallback for non-standard parameters."""
    # Simple greedy construction
    code = []
    uncovered = set(product(range(q), repeat=n))
    
    while uncovered and time.time() < deadline:
        # Pick random uncovered word, add its covering ball center
        w = random.choice(list(uncovered))
        code.append(list(w))
        # Remove all words within radius of w
        to_remove = set()
        for v in product(range(q), repeat=n):
            dist = sum(1 for a, b in zip(v, w) if a != b)
            if dist <= radius:
                to_remove.add(v)
        uncovered -= to_remove
    
    return code
