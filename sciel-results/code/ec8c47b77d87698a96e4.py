import time
import random
import itertools

Q = 3
N = 8
RADIUS = 1
TOTAL_WORDS = Q ** N # 6561

spheres = [None] * TOTAL_WORDS

def decode(word_int):
    return [word_int % (Q ** (N - i)) // (Q ** (N - i - 1)) for i in range(N)]

def encode(digits):
    val = 0
    for i, d in enumerate(reversed(digits)):
        val += d * (Q ** i)
    return val

# Precompute spheres
for i in range(TOTAL_WORDS):
    d = decode(i)
    sphere = [i]
    for pos in range(N):
        for new_val in range(Q):
            if new_val != d[pos]:
                new_d = d.copy()
                new_d[pos] = new_val
                sphere.append(encode(new_d))
    spheres[i] = sphere

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0
    
    if q != 3 or n != 8 or radius != 1:
        return [[0]*n] # Fallback
        
    best_code = None
    best_size = float('inf')
    
    # Initial code: 486 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            init_code.append(encode(h + list(tail)))
            
    current_code = init_code[:]
    
    # 1. Greedy reduce initial code
    current_code, _ = greedy_reduce(current_code, deadline)
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = current_code[:]
        
    # 2. Perturbation Loop
    while time.time() < deadline:
        # Pick a random codeword to remove
        if not current_code:
            break
            
        idx = random.randint(0, len(current_code) - 1)
        c_remove = current_code[idx]
        
        # Remove it
        coverage_count = [0] * TOTAL_WORDS
        for c in current_code:
            for w in spheres[c]:
                coverage_count[w] += 1
                
        # Actually, maintaining coverage_count globally is faster.
        # Let's maintain it outside the loop.
        
    # Let's restructure to maintain coverage_count globally.
    
    return [decode(c) for c in best_code]

# Corrected Implementation

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0
    
    if q != 3 or n != 8 or radius != 1:
        return [[0]*n]
        
    best_code = None
    best_size = float('inf')
    
    # Initial code: 486 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            init_code.append(encode(h + list(tail)))
            
    current_code = init_code[:]
    
    # Initialize coverage count
    coverage_count = [0] * TOTAL_WORDS
    for c in current_code:
        for w in spheres[c]:
            coverage_count[w] += 1
            
    # 1. Greedy reduce
    current_code = greedy_reduce_with_counts(current_code, coverage_count, deadline)
    if len(current_code) < best_size:
        best_size = len(current_code)
        best_code = current_code[:]
        
    # 2. Perturbation Loop
    while time.time() < deadline:
        if not current_code:
            break
            
        # Pick a random codeword to remove
        idx = random.randint(0, len(current_code) - 1)
        c_remove = current_code[idx]
        
        # Check if removal is safe (no word becomes uncovered)
        safe = True
        for w in spheres[c_remove]:
            if coverage_count[w] == 1:
                safe = False
                break
                
        if safe:
            # Remove it
            for w in spheres[c_remove]:
                coverage_count[w] -= 1
            current_code[idx] = current_code[-1]
            current_code.pop()
            
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code[:]
                
            # Continue greedy reduction
            current_code = greedy_reduce_with_counts(current_code, coverage_count, deadline)
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code[:]
        else:
            # Try to repair: Add a random codeword that covers a hole?
            # Or, swap: Remove c_remove, add c_new.
            # If we remove c_remove, holes appear.
            # We need to add codewords to cover holes.
            # Let's try adding a random codeword and see if it helps.
            # But we want to *reduce* size. So we shouldn't add.
            # Instead, we can try to *swap*: Remove c_remove, add c_new.
            # If the new code is valid, we have the same size.
            # Then we try to remove more.
            
            # Find holes
            holes = [w for w in spheres[c_remove] if coverage_count[w] == 1]
            if not holes:
                # This shouldn't happen if safe was False
                continue
                
            # Try to find a replacement c_new that covers all holes?
            # Or just pick a random c_new and check if code remains valid.
            # If we remove c_remove, coverage_count[w] becomes 0 for holes.
            # We need c_new to cover all holes.
            # Let's pick a random c_new.
            # Check if c_new covers all holes.
            # If yes, perform swap.
            
            # To speed up, we can pick c_new from the sphere of a random hole.
            hole = random.choice(holes)
            c_new = random.choice(spheres[hole])
            
            # Check if c_new covers all holes
            covers_all = True
            for h in holes:
                if h not in spheres[c_new]:
                    covers_all = False
                    break
                    
            if covers_all:
                # Perform swap
                # Remove c_remove
                for w in spheres[c_remove]:
                    coverage_count[w] -= 1
                # Add c_new
                for w in spheres[c_new]:
                    coverage_count[w] += 1
                    
                # Update code list
                current_code[idx] = c_new
                
                # Now try to remove more
                current_code = greedy_reduce_with_counts(current_code, coverage_count, deadline)
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = current_code[:]
            else:
                # Try a few more random c_new
                for _ in range(10):
                    c_new = random.choice(spheres[hole])
                    covers_all = True
                    for h in holes:
                        if h not in spheres[c_new]:
                            covers_all = False
                            break
                    if covers_all:
                        for w in spheres[c_remove]:
                            coverage_count[w] -= 1
                        for w in spheres[c_new]:
                            coverage_count[w] += 1
                        current_code[idx] = c_new
                        current_code = greedy_reduce_with_counts(current_code, coverage_count, deadline)
                        if len(current_code) < best_size:
                            best_size = len(current_code)
                            best_code = current_code[:]
                        break
                        
    if best_code is None:
        return [decode(c) for c in init_code]
        
    return [decode(c) for c in best_code]

def greedy_reduce_with_counts(code, coverage_count, deadline):
    i = 0
    while i < len(code):
        if time.time() > deadline:
            break
            
        c = code[i]
        redundant = True
        for w in spheres[c]:
            if coverage_count[w] == 1:
                redundant = False
                break
        
        if redundant:
            for w in spheres[c]:
                coverage_count[w] -= 1
            code[i] = code[-1]
            code.pop()
        else:
            i += 1
            
    return code
