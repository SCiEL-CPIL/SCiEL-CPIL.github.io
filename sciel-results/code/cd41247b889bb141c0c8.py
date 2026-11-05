import time
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    if q != 3 or n != 8 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    N = 3 ** 8
    coverage = [[] for _ in range(N)]
    
    # Precompute word to int and int to word is not needed for coverage if we work in int space directly
    # But we need to output words.
    
    # Let's precompute the coverage lists in int space
    for i in range(N):
        w = [0] * 8
        temp = i
        for j in range(8):
            w[j] = temp % 3
            temp //= 3
            
        covered = [i]
        for pos in range(8):
            for delta in [-1, 1]:
                new_val = (w[pos] + delta) % 3
                # Calculate new int
                # i + delta * 3^pos is not quite right because of modulo
                # Let's just do the conversion
                new_w = w[:]
                new_w[pos] = new_val
                new_i = 0
                for k in range(8):
                    new_i = new_i * 3 + new_w[k]
                covered.append(new_i)
        coverage[i] = covered

    covered = [False] * N
    remaining = N
    best_code = []
    
    while remaining > 0:
        if time.time() > deadline:
            # Fallback
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        best_gain = -1
        best_idx = -1
        
        for i in range(N):
            gain = 0
            for j in coverage[i]:
                if not covered[j]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_gain == 0:
            break
        
        # Convert best_idx to word
        w = [0] * 8
        temp = best_idx
        for j in range(8):
            w[j] = temp % 3
            temp //= 3
        best_code.append(w)
        
        for j in coverage[best_idx]:
            if not covered[j]:
                covered[j] = True
                remaining -= 1
        
    if remaining == 0:
        return best_code
    else:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
