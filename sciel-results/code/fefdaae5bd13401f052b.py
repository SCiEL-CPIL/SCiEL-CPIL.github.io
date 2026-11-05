import random
import time

# Global precomputation for the specific problem instance q=3, n=7
Q_GLOBAL = 3
N_GLOBAL = 7
TOTAL_WORDS = Q_GLOBAL ** N_GLOBAL
neighbors = []
for i in range(TOTAL_WORDS):
    digits = []
    x = i
    for _ in range(N_GLOBAL):
        digits.append(x % Q_GLOBAL)
        x //= Q_GLOBAL
    digits = tuple(digits)
    
    neigh = [i]
    for pos in range(N_GLOBAL):
        for val in range(Q_GLOBAL):
            if val != digits[pos]:
                new_digits = list(digits)
                new_digits[pos] = val
                new_idx = 0
                for d in new_digits:
                    new_idx = new_idx * Q_GLOBAL + d
                neigh.append(new_idx)
    neighbors.append(neigh)

def construct(q, n, radius):
    if q != Q_GLOBAL or n != N_GLOBAL or radius != 1:
        # Fallback for other parameters if ever called, though task is specific
        # Simple greedy for general case
        total = q ** n
        # ... (implementation omitted for brevity in thought, but should be robust)
        # Since the task is specific, we can just return a valid large code or raise
        # To be safe, let's implement a very simple generic greedy
        def dist(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)
        
        covered = [False] * total
        code = []
        while True:
            # Find uncovered
            uncovered = [i for i in range(total) if not covered[i]]
            if not uncovered:
                break
            # Pick random uncovered
            u_idx = random.choice(uncovered)
            # Convert to digits
            u_digits = []
            x = u_idx
            for _ in range(n):
                u_digits.append(x % q)
                x //= q
            u_digits = tuple(u_digits)
            code.append(u_digits)
            # Update covered
            for i in range(total):
                i_digits = []
                x = i
                for _ in range(n):
                    i_digits.append(x % q)
                    x //= q
                if dist(u_digits, tuple(i_digits)) <= radius:
                    covered[i] = True
        return code

    start_time = time.time()
    time_limit = 140.0
    
    best_code = None
    best_size = float('inf')
    
    while time.time() - start_time < time_limit:
        code = greedy_cover()
        code = local_improve(code)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
            
        if best_size <= 156:
            break

    result = []
    for idx in best_code:
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % q)
            x //= q
        result.append(digits)
        
    return result

def greedy_cover():
    covered = [False] * TOTAL_WORDS
    code = []
    uncovered_count = TOTAL_WORDS
    
    while uncovered_count > 0:
        M = 100
        candidates = []
        attempts = 0
        while len(candidates) < M and attempts < 1000:
            w = random.randint(0, TOTAL_WORDS - 1)
            if not covered[w]:
                candidates.append(w)
            attempts += 1
        
        if not candidates:
            # Fallback
            for w in range(TOTAL_WORDS):
                if not covered[w]:
                    candidates.append(w)
                    break
            if not candidates:
                break

        best_candidate = None
        max_gain = -1
        
        for w in candidates:
            gain = 0
            for nb in neighbors[w]:
                if not covered[nb]:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_candidate = w
                
        if best_candidate is None:
            break
            
        code.append(best_candidate)
        
        for nb in neighbors[best_candidate]:
            if not covered[nb]:
                covered[nb] = True
                uncovered_count -= 1
                
    return code

def local_improve(code):
    cover_count = [0] * TOTAL_WORDS
    for c in code:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    changed = True
    while changed:
        changed = False
        to_remove = []
        for i in range(len(code)):
            c = code[i]
            redundant = True
            for nb in neighbors[c]:
                if cover_count[nb] <= 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
                
        if to_remove:
            to_remove.sort(reverse=True)
            for idx in to_remove:
                c = code[idx]
                for nb in neighbors[c]:
                    cover_count[nb] -= 1
                del code[idx]
            changed = True
            
    return code
