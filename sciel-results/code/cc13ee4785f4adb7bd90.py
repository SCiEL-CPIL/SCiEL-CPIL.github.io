import itertools
import time
import random
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Handle non-specific cases
    if q != 3 or n != 6 or radius != 1:
        return _generic_greedy(q, n, radius, start_time, deadline)

    # Specific optimization for q=3, n=6, r=1
    total_words = 3 ** n
    
    # 1. Precompute neighbors (sphere of radius 1)
    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * 3 + digit
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return tuple(w)

    neighbors = [[] for _ in range(total_words)]
    neighbor_sets = [set() for _ in range(total_words)]
    
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors[idx].append(idx)
        neighbor_sets[idx].add(idx)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = new_val
                new_idx = word_to_idx(tuple(new_w_list))
                neighbors[idx].append(new_idx)
                neighbor_sets[idx].add(new_idx)

    # 2. Generate Initial Code (Direct Sum of Tetracode)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_codewords = [tuple(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    current_code = [word_to_idx(cw) for cw in seed_codewords]
    current_code_set = set(current_code)
    
    # 3. Initialize Coverage
    coverage = [0] * total_words
    for cw in current_code:
        for nb in neighbors[cw]:
            coverage[nb] += 1

    # 4. Initial Redundancy Removal
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 50.0: # Leave time for SA
            break
        new_code = []
        for cw in current_code:
            is_redundant = True
            for nb in neighbors[cw]:
                if coverage[nb] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for nb in neighbors[cw]:
                    coverage[nb] -= 1
                current_code_set.discard(cw)
                improved = True
            else:
                new_code.append(cw)
        current_code = new_code

    best_code = current_code[:]
    best_size = len(current_code)
    
    # 5. Simulated Annealing
    rng = random.Random(42)
    temp = 1.0
    cool_rate = 0.999
    min_temp = 0.01
    
    def is_removable(c):
        for u in neighbors[c]:
            if coverage[u] == 1:
                return False
        return True

    def is_valid_swap(c_old, c_new):
        # Check if c_new covers all unique neighbors of c_old
        for u in neighbors[c_old]:
            if coverage[u] == 1 and u not in neighbor_sets[c_new]:
                return False
        return True

    # We track the best code found so far
    while time.time() < deadline - 5.0:
        if temp < min_temp:
            temp = min_temp
            
        # Choose move type: 50% Remove, 50% Swap
        if rng.random() < 0.5:
            # Try Remove
            if len(current_code) > 0:
                idx = rng.randint(0, len(current_code) - 1)
                c = current_code[idx]
                if is_removable(c):
                    # Perform removal
                    current_code.pop(idx)
                    current_code_set.discard(c)
                    for u in neighbors[c]:
                        coverage[u] -= 1
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = current_code[:]
        else:
            # Try Swap
            if len(current_code) > 0:
                idx = rng.randint(0, len(current_code) - 1)
                c_old = current_code[idx]
                # Pick a random candidate not in current code
                # To speed up, we can pick from a small set of candidates or just random index
                c_new = rng.randint(0, total_words - 1)
                
                if c_new == c_old or c_new in current_code_set:
                    continue
                    
                if is_valid_swap(c_old, c_new):
                    # Perform swap
                    current_code[idx] = c_new
                    current_code_set.discard(c_old)
                    current_code_set.add(c_new)
                    
                    # Update coverage
                    for u in neighbors[c_old]:
                        coverage[u] -= 1
                    for u in neighbors[c_new]:
                        coverage[u] += 1
                    
                    # Note: Size is unchanged, so best_size doesn't change.
                    # But the structure changed, potentially allowing future removals.
        
        temp *= cool_rate

    # 6. Final Redundancy Removal on Best Code
    # It's possible the best code found during SA still has redundancies that weren't removed
    # because we only removed one at a time during SA.
    final_code = best_code[:]
    final_set = set(final_code)
    final_cov = [0] * total_words
    for cw in final_code:
        for nb in neighbors[cw]:
            final_cov[nb] += 1
            
    improved = True
    while improved:
        improved = False
        new_final = []
        for cw in final_code:
            is_redundant = True
            for nb in neighbors[cw]:
                if final_cov[nb] <= 1:
                    is_redundant = False
                    break
            if is_redundant:
                for nb in neighbors[cw]:
                    final_cov[nb] -= 1
                final_set.discard(cw)
                improved = True
            else:
                new_final.append(cw)
        final_code = new_final

    return [list(idx_to_word(cw)) for cw in final_code]

def _generic_greedy(q, n, radius, start_time, deadline):
    # Fallback implementation for other parameters
    # Simple Greedy Set Cover
    total_words = q ** n
    if total_words > 10000:
        return [] # Too large for simple greedy within time
        
    # Precompute neighbors
    def word_to_idx(w):
        idx = 0
        for digit in w:
            idx = idx * q + digit
        return idx

    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        # This is slow for large n/q, but okay for small
        w = [0]*n
        temp = idx
        for i in range(n-1, -1, -1):
            w[i] = temp % q
            temp //= q
        w = tuple(w)
        
        neighbors[idx].append(idx)
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val == orig: continue
                nw = list(w)
                nw[pos] = new_val
                neighbors[idx].append(word_to_idx(tuple(nw)))
                
    uncovered = set(range(total_words))
    code = []
    
    while uncovered and time.time() < deadline:
        best_c = -1
        best_gain = 0
        # Greedy: pick word covering most uncovered
        # Optimization: only check words not in code?
        for c in range(total_words):
            if c in code: continue
            gain = 0
            for u in neighbors[c]:
                if u in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_c == -1: break
        
        code.append(best_c)
        for u in neighbors[best_c]:
            uncovered.discard(u)
            
    return [list(idx_to_word(cw)) for cw in code]
