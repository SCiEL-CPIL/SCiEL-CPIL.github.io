import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    # Use 145s to be safe against overhead
    deadline = start_time + 145.0
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    num_words = len(all_words)
    
    # Precompute neighbors for radius=1 (specific to this task's known params)
    # If radius is different, this needs to change. 
    # Given the task title, radius=1 is expected.
    if radius != 1:
        # Fallback to a simpler greedy if radius is not 1, or just handle radius 1
        # For now, assume radius 1 as per task description "K_3(7,1)"
        pass
        
    neighbors = [[] for _ in range(num_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                new_w = tuple(new_w)
                neighbors[i].append(word_to_idx[new_w])
        
    # Initial Code: Tetracode construction
    tetracode = []
    for a in range(3):
        for b in range(3):
            tetracode.append((a, b, (a+b)%3, (a+2*b)%3))
            
    initial_code_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n-4):
            full_word = h + tail
            initial_code_indices.append(word_to_idx[full_word])
            
    code = set(initial_code_indices)
    
    cover_count = [0] * num_words
    for c in code:
        for w in neighbors[c]:
            cover_count[w] += 1
            
    best_code = list(code)
    best_size = len(best_code)
    
    def is_removable(c):
        for w in neighbors[c]:
            if cover_count[w] <= 1:
                return False
        return True
        
    def remove_redundant():
        nonlocal code, cover_count, best_code, best_size
        changed = True
        while changed:
            changed = False
            for c in list(code):
                if is_removable(c):
                    code.remove(c)
                    for w in neighbors[c]:
                        cover_count[w] -= 1
                    changed = True
                    if len(code) < best_size:
                        best_size = len(code)
                        best_code = list(code)
                        
    remove_redundant()
    
    while time.time() < deadline:
        k = random.randint(2, 10)
        if len(code) <= k:
            continue
            
        to_remove = random.sample(list(code), k)
        for c in to_remove:
            code.remove(c)
            for w in neighbors[c]:
                cover_count[w] -= 1
                
        uncovered = [w for w in range(num_words) if cover_count[w] == 0]
        
        while uncovered:
            best_c = -1
            best_gain = -1
            for c in range(num_words):
                if c in code:
                    continue
                gain = 0
                for w in neighbors[c]:
                    if cover_count[w] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
                    
            if best_c == -1:
                break
                
            code.add(best_c)
            for w in neighbors[best_c]:
                cover_count[w] += 1
                
            uncovered = [w for w in range(num_words) if cover_count[w] == 0]
            
        remove_redundant()
        
    return [list(all_words[i]) for i in best_code]
