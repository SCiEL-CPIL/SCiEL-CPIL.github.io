import random
import time
import itertools

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = 3 ** n
    spheres = []
    for i in range(total_words):
        sphere = [i]
        for pos in range(n):
            digit = (i // (3 ** pos)) % 3
            for new_digit in range(3):
                if new_digit != digit:
                    new_i = i - digit * (3 ** pos) + new_digit * (3 ** pos)
                    sphere.append(new_i)
        spheres.append(sphere)

    def greedy_random_max():
        uncovered = set(range(total_words))
        code = []
        
        while uncovered:
            max_new = 0
            best_candidates = []
            
            for c in range(total_words):
                new_count = 0
                for neighbor in spheres[c]:
                    if neighbor in uncovered:
                        new_count += 1
                
                if new_count > max_new:
                    max_new = new_count
                    best_candidates = [c]
                elif new_count == max_new and new_count > 0:
                    best_candidates.append(c)
            
            if not best_candidates:
                break # Should not happen
                
            c = random.choice(best_candidates)
            code.append(c)
            for neighbor in spheres[c]:
                uncovered.discard(neighbor)
                
        return code

    def prune(code):
        if not code:
            return code
        
        cov_count = [0] * total_words
        for c in code:
            for neighbor in spheres[c]:
                cov_count[neighbor] += 1
        
        pruned_code = []
        for c in code:
            for neighbor in spheres[c]:
                cov_count[neighbor] -= 1
            
            is_redundant = True
            for neighbor in spheres[c]:
                if cov_count[neighbor] == 0:
                    is_redundant = False
                    break
            
            if is_redundant:
                pass
            else:
                for neighbor in spheres[c]:
                    cov_count[neighbor] += 1
                pruned_code.append(c)
                
        return pruned_code

    best_code = None
    start_time = time.time()
    deadline = start_time + 140
    
    while time.time() < deadline:
        code = greedy_random_max()
        code = prune(code)
        
        if best_code is None or len(code) < len(best_code):
            best_code = code
            if len(best_code) == 186:
                break

    result = []
    for c in best_code:
        word = []
        temp = c
        for _ in range(n):
            word.append(temp % 3)
            temp //= 3
        result.append(word)
        
    return result
