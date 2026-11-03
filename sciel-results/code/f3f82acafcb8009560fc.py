import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    total_words = q ** n
    
    # Precompute all words as tuples
    all_words = []
    for i in range(total_words):
        word = []
        x = i
        for pos in range(n):
            word.append(x % q)
            x //= q
        all_words.append(tuple(word))
    
    # Build ball membership: for each position and value, list of indices
    # ball_membership[pos][val] = list of indices where word[pos] == val
    ball_membership = [[[] for _ in range(q)] for _ in range(n)]
    for i, w in enumerate(all_words):
        for pos in range(n):
            ball_membership[pos][w[pos]].append(i)
    
    # For radius=1, compute ball as: self + all words differing in exactly one position
    # Ball size = 1 + n*(q-1)
    ball_size = 1 + n * (q - 1) if radius == 1 else None
    
    # Precompute balls for all words (for radius=1 case)
    if radius == 1:
        balls = []
        for i, w in enumerate(all_words):
            ball = [i]
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    # Find all words that differ only at pos and have val at pos
                    # These are words where for all j != pos, word[j] = w[j], and word[pos] = val
                    # We can compute the index directly
                    # Actually, let's just iterate through the precomputed lists
                    for j in ball_membership[pos][val]:
                        # Check if j differs from i only at position pos
                        # This is expensive, let's use a different approach
                        pass
            balls.append(ball)
        
        # Better approach: for each word, compute its ball by direct index manipulation
        balls = []
        for i, w in enumerate(all_words):
            ball = {i}
            for pos in range(n):
                for val in range(q):
                    if val == w[pos]:
                        continue
                    # Compute the index of the word that is w but with pos set to val
                    # The index is base-q representation
                    new_idx = 0
                    multiplier = 1
                    for p in range(n):
                        if p == pos:
                            new_idx += val * multiplier
                        else:
                            new_idx += w[p] * multiplier
                        multiplier *= q
                    ball.add(new_idx)
            balls.append(list(ball))
    else:
        # For general radius, use Hamming distance
        balls = []
        for i, w in enumerate(all_words):
            ball = []
            for j, v in enumerate(all_words):
                dist = sum(1 for a, b in zip(w, v) if a != b)
                if dist <= radius:
                    ball.append(j)
            balls.append(ball)
    
    # Coverage tracking
    def compute_coverage(code_indices):
        cov = [0] * total_words
        for idx in code_indices:
            for nb in balls[idx]:
                cov[nb] += 1
        return cov
    
    def is_covered(cov):
        return all(c > 0 for c in cov)
    
    def count_uncovered(cov):
        return sum(1 for c in cov if c == 0)
    
    # Greedy set cover
    def greedy_cover():
        uncovered = [1] * total_words
        code_indices = []
        while any(uncovered):
            best_idx = -1
            best_count = 0
            for i in range(total_words):
                count = sum(1 for nb in balls[i] if uncovered[nb])
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            code_indices.append(best_idx)
            for nb in balls[best_idx]:
                uncovered[nb] = 0
        return code_indices
    
    # Randomized greedy
    def randomized_greedy():
        uncovered = [1] * total_words
        code_indices = []
        while any(uncovered):
            counts = []
            for i in range(total_words):
                counts.append(sum(1 for nb in balls[i] if uncovered[nb]))
            max_count = max(counts) if counts else 0
            candidates = [i for i, c in enumerate(counts) if c == max_count and c > 0]
            if not candidates:
                break
            best_idx = random.choice(candidates)
            code_indices.append(best_idx)
            for nb in balls[best_idx]:
                uncovered[nb] = 0
        return code_indices
    
    # Minimize by removing redundant codewords
    def minimize(code_indices):
        changed = True
        while changed:
            changed = False
            cov = compute_coverage(code_indices)
            for i in range(len(code_indices)):
                test_cov = cov[:]
                for nb in balls[code_indices[i]]:
                    test_cov[nb] -= 1
                if is_covered(test_cov):
                    code_indices = code_indices[:i] + code_indices[i+1:]
                    cov = test_cov
                    changed = True
                    break
        return code_indices
    
    # Local search: swap-based
    def local_search(initial_code, time_limit):
        current = minimize(initial_code[:])
        cov = compute_coverage(current)
        current_set = set(current)
        best = current[:]
        best_size = len(best)
        
        while time.time() < time_limit:
            if len(current) == 0:
                break
            
            i = random.randint(0, len(current) - 1)
            old = current[i]
            new = random.randint(0, total_words - 1)
            
            if new in current_set:
                continue
            
            # Compute tentative coverage
            test_cov = cov[:]
            for nb in balls[old]:
                test_cov[nb] -= 1
            for nb in balls[new]:
                test_cov[nb] += 1
            
            if not is_covered(test_cov):
                continue
            
            tent_code = current[:]
            tent_code[i] = new
            tent_code = minimize(tent_code)
            
            if len(tent_code) < len(current):
                current = tent_code
                cov = compute_coverage(current)
                current_set = set(current)
                if len(current) < best_size:
                    best = current[:]
                    best_size = len(best)
            elif len(tent_code) == len(current) and random.random() < 0.05:
                current = tent_code
                cov = compute_coverage(current)
                current_set = set(current)
        
        return best
    
    # Strategy 1: Greedy
    code1 = greedy_cover()
    code1 = minimize(code1)
    best_code = code1
    best_size = len(best_code)
    
    # Strategy 2: Randomized greedy (multiple runs)
    remaining_time = deadline - time.time()
    num_runs = max(1, int(remaining_time / 2.0))
    
    for run in range(num_runs):
        if time.time() > deadline - 5.0:
            break
        code = randomized_greedy()
        code = minimize(code)
        if len(code) < best_size:
            best_code = code
            best_size = len(best_code)
    
    # Strategy 3: Local search from best
    remaining = deadline - time.time()
    if remaining > 3.0:
        ls_result = local_search(best_code, deadline - 1.0)
        if len(ls_result) < best_size:
            best_code = ls_result
            best_size = len(best_code)
    
    # Final minimize
    best_code = minimize(best_code)
    
    # Convert to words
    return [list(all_words[idx]) for idx in best_code]
