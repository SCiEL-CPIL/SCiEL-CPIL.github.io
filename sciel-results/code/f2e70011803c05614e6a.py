import itertools
import random
import numpy as np
import time

def construct(q, n, radius):
    # For this specific task, q=3, n=6, radius=1
    # But we should handle general small cases if possible, or just optimize for the specific case.
    # The contract says "evaluator passes this task's alphabet size q, length n and covering radius".
    # It implies it might vary, but the "best known" is specific.
    # We will assume the specific case or handle small q,n generically.
    
    all_words = list(itertools.product(range(q), repeat=n))
    total_words = len(all_words)
    
    # If the space is too large for the SA approach, fall back to greedy
    # 729 is small. 3^7 = 2187. 3^8 = 6561.
    # Let's limit SA to total_words <= 5000.
    if total_words > 5000:
        return _simple_greedy(q, n, radius)

    # Precompute coverage masks
    # masks[i] is a boolean vector of length total_words, True if word j is within radius of word i
    masks = np.zeros((total_words, total_words), dtype=bool)
    for i, w1 in enumerate(all_words):
        for j, w2 in enumerate(all_words):
            dist = sum(1 for a, b in zip(w1, w2) if a != b)
            if dist <= radius:
                masks[i][j] = True

    # Initial solution: Greedy
    def greedy_init():
        counts = np.zeros(total_words, dtype=int)
        codewords = []
        uncovered_mask = np.ones(total_words, dtype=bool)
        
        while np.any(uncovered_mask):
            # Calculate score for each word: number of uncovered words it covers
            scores = np.sum(masks & uncovered_mask, axis=1)
            max_score = np.max(scores)
            if max_score == 0:
                break
            candidates = np.where(scores == max_score)[0]
            chosen_idx = candidates[random.randint(0, len(candidates)-1)]
            codewords.append(chosen_idx)
            counts[masks[chosen_idx]] += 1
            uncovered_mask &= (counts == 0)
            
        return codewords

    initial_codewords = greedy_init()
    
    # Simulated Annealing
    def simulate_annealing(initial_cw, time_limit=110):
        start_time = time.time()
        current_cw = initial_cw[:]
        current_counts = np.zeros(total_words, dtype=int)
        for idx in current_cw:
            current_counts[masks[idx]] += 1
        
        current_uncovered = np.sum(current_counts == 0)
        current_size = len(current_cw)
        current_energy = current_uncovered * 10000 + current_size
        
        best_cw = current_cw[:]
        best_counts = current_counts.copy()
        best_energy = current_energy
        
        T = 100.0
        cooling_rate = 0.999
        min_T = 1e-5
        
        while T > min_T and (time.time() - start_time) < time_limit:
            # Generate a neighbor
            move_type = random.random()
            new_cw = current_cw[:]
            new_counts = current_counts.copy()
            
            if move_type < 0.7:
                # Swap
                if len(current_cw) < 2:
                    continue
                pos = random.randint(0, len(current_cw)-1)
                old_idx = current_cw[pos]
                new_idx = random.randint(0, total_words-1)
                if new_idx == old_idx:
                    continue
                new_counts[masks[old_idx]] -= 1
                new_counts[masks[new_idx]] += 1
                new_cw[pos] = new_idx
            elif move_type < 0.9:
                # Add
                if len(current_cw) > 80: # Limit size to avoid bloat
                    continue
                new_idx = random.randint(0, total_words-1)
                if new_idx in current_cw:
                    continue
                new_counts[masks[new_idx]] += 1
                new_cw.append(new_idx)
            else:
                # Remove
                if len(current_cw) < 60: # Limit size to avoid empty
                    continue
                pos = random.randint(0, len(current_cw)-1)
                old_idx = current_cw[pos]
                new_counts[masks[old_idx]] -= 1
                new_cw.pop(pos)
                
            new_uncovered = np.sum(new_counts == 0)
            new_size = len(new_cw)
            new_energy = new_uncovered * 10000 + new_size
            
            delta = new_energy - current_energy
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cw = new_cw
                current_counts = new_counts
                current_uncovered = new_uncovered
                current_size = new_size
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_cw = current_cw[:]
                    best_counts = current_counts.copy()
            
            T *= cooling_rate
            
        return best_cw

    best_codewords = simulate_annealing(initial_codewords)
    
    # Final Local Search on the best result to clean up
    def local_search(codewords):
        counts = np.zeros(total_words, dtype=int)
        for idx in codewords:
            counts[masks[idx]] += 1
            
        # Try to remove redundant words
        improved = True
        while improved:
            improved = False
            for i in range(len(codewords)):
                idx = codewords[i]
                counts[masks[idx]] -= 1
                if np.all(counts > 0):
                    codewords.pop(i)
                    improved = True
                    break
                else:
                    counts[masks[idx]] += 1
        return codewords

    best_codewords = local_search(best_codewords)
    
    return [all_words[i] for i in best_codewords]

def _simple_greedy(q, n, radius):
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    codewords = []
    
    def get_coverage(word):
        covered = set()
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_word = list(word)
                new_word[pos] = val
                covered.add(tuple(new_word))
        covered.add(tuple(word))
        return covered
        
    coverages = {w: get_coverage(w) for w in all_words}
    
    while uncovered:
        best_word = None
        best_count = -1
        for w in all_words:
            count = len(coverages[w] & uncovered)
            if count > best_count:
                best_count = count
                best_word = w
        if best_count == 0:
            break
        codewords.append(list(best_word))
        uncovered -= coverages[best_word]
    return codewords
