"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np


def _generate_ball(word, q, n, radius):
    """Generate all words within Hamming distance `radius` from `word`."""
    ball = [list(word)]
    for r in range(1, radius + 1):
        for positions in itertools.combinations(range(n), r):
            for values in itertools.product(range(1, q), repeat=r):
                new_word = list(word)
                for pos, val in zip(positions, values):
                    new_word[pos] = (word[pos] + val) % q
                ball.append(new_word)
    return ball


def _hamming_distance(w1, w2):
    """Compute Hamming distance between two words."""
    return sum(1 for a, b in zip(w1, w2) if a != b)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 120  # Leave safety margin before 150s kill
    
    # Precompute all possible words
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    total_words = len(all_words)
    
    # Precompute balls for all words
    balls = {}
    for w in all_words:
        balls[tuple(w)] = _generate_ball(w, q, n, radius)
    
    # Greedy set-covering: start with random seed, then iteratively add/remove
    best_solution = None
    best_size = float('inf')
    
    # Try multiple random restarts
    for attempt in range(20):
        if time.time() > deadline - 30:  # Leave 30s for final validation
            break
        
        random.seed(42 + attempt)
        
        # Initialize with a random word
        current_solution = [random.choice(all_words)]
        covered = set(tuple(w) for w in balls[tuple(current_solution[0])])
        
        # Greedy: add words that cover the most uncovered words
        uncovered = set(range(total_words))
        for i, w in enumerate(all_words):
            if tuple(w) in covered:
                uncovered.discard(i)
        
        while uncovered:
            if time.time() > deadline - 30:
                break
            
            # Find word that covers most uncovered words
            best_gain = -1
            best_word_idx = -1
            
            for i, w in enumerate(all_words):
                if i in covered:
                    continue
                gain = len(set(tuple(v) for v in balls[tuple(w)]) & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_word_idx = i
            
            if best_word_idx == -1 or best_gain == 0:
                break
            
            current_solution.append(all_words[best_word_idx])
            for v in balls[tuple(all_words[best_word_idx])]:
                covered.add(tuple(v))
            
            # Update uncovered
            for i, w in enumerate(all_words):
                if tuple(w) in covered:
                    uncovered.discard(i)
        
        # Simulated annealing to try to remove redundant codewords
        temp = 10.0
        cooling = 0.95
        
        for sa_iter in range(500):
            if time.time() > deadline - 30:
                break
            
            if len(current_solution) <= 1:
                break
            
            # Try removing a random codeword
            idx_to_remove = random.randint(0, len(current_solution) - 1)
            removed_word = current_solution[idx_to_remove]
            
            # Check if still covered without this word
            temp_covered = set()
            for w in current_solution:
                if w == removed_word:
                    continue
                temp_covered.update(tuple(v) for v in balls[tuple(w)])
            
            if len(temp_covered) == total_words:
                # Safe to remove
                current_solution.pop(idx_to_remove)
                temp *= cooling
            else:
                # Not safe, accept with probability based on temperature
                # Penalty = number of uncovered words
                penalty = total_words - len(temp_covered)
                if random.random() < np.exp(-penalty / max(temp, 0.01)):
                    current_solution.pop(idx_to_remove)
                temp *= cooling
        
        # Validate and update best
        final_covered = set()
        for w in current_solution:
            final_covered.update(tuple(v) for v in balls[tuple(w)])
        
        if len(final_covered) == total_words:
            if len(current_solution) < best_size:
                best_size = len(current_solution)
                best_solution = [list(w) for w in current_solution]
    
    if best_solution is None:
        # Fallback: use direct sum construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return best_solution
# EVOLVE-BLOCK-END
