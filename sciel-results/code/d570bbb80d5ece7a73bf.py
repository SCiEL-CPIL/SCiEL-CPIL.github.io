import time
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    total_words = q ** n

    # Precompute index conversion helpers
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    def idx_to_tuple(idx):
        t = [0] * n
        rem = idx
        for i in range(n):
            t[i] = rem % q
            rem //= q
        return tuple(t)

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx += t[i] * powers[i]
        return idx

    # Precompute all balls (sets of indices within Hamming distance <= radius)
    # For radius=1, ball size is 1 + n*(q-1)
    balls = [set() for _ in range(total_words)]
    for i in range(total_words):
        w = idx_to_tuple(i)
        balls[i].add(i)
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                for new_val in range(q):
                    if new_val != orig_val:
                        new_w = list(w)
                        new_w[pos] = new_val
                        new_idx = tuple_to_idx(tuple(new_w))
                        balls[i].add(new_idx)

    def is_covered(code_indices, balls, total_words):
        """Check if code_indices covers all total_words."""
        covered = set()
        for c in code_indices:
            covered.update(balls[c])
        return len(covered) == total_words

    def prune_cover(indices, balls, total_words):
        """Remove redundant codewords."""
        indices = list(indices)
        if not indices:
            return indices
        
        # Count coverage
        cover_counts = np.zeros(total_words, dtype=np.int32)
        for idx in indices:
            for w in balls[idx]:
                cover_counts[w] += 1
        
        changed = True
        while changed:
            changed = False
            new_indices = []
            for idx in indices:
                necessary = False
                for w in balls[idx]:
                    if cover_counts[w] == 1:
                        necessary = True
                        break
                if necessary:
                    new_indices.append(idx)
                else:
                    for w in balls[idx]:
                        cover_counts[w] -= 1
                    changed = True
            indices = new_indices
        return indices

    def greedy_set_cover(balls, total_words, time_limit):
        """Standard greedy set cover algorithm."""
        uncovered = set(range(total_words))
        code_indices = []
        
        while uncovered and time.time() < time_limit:
            best_gain = -1
            best_candidates = []
            for c in range(total_words):
                gain = len(balls[c] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [c]
                elif gain == best_gain:
                    best_candidates.append(c)
            
            if not best_candidates:
                break
                
            best_c = random.choice(best_candidates)
            code_indices.append(best_c)
            uncovered -= balls[best_c]
            
        # If we ran out of time, add all remaining uncovered words to ensure validity
        if uncovered:
            for u in uncovered:
                code_indices.append(u)
                
        return code_indices

    def local_search_improve(code_indices, balls, total_words, time_limit):
        """Try to reduce the number of codewords via removal and swap operations."""
        code_indices = list(code_indices)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try removing one codeword
            for i in range(len(code_indices)):
                if time.time() > time_limit:
                    break
                trial_code = [c for j, c in enumerate(code_indices) if j != i]
                if is_covered(trial_code, balls, total_words):
                    code_indices = trial_code
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping one codeword with another
            code_set = set(code_indices)
            for i in range(len(code_indices)):
                if time.time() > time_limit:
                    break
                a = code_indices[i]
                # Try a few random swaps
                for _ in range(10):
                    if time.time() > time_limit:
                        break
                    b = random.randint(0, total_words - 1)
                    if b in code_set:
                        continue
                    trial_code = [c for c in code_indices if c != a]
                    trial_code.append(b)
                    
                    if is_covered(trial_code, balls, total_words):
                        pruned_trial = prune_cover(trial_code, balls, total_words)
                        if len(pruned_trial) < len(code_indices):
                            code_indices = pruned_trial
                            improved = True
                            break
                if improved:
                    break
                    
        return code_indices

    def simulated_annealing(code_indices, balls, total_words, time_limit):
        """Simulated annealing to escape local minima."""
        current_code = list(code_indices)
        best_code = list(current_code)
        best_size = len(current_code)
        
        temperature = 10.0
        cooling_rate = 0.995
        
        while time.time() < time_limit and temperature > 0.1:
            # Perturb: remove a random codeword and add a random new one
            if len(current_code) > 0:
                remove_idx = random.randint(0, len(current_code) - 1)
                removed_word = current_code[remove_idx]
                current_code.pop(remove_idx)
                
                add_word = random.randint(0, total_words - 1)
                current_code.append(add_word)
                
                # Prune to remove redundancies
                current_code = prune_cover(current_code, balls, total_words)
                
                # Check if valid
                if is_covered(current_code, balls, total_words):
                    new_size = len(current_code)
                    old_size = best_size
                    delta = new_size - old_size
                    
                    # Accept if better or with probability exp(-delta/temperature)
                    if delta < 0 or random.random() < np.exp(-delta / temperature):
                        if new_size < best_size:
                            best_code = list(current_code)
                            best_size = new_size
                    else:
                        # Reject: revert
                        current_code = list(best_code)
                else:
                    # Invalid: revert
                    current_code = list(best_code)
            
            temperature *= cooling_rate
        
        return best_code

    # Main algorithm
    # 1. Initial greedy solution
    initial_code = greedy_set_cover(balls, total_words, deadline - 30.0)
    initial_code = prune_cover(initial_code, balls, total_words)
    
    # 2. Local search improvement
    improved_code = local_search_improve(initial_code, balls, total_words, deadline - 15.0)
    
    # 3. Simulated annealing for further improvement
    final_code = simulated_annealing(improved_code, balls, total_words, deadline - 5.0)
    
    # Final verification and pruning
    final_code = prune_cover(final_code, balls, total_words)
    
    # Ensure validity (should always be true, but just in case)
    if not is_covered(final_code, balls, total_words):
        # Fallback: add all uncovered words
        covered = set()
        for c in final_code:
            covered.update(balls[c])
        for u in range(total_words):
            if u not in covered:
                final_code.append(u)
        final_code = prune_cover(final_code, balls, total_words)
    
    return [list(idx_to_tuple(c)) for c in final_code]
