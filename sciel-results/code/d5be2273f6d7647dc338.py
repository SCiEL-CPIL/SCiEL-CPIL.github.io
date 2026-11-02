import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute Hamming distance balls
    # ball_masks[i] is a bitmask where bit j is 1 if all_words[j] is within radius of all_words[i]
    ball_masks = [0] * total_words
    
    # Efficiently compute ball masks
    # For each word, find all words within radius
    # Radius 1: The word itself + words differing in 1 position.
    # General radius implementation:
    for i, w in enumerate(all_words):
        mask = 0
        for j, v in enumerate(all_words):
            # Hamming distance check
            dist = sum(1 for a, b in zip(w, v) if a != b)
            if dist <= radius:
                mask |= (1 << j)
        ball_masks[i] = mask

    all_ones = (1 << total_words) - 1
    
    def is_valid(code_indices):
        """Check if the set of codeword indices covers all points."""
        covered = 0
        for idx in code_indices:
            covered |= ball_masks[idx]
        return covered == all_ones

    def minimize_code(code_indices):
        """Remove redundant codewords."""
        changed = True
        while changed:
            changed = False
            for i in range(len(code_indices)):
                # Try removing code_indices[i]
                new_code = code_indices[:i] + code_indices[i+1:]
                if is_valid(new_code):
                    code_indices = new_code
                    changed = True
                    break # Restart loop as indices shifted
        return code_indices

    def greedy_cover():
        """Standard greedy set cover."""
        uncovered = all_ones
        code_indices = []
        while uncovered != 0:
            best_idx = -1
            best_count = 0
            for i in range(total_words):
                # Count how many new points this word covers
                new_points = uncovered & ball_masks[i]
                count = new_points.bit_count()
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx == -1:
                break
            code_indices.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
        return code_indices

    def randomized_greedy_cover(randomness_factor=1.0):
        """Greedy with randomization among top candidates."""
        uncovered = all_ones
        code_indices = []
        while uncovered != 0:
            candidates = []
            max_count = 0
            for i in range(total_words):
                new_points = uncovered & ball_masks[i]
                count = new_points.bit_count()
                if count > max_count:
                    max_count = count
                    candidates = [i]
                elif count == max_count and count > 0:
                    candidates.append(i)
            
            if not candidates:
                break
            
            # Pick randomly from top candidates
            # randomness_factor controls how many top candidates to consider
            # If 1.0, only the single best. If higher, consider more.
            # Actually, let's just pick the best, but add noise if we want.
            # For now, standard greedy is a good baseline.
            # Let's add a slight randomization: pick from top 5 if available.
            if len(candidates) > 5:
                # Sort by count? They all have same count in this logic.
                # Just pick random from candidates.
                best_idx = random.choice(candidates)
            else:
                best_idx = random.choice(candidates)
                
            code_indices.append(best_idx)
            uncovered &= ~ball_masks[best_idx]
        return code_indices

    # 1. Generate initial solution using randomized greedy
    best_code = randomized_greedy_cover()
    best_code = minimize_code(best_code)
    best_size = len(best_code)

    # 2. Local Search: Swap and Reduce
    # We try to swap a codeword with another, then reduce.
    # If the size decreases, accept.
    
    # To make it faster, we can precompute coverage of the current set
    # But re-computing is O(K * N) where K is code size. N=729. K~80. 80*729 ~ 58k ops.
    # This is fast enough to do many iterations.

    current_code = best_code[:]
    
    while time.time() < deadline - 10.0:
        improved = False
        
        # Try removing redundant codewords (already done in minimize, but check again)
        current_code = minimize_code(current_code)
        
        if len(current_code) < best_size:
            best_code = current_code[:]
            best_size = len(best_code)
            improved = True
            
        # Try swapping
        # Pick a random codeword to swap out
        if len(current_code) > 0:
            i = random.randint(0, len(current_code) - 1)
            old_idx = current_code[i]
            
            # Try swapping with a random other codeword
            new_idx = random.randint(0, total_words - 1)
            
            if new_idx != old_idx:
                # Check if new_idx is already in current_code
                if new_idx not in current_code:
                    candidate_code = current_code[:]
                    candidate_code[i] = new_idx
                    # Reduce
                    candidate_code = minimize_code(candidate_code)
                    
                    if len(candidate_code) < len(current_code):
                        current_code = candidate_code
                        improved = True
                    elif len(candidate_code) == len(current_code) and random.random() < 0.1:
                        # Occasional plateau move to escape local optima
                        current_code = candidate_code
                        improved = True
        
        # If we found a new best, we might want to restart or continue
        # If no improvement for a while, try a different random seed or strategy?
        # For simplicity, just keep swapping.

    # Final minimization
    best_code = minimize_code(best_code)
    
    # Convert indices back to words
    code_words = [list(all_words[idx]) for idx in best_code]
    
    return code_words
