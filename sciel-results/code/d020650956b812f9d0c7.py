"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Generate all words in the space
    total_words = q ** n
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute balls for each word
    # ball[idx] is a list of indices of words within Hamming distance <= radius
    balls = []
    for idx in range(total_words):
        word = all_words[idx]
        ball = set([idx])
        # For radius=1 (standard for this problem size usually, but handle generally)
        # If radius > 1, this needs a BFS or combinatorial generation.
        # Given the task "K_3(7,1)", radius is 1.
        # We assume radius=1 for optimization, but add a fallback for general radius if needed.
        # However, the prompt implies specific task parameters. 
        # "every one of the 2187 words differs from some chosen word in at most one position" -> radius=1.
        # We will hardcode radius=1 logic for speed, as the task contract specifies this instance.
        if radius == 1:
            for pos in range(n):
                for new_val in range(q):
                    if new_val != word[pos]:
                        modified = word.copy()
                        modified[pos] = new_val
                        ball.add(word_to_idx[tuple(modified)])
        else:
            # Fallback for general radius (slower)
            # Generate all words within distance radius
            # This is expensive for large n, but n=7 is small.
            for dist in range(radius + 1):
                if dist == 0:
                    ball.add(idx)
                else:
                    for positions in itertools.combinations(range(n), dist):
                        for vals in itertools.product(range(q), repeat=dist):
                            if any(v == word[p] for v, p in zip(vals, positions)):
                                continue
                            modified = word.copy()
                            for p, v in zip(positions, vals):
                                modified[p] = v
                            ball.add(word_to_idx[tuple(modified)])
        balls.append(list(ball))
        
    # Convert balls to sets for faster intersection if needed, but lists are better for iteration.
    # We will use lists for balls.
    
    def greedy_cover(uncovered_set, current_codewords, coverage_count):
        """
        Greedily add codewords to cover uncovered_set.
        Returns updated codewords list and coverage_count.
        """
        # We need to find the codeword that covers the most uncovered words.
        # To do this efficiently, we can just iterate all candidates.
        # Since total_words is 2187, and we do this maybe 100-200 times, it's fast enough.
        
        while uncovered_set:
            if time.time() > deadline - 5:
                break
            
            best_idx = -1
            max_cover = -1
            
            # Optimization: Only consider candidates that cover at least one uncovered word?
            # Actually, just check all.
            for idx in range(total_words):
                # Count how many words in ball[idx] are in uncovered_set
                # Using set intersection is fast in Python
                count = len(set(balls[idx]) & uncovered_set)
                if count > max_cover:
                    max_cover = count
                    best_idx = idx
                    
            if best_idx == -1 or max_cover == 0:
                break # Should not happen if uncovered_set is not empty and balls cover space
            
            # Add best_idx to codewords
            current_codewords.append(best_idx)
            
            # Update coverage counts
            for w in balls[best_idx]:
                coverage_count[w] += 1
                
            # Update uncovered_set
            uncovered_set -= set(balls[best_idx])
            
        return current_codewords, coverage_count, uncovered_set

    def remove_redundant(codewords, coverage_count):
        """
        Remove codewords that are redundant.
        A codeword c is redundant if for all w in ball(c), coverage_count[w] > 1.
        """
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            i = 0
            while i < len(codewords):
                if time.time() > deadline - 10:
                    break
                
                c = codewords[i]
                # Check if c can be removed
                can_remove = True
                for w in balls[c]:
                    if coverage_count[w] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove c
                    codewords.pop(i)
                    for w in balls[c]:
                        coverage_count[w] -= 1
                    changed = True
                    # Do not increment i, check same index again
                else:
                    i += 1
                    
        return codewords, coverage_count

    # 1. Initial Greedy Construction
    # Start with empty code
    codewords = []
    coverage_count = [0] * total_words
    uncovered = set(range(total_words))
    
    codewords, coverage_count, uncovered = greedy_cover(uncovered, codewords, coverage_count)
    
    # 2. Initial Cleanup
    codewords, coverage_count = remove_redundant(codewords, coverage_count)
    
    best_codewords = codewords[:]
    best_size = len(codewords)
    
    # 3. Perturbation Loop
    # Try to improve by removing some codewords and re-greedy covering.
    # This helps escape local minima.
    
    # Set random seed for reproducibility? No, usually better to let it vary or fix for debugging.
    # The task doesn't require deterministic output, just a good score.
    
    while time.time() < deadline - 15:
        # Create a copy of the current state
        curr_codewords = best_codewords[:]
        curr_coverage = [0] * total_words
        for c in curr_codewords:
            for w in balls[c]:
                curr_coverage[w] += 1
        
        # Perturb: Remove k random codewords
        # k can be 1, 2, or 3
        k = random.randint(1, 3)
        if len(curr_codewords) <= k:
            continue
            
        # Select indices to remove
        remove_indices = random.sample(range(len(curr_codewords)), k)
        # Sort in reverse to pop safely
        remove_indices.sort(reverse=True)
        
        uncovered = set()
        for idx in remove_indices:
            c = curr_codewords[idx]
            for w in balls[c]:
                curr_coverage[w] -= 1
                if curr_coverage[w] == 0:
                    uncovered.add(w)
            curr_codewords.pop(idx)
            
        # Re-greedy cover the uncovered words
        curr_codewords, curr_coverage, uncovered = greedy_cover(uncovered, curr_codewords, curr_coverage)
        
        # Cleanup redundancies
        curr_codewords, curr_coverage = remove_redundant(curr_codewords, curr_coverage)
        
        # Check if improved
        if len(curr_codewords) < best_size:
            best_size = len(curr_codewords)
            best_codewords = curr_codewords[:]
            
        # Optional: If we found a valid solution, we can also try a simple 1-for-1 swap
        # But the perturb+greedy is usually stronger.
        
    # Return the best codewords found
    # The interface requires a list of codewords (lists of ints)
    return [all_words[c] for c in best_codewords]

# EVOLVE-BLOCK-END
