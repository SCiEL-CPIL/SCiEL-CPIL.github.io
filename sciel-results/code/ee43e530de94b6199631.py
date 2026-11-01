import numpy as np
import time
import itertools

def construct(q, n, radius):
    """
    Construct a covering code using a greedy heuristic.
    """
    start_time = time.time()
    timeout = 110  # Safety margin from 120s
    
    total_words = q ** n
    
    # If the problem is small enough, we could do exact, but greedy is safer for time.
    # For n=8, q=3, total_words=6561.
    
    # Generate all words as integers
    # We need to map integer <-> tuple/list
    # For efficiency in the inner loop, we work with integers.
    
    # Precompute the ball (neighbors within radius) for each word.
    # ball[i] is a list of integers representing words within Hamming distance 'radius' of word i.
    
    # Helper to get neighbors of an integer word
    def get_neighbors(word_int):
        # Convert to base-q digits
        digits = [0] * n
        temp = word_int
        for i in range(n):
            digits[i] = temp % q
            temp //= q
        
        neighbors = [word_int]
        
        # For radius 1, we just change one digit
        if radius >= 1:
            for i in range(n):
                orig = digits[i]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    # Calculate new integer
                    # new_int = word_int - orig * (q**i) + new_val * (q**i)
                    diff = (new_val - orig) * (q ** i)
                    neighbors.append(word_int + diff)
        
        return neighbors

    # Precompute balls for all words
    # This might take a bit, but 6561 * 17 is small.
    balls = []
    for i in range(total_words):
        balls.append(get_neighbors(i))
        
    # Convert balls to numpy arrays for faster indexing if needed, 
    # but lists of ints are fine for the update loop if we keep it tight.
    
    # uncovered: boolean array, True if word is not covered yet
    uncovered = np.ones(total_words, dtype=bool)
    
    # coverage_count: for each candidate codeword c, how many uncovered words are in its ball
    # coverage_count[c] = sum(uncovered[w] for w in balls[c])
    coverage_count = np.zeros(total_words, dtype=np.int32)
    
    # Initialize coverage_count
    for c in range(total_words):
        # Count how many uncovered words are in balls[c]
        # Since all are uncovered initially, it's just len(balls[c])
        coverage_count[c] = len(balls[c])
        
    code = []
    
    # Main greedy loop
    while True:
        if time.time() - start_time > timeout:
            break
            
        # Find the codeword with max coverage
        # argmax
        best_idx = np.argmax(coverage_count)
        best_count = coverage_count[best_idx]
        
        if best_count == 0:
            break # All covered
            
        # Add best_idx to code
        code.append([int(d) for d in np.base_repr(best_idx, base=q).zfill(n)[-n:] if d.isdigit()])
        # Wait, np.base_repr returns string. Let's use a faster conversion or just store int and convert at end?
        # The interface requires list of codewords, each a list of ints.
        # Let's store ints in code list and convert at the very end to save time in the loop.
        # Actually, appending to list is fast. Let's store the integer representation.
        # I'll revert to storing the integer for now and convert at the end.
        
        # Undo the append above, store int
        code.pop()
        code.append(best_idx)
        
        # Update uncovered and coverage_count
        # 1. Mark words in balls[best_idx] as covered
        # 2. For each newly covered word w, decrement coverage_count[d] for all d in balls[w]
        
        # Get the list of words covered by best_idx
        covered_by_best = balls[best_idx]
        
        # We only care about words that were *newly* covered
        # newly_covered = [w for w in covered_by_best if uncovered[w]]
        
        # Optimization: 
        # We iterate through covered_by_best.
        for w in covered_by_best:
            if uncovered[w]:
                uncovered[w] = False
                # Decrement coverage_count for all d that cover w
                # d is in balls[w]
                for d in balls[w]:
                    coverage_count[d] -= 1
                    
    # Convert code integers to lists of digits
    final_code = []
    for c_int in code:
        digits = []
        temp = c_int
        for _ in range(n):
            digits.append(temp % q)
            temp //= q
        final_code.append(digits)
        
    return final_code
