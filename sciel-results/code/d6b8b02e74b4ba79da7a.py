import time
import random
import sys

# EVOLVE-BLOCK-START

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters if they ever appear, though task is specific
        # For now, just return a trivial cover if not the specific case
        # But the task says "evaluator passes this task's alphabet size q, length n and covering radius"
        # So we assume q=3, n=7, radius=1.
        # We'll implement a generic greedy for safety, but optimize for 3,7,1.
        pass

    total_words = q ** n
    words = list(range(total_words))
    
    # Precompute neighbors for each word in the Hamming space of radius 'radius'
    # For q=3, n=7, radius=1:
    # Neighbors of w are w itself and w with one digit changed to one of the other q-1 values.
    
    # Helper to decode integer to list of digits
    def decode(val):
        return [val // (q ** (n - i - 1)) % q for i in range(n)]
    
    # Helper to encode list of digits to integer
    def encode(digits):
        val = 0
        for i, d in enumerate(digits):
            val += d * (q ** (n - i - 1))
        return val

    # Precompute all balls
    # ball[w] is a list of words covered by w
    balls = {}
    for w in words:
        digs = decode(w)
        neighbors = [w]
        for pos in range(n):
            original = digs[pos]
            for new_val in range(q):
                if new_val != original:
                    new_digs = digs[:]
                    new_digs[pos] = new_val
                    neighbors.append(encode(new_digs))
        balls[w] = neighbors

    # Greedy Set Cover
    uncovered = set(words)
    selected = []
    
    # Optimization: 
    # Instead of recomputing coverage for all candidates every step (O(N^2)),
    # we can pick a random candidate from uncovered, or use a heuristic.
    # A standard greedy picks the one with max intersection with uncovered.
    # For 2187 words, N^2 is ~ 4.7 million ops. With 200 steps, that's ~1 billion ops. 
    # In Python, this might be too slow.
    # We need a faster heuristic or a randomized approach.
    
    # Let's try a randomized greedy with local search.
    # 1. Randomly select a starting set that covers everything? No, that's hard.
    # 2. Greedy: 
    #    To speed up, we can maintain a count of how many times each word is covered by *potential* candidates?
    #    No, standard greedy is:
    #    while uncovered:
    #       find w in words that maximizes len(balls[w] ∩ uncovered)
    #       add w to selected
    #       uncovered -= balls[w]
    
    # To make this faster in Python:
    # We can sample a subset of candidates.
    
    best_selected = []
    
    # Run multiple restarts of the greedy+prune to find a better solution
    # Since the space is small, we can try a few different random orderings or tie-breakers.
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        uncovered = set(words)
        current_selected = []
        
        # Randomize the order of words to break ties in greedy selection
        # Or just iterate through all, but pick the best one.
        # To avoid O(N^2) full scan, we can use a "lazy" approach or just accept the cost if N is small enough.
        # 2187 is small. 2187^2 * 200 iterations = 950 million operations. 
        # Python might take 10-20 seconds per restart. 5 restarts = 50-100s. 
        # This is risky but potentially fits in 110s.
        # Let's optimize the inner loop.
        
        # Pre-compute ball sets for faster intersection? 
        # balls[w] is a list. Converting to set every time is expensive.
        # Let's pre-compute balls as frozensets? 
        # frozenset(balls[w]) for all w.
        
        if 'ball_sets' not in locals():
            ball_sets = {w: frozenset(neighbors) for w, neighbors in balls.items()}
        else:
            ball_sets = ball_sets

        while uncovered:
            best_w = -1
            max_covered = 0
            
            # Optimization: If we have many uncovered, picking any word is good.
            # We can iterate through a random sample of words to find a good candidate, 
            # rather than checking all 2187.
            # However, checking all 2187 is only 2187 set intersections.
            # len(frozenset & set) is fast in C.
            
            # Let's check all words but stop early if we find a perfect cover? No.
            # Just find the max.
            
            for w in words:
                # Optimization: skip if w is already selected? 
                # Actually, adding a duplicate doesn't help, but we are picking from all words.
                # If w is already in current_selected, its ball is already subtracted from uncovered, 
                # so len(balls[w] & uncovered) will be 0 (or small if overlaps).
                # So we can just check all.
                
                covered_count = len(ball_sets[w] & uncovered)
                if covered_count > max_covered:
                    max_covered = covered_count
                    best_w = w
                    if max_covered == len(uncovered):
                        break # Can't do better than covering all remaining
            
            if best_w == -1:
                break # Should not happen unless uncovered is empty
                
            current_selected.append(best_w)
            uncovered -= ball_sets[best_w]
            
            if time.time() > deadline:
                break
        
        if time.time() > deadline:
            break
            
        # Local Search: Greedy Removal
        # Try to remove each codeword if it doesn't break coverage.
        # We need to maintain the set of covered words.
        # covered = union of balls of current_selected
        # This is expensive to recompute from scratch every removal.
        
        # Better approach for removal:
        # Count how many selected codewords cover each word.
        # If count[word] > 1 for all word in ball[c], we can remove c.
        
        # Let's implement the counting method.
        # counts = {w: 0 for w in words}
        # for c in current_selected:
        #    for w in balls[c]:
        #       counts[w] += 1
        
        # To remove c:
        # Check if for all w in balls[c], counts[w] > 1.
        # If yes, decrement counts[w] for w in balls[c], and remove c.
        
        # This is O(N * ball_size) per removal check.
        # N ~ 200, ball_size ~ 15. 3000 ops per check.
        # We check all N candidates. 200 * 3000 = 600,000 ops per pass.
        # Very fast.
        
        counts = {w: 0 for w in words}
        for c in current_selected:
            for w in balls[c]:
                counts[w] += 1
        
        # Iterate until no more removals
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to allow index removal
            for i in range(len(current_selected) - 1, -1, -1):
                c = current_selected[i]
                # Check if all words in balls[c] have count > 1
                can_remove = True
                for w in balls[c]:
                    if counts[w] <= 1:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove c
                    for w in balls[c]:
                        counts[w] -= 1
                    current_selected.pop(i)
                    changed = True
                    if time.time() > deadline:
                        break
            
            if time.time() > deadline:
                break
        
        # Update best
        if not best_selected or len(current_selected) < len(best_selected):
            best_selected = current_selected[:]
            
        if len(best_selected) == 156: # Lower bound reached
            break
            
    # If we didn't find anything (shouldn't happen), return a valid (though large) cover
    # The greedy loop above should always produce a valid cover if it completes.
    # If it timed out mid-greedy, best_selected might be empty or partial.
    # If best_selected is empty, we need a fallback.
    # Let's ensure we at least return the seed solution if we fail.
    
    if not best_selected:
        # Fallback to seed construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        best_selected = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert back to lists of integers
    # best_selected contains integers.
    return [decode(w) for w in best_selected]

# EVOLVE-BLOCK-END
