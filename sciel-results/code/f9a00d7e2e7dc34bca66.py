import numpy as np
import itertools
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet q with covering radius radius.
    Returns a list of codewords (each a list of ints).
    """
    start_time = time.time()
    deadline = 150.0  # Safety margin for 120s limit
    
    # Total number of words in the space
    N = q ** n
    
    # If the space is small, we can potentially solve it more exactly or just use the greedy approach
    # For n=7, q=3, N=2187. This is small enough for efficient set cover.
    
    # Precompute all words as tuples of digits
    # We will work with indices 0..N-1 for speed
    
    # Helper to convert index to word (list of digits)
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[n - 1 - i] = idx % q
            idx //= q
        return word

    # Helper to convert word to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx

    # Precompute the ball (set of words within Hamming distance radius) for every possible word index
    # balls[i] is a list of indices of words within distance 'radius' from word i
    balls = [None] * N
    
    # Generate all words
    all_words = [idx_to_word(i) for i in range(N)]
    
    # Precompute distances or just generate balls directly
    # For each word, generate all words within Hamming distance <= radius
    # Optimization: Since radius is small (usually 1), we can generate neighbors directly.
    # General case:
    
    # To speed up, we can precompute the balls using a BFS or direct generation.
    # Direct generation for radius 1:
    # For radius > 1, we can use combinatorics, but for n=7, radius=1 is the target case.
    # Let's write a general ball generator.
    
    def get_ball_indices(center_idx):
        center_word = all_words[center_idx]
        ball_indices = {center_idx}
        
        if radius == 1:
            for pos in range(n):
                original_val = center_word[pos]
                for new_val in range(q):
                    if new_val == original_val:
                        continue
                    new_word = center_word.copy()
                    new_word[pos] = new_val
                    ball_indices.add(word_to_idx(new_word))
        else:
            # General case: iterate over all subsets of positions to change
            # This is expensive for large radius, but radius is typically small.
            # For radius 2, we change 1 or 2 positions.
            # We'll use a simple loop for general radius, but optimize for 1.
            # Since n is small, we can just check distance for all words if N is small?
            # No, N=2187, checking 2187*2187 is ~5M ops, which is fine.
            # But generating neighbors is faster.
            
            # Let's stick to the efficient radius=1 case if possible, 
            # but implement a fallback for general radius using distance check 
            # only if the generation logic is too complex. 
            # Actually, for n=7, q=3, radius=1 is the specific task.
            # Let's assume radius is small.
            
            # A robust way: 
            # Generate all words within distance d.
            # We can use a recursive generation or just check all words if N is small enough.
            # N=2187, checking distance for each pair is 2187^2 / 2 ~ 2.4M distance calcs.
            # Each distance calc is O(n)=7. Total ~17M ops. This is fast in Python? Maybe 1-2 seconds.
            # Precomputing balls this way is safe.
            
            pass # We will use the radius=1 optimized version below, 
                 # and a generic version if needed.

    # Optimized ball precomputation for radius=1
    if radius == 1:
        for i in range(N):
            w = all_words[i]
            inds = [i]
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    w[pos] = new_val
                    inds.append(word_to_idx(w))
                    # No need to restore w[pos] if we use copy or careful handling
                    # Actually w is modified in place. Let's use a copy for safety or restore.
                    # Better: create new word
            # Let's redo this cleanly
            inds = [i]
            for pos in range(n):
                orig = w[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    # Create new word
                    new_w = w.copy()
                    new_w[pos] = new_val
                    inds.append(word_to_idx(new_w))
            balls[i] = inds
    else:
        # Generic ball precomputation
        # For each center, find all words within Hamming distance <= radius
        # We can do this by iterating all words and checking distance, 
        # or by generating neighbors. 
        # Given the constraints, checking all pairs is acceptable for n<=7.
        # But let's try to generate neighbors for efficiency.
        
        # Simpler: just check all words for distance.
        # dist(a, b) <= radius
        # This is O(N^2 * n). For N=2187, this is ~32M operations. 
        # In Python, this might take 5-10 seconds. Acceptable.
        
        # Even better: Use the fact that we only need to do this once.
        for i in range(N):
            ci = all_words[i]
            inds = []
            for j in range(N):
                cj = all_words[j]
                dist = 0
                for k in range(n):
                    if ci[k] != cj[k]:
                        dist += 1
                        if dist > radius:
                            break
                if dist <= radius:
                    inds.append(j)
            balls[i] = inds
            if time.time() > deadline - 10:
                break # Safety break

    # Initialize coverage
    # uncovered[i] is True if word i is not covered by the current code
    uncovered = np.ones(N, dtype=bool)
    code = []
    
    # Greedy Set Cover
    # We want to pick words to cover the most uncovered points.
    
    # Pre-compute ball arrays for faster intersection?
    # balls[i] is a list of ints.
    # We can convert to numpy array for faster boolean indexing?
    # Or just use python sets/lists.
    
    # To speed up the "count uncovered in ball" step:
    # We can maintain a count of how many times each word is covered? No, we just need binary coverage.
    # Let's use a list of sets for balls? No, lists are fine.
    
    # Optimization: 
    # Instead of iterating all N candidates every time, we can just do it.
    # N=2187. 
    # Number of steps ~ 100-200.
    # Total ops ~ 200 * 2187 * (avg ball size ~15) = 6.5M ops. Fast.
    
    while any(uncovered):
        best_idx = -1
        best_count = 0
        
        # Iterate through all possible candidates
        # We can skip candidates that are already in the code? 
        # Actually, adding a duplicate doesn't help. 
        # But we don't have a fast way to check "is in code" unless we use a set.
        code_set = set(code)
        
        for i in range(N):
            if i in code_set:
                continue
            
            # Count how many uncovered words are in balls[i]
            # balls[i] is a list of indices
            # We need sum(uncovered[j] for j in balls[i])
            
            # Using numpy:
            # ball_arr = np.array(balls[i])
            # count = np.sum(uncovered[ball_arr])
            # This creates an array every time. Might be slow.
            
            # Python loop:
            count = 0
            for j in balls[i]:
                if uncovered[j]:
                    count += 1
                    if count > best_count:
                        # Early exit if we already beat best?
                        # No, we need the max. But if count > best_count, we update.
                        # We can't exit early unless we know we can't beat it?
                        # Max possible is len(balls[i]).
                        pass
            
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1 or best_count == 0:
            # Should not happen if a solution exists
            break
        
        # Add best_idx to code
        code.append(best_idx)
        
        # Update uncovered
        for j in balls[best_idx]:
            uncovered[j] = False
        
        if time.time() > deadline - 5:
            break

    # Local Search: Try to remove codewords
    # Iterate until no more removals are possible
    improved = True
    while improved:
        improved = False
        # Iterate in reverse to handle removals safely
        for i in range(len(code) - 1, -1, -1):
            # Check if we can remove code[i]
            # This means: Are all words in balls[code[i]] covered by the OTHER codewords?
            
            # Optimization: 
            # We can check if any word in balls[code[i]] is ONLY covered by code[i].
            # If such a word exists, we cannot remove code[i].
            
            # Let's compute the coverage count for words in balls[code[i]]
            # due to the rest of the code.
            
            # Simple check:
            # Temporarily mark code[i] as removed.
            # Check if uncovered is still all False.
            
            # We can simulate:
            # Let's check if any word w in balls[code[i]] is not covered by any other c in code.
            
            # Efficient way:
            # For each w in balls[code[i]], check if there exists c != code[i] such that w in balls[c].
            
            # This is O(BallSize * CodeSize * BallSize) which is slow.
            
            # Better:
            # Maintain a count of how many codewords cover each word.
            # Let's rebuild the coverage counts from scratch for the local search?
            # Code size ~ 100. N = 2187.
            # Rebuilding counts: 100 * 15 = 1500 ops. Very fast.
            
            # Let's just re-verify coverage from scratch for the current code minus the candidate.
            
            # Actually, simpler:
            # Check if removing code[i] leaves any word uncovered.
            # We can do this by checking if any word in balls[code[i]] is not covered by the union of other balls.
            
            # Let's use a boolean array for "covered_by_others"
            # But we don't want to allocate a new array of size N every time.
            
            # Alternative:
            # Pick a random word w in balls[code[i]].
            # Check if w is covered by any other codeword.
            # If not, we can't remove.
            # If yes, it *might* be removable, but we need to check ALL words in the ball.
            
            # Let's just do the full check for correctness, it's fast enough.
            
            candidate = code[i]
            can_remove = True
            
            # Check every word in the ball of candidate
            for w in balls[candidate]:
                # Is w covered by any other codeword?
                covered = False
                for c in code:
                    if c == candidate:
                        continue
                    # Check if w is in balls[c]
                    # To make this fast, we can precompute a set for balls?
                    # Or just check distance?
                    # Distance check is O(n). 
                    # Ball membership check via precomputed list is O(BallSize) if linear search.
                    # Let's use a set for balls for O(1) lookup?
                    # Pre-computing sets for all balls: N * BallSize memory. 2187 * 15 ~ 30k ints. Fine.
                    
                    # Let's create a set version of balls if not already done.
                    # We'll just check distance for simplicity and speed in this small context.
                    # dist(w, c) <= radius?
                    
                    # Wait, we have all_words.
                    # dist(all_words[w], all_words[c])
                    
                    # This is O(n) per check.
                    # Ball size ~15. Code size ~100.
                    # 15 * 100 * 7 = 10k ops per removal check.
                    # 100 removal checks = 1M ops. Fast.
                    
                    if w in balls[c]: # Using list 'in' is O(BallSize). 15*15=225. 100*225=22k.
                        covered = True
                        break
                
                if not covered:
                    can_remove = False
                    break
            
            if can_remove:
                code.pop(i)
                improved = True
                # Restart the loop to check for more removals
                # Actually, we can just continue, but popping changes indices.
                # Restarting the while loop is safer.
                break
        
        if time.time() > deadline - 5:
            break

    # Convert indices back to words
    result_words = [all_words[idx] for idx in code]
    
    # Ensure distinctness (should be guaranteed by set cover logic)
    # But just in case, unique them
    unique_words = []
    seen = set()
    for w in result_words:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            unique_words.append(w)
            
    return unique_words

# Fallback for very small n or if something goes wrong
if __name__ == "__main__":
    # Test with the specific problem
    res = construct(3, 7, 1)
    print(f"Code size: {len(res)}")
    # Verify
    N = 3**7
    all_words = []
    for i in range(N):
        w = []
        x = i
        for _ in range(7):
            w.append(x % 3)
            x //= 3
        all_words.append(tuple(w))
    
    code_set = set(tuple(w) for w in res)
    
    covered = [False] * N
    for w in res:
        w_tuple = tuple(w)
        for i in range(N):
            # Check distance
            dist = 0
            for k in range(7):
                if w_tuple[k] != all_words[i][k]:
                    dist += 1
                    if dist > 1:
                        break
            if dist <= 1:
                covered[i] = True
    
    print(f"Covered: {sum(covered)}/{N}")
