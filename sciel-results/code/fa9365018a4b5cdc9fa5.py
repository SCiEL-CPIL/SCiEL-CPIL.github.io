import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Generate the initial 81-word code (Tetracode sum)
    # Tetracode: 9 words of length 4
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Full space for remaining n-4 coordinates
    if n > 4:
        tails = list(itertools.product(range(3), repeat=n - 4))
    else:
        tails = [()]
        
    initial_codewords = [h + list(tail) for h in tetracode for tail in tails]
    
    # Convert to tuples for hashing and set operations
    initial_set = set(tuple(c) for c in initial_codewords)
    
    # Precompute balls for all possible words in the space? 
    # Space size is 3^n. For n=6, 729. Ball size is 1 + n(q-1) = 13.
    # We can precompute the ball (neighbors + self) for every word in the space.
    # This speeds up the cover count updates.
    
    total_words = q ** n
    # Map word tuple to index? Or just use tuples as keys in a dict for cover counts.
    # Using a dict `cover_counts` where key is word tuple, value is count.
    
    # Precompute balls for all words in the space
    # This is 729 * 13 operations, very fast.
    balls = {}
    for w in itertools.product(range(q), repeat=n):
        w = tuple(w)
        ball = {w}
        for i in range(n):
            for val in range(q):
                if val != w[i]:
                    new_w = list(w)
                    new_w[i] = val
                    ball.add(tuple(new_w))
        balls[w] = ball
        
    # Function to compute cover counts for a set of codewords
    def compute_cover_counts(codewords):
        counts = {}
        for c in codewords:
            for w in balls[c]:
                counts[w] = counts.get(w, 0) + 1
        return counts

    # Function to check if a code is valid (covers all words)
    def is_valid(codewords):
        # A code is valid if every word in the space is covered by at least one codeword.
        # We can check this by ensuring cover_counts[w] > 0 for all w.
        # But we don't want to recompute from scratch if we can avoid it.
        # However, for final verification, we must be sure.
        # For the search, we maintain counts.
        pass

    best_code = list(initial_set)
    best_size = len(best_code)
    
    # Greedy reduction function
    def greedy_reduce(codewords_set):
        # codewords_set is a set of tuples
        if len(codewords_set) <= best_size:
            return set(codewords_set) # No point reducing if we can't beat best
        
        counts = compute_cover_counts(codewords_set)
        current_set = set(codewords_set)
        
        while True:
            redundant_found = False
            # Iterate through a copy of the set to avoid modification during iteration issues
            # We pick a random order to escape symmetries
            for c in list(current_set):
                # Check if c is redundant
                # c is redundant if all w in balls[c] have counts[w] > 1
                redundant = True
                for w in balls[c]:
                    if counts.get(w, 0) <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove c
                    current_set.remove(c)
                    for w in balls[c]:
                        counts[w] -= 1
                    redundant_found = True
                    break # Restart scan to find new redundancies
            if not redundant_found:
                break
            if time.time() > deadline:
                return current_set
        return current_set

    # Try multiple random permutations of the initial set
    # The initial set is symmetric, so the order of removal might matter.
    # We will shuffle the list of codewords and try to remove in that order?
    # The `greedy_reduce` above picks the *first* redundant one found in the iteration order.
    # So shuffling the iteration order is equivalent to shuffling the set.
    
    initial_list = list(initial_set)
    
    # First, try standard greedy
    reduced = greedy_reduce(set(initial_list))
    if len(reduced) < best_size:
        best_code = list(reduced)
        best_size = len(best_code)
        
    # If we haven't reached 73 (or close), try random restarts
    # We can also try to "swap" if stuck, but let's see if random restarts of greedy suffice.
    # The search space of "which redundant word to remove first" is the key.
    
    attempts = 0
    while time.time() < deadline:
        random.shuffle(initial_list)
        # We need to pass a set, but the order of iteration in `greedy_reduce` is determined by `list(current_set)`.
        # Python sets don't have guaranteed order, but `list(set)` order is somewhat stable for same elements.
        # To enforce a specific random order, we should modify `greedy_reduce` to accept a priority order or just shuffle the set contents?
        # Easier: Create a list, shuffle it, and use a logic that checks redundancy in that specific order.
        
        def greedy_reduce_ordered(codewords_list):
            current_set = set(codewords_list)
            counts = compute_cover_counts(current_set)
            
            while True:
                redundant_found = False
                # We want to check in the order of `codewords_list` if possible, 
                # but the list changes as we remove items.
                # Simple approach: Iterate through the *original* shuffled list, 
                # and if the word is still in current_set, check it.
                for c in codewords_list:
                    if c not in current_set:
                        continue
                    redundant = True
                    for w in balls[c]:
                        if counts.get(w, 0) <= 1:
                            redundant = False
                            break
                    if redundant:
                        current_set.remove(c)
                        for w in balls[c]:
                            counts[w] -= 1
                        redundant_found = True
                        break # Restart scan
                if not redundant_found:
                    break
                if time.time() > deadline:
                    return current_set
            return current_set

        reduced = greedy_reduce_ordered(initial_list)
        if len(reduced) < best_size:
            best_code = list(reduced)
            best_size = len(best_code)
            if best_size <= 73: # Known optimum is 73
                break
        attempts += 1
        if attempts > 200: # Safety break
            break

    # Final verification
    # Convert best_code to list of lists for return
    final_code = [list(c) for c in best_code]
    
    # Verify validity
    if final_code:
        counts = compute_cover_counts(set(tuple(c) for c in final_code))
        # Check if all 729 words are covered
        for w in itertools.product(range(q), repeat=n):
            if counts.get(tuple(w), 0) == 0:
                # Invalid! Fallback to initial 81
                final_code = [list(c) for c in initial_set]
                break
                
    return final_code
