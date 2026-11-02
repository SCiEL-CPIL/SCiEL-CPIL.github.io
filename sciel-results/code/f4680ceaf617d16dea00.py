import heapq
import time
import itertools

def construct(q, n, radius):
    # This implementation is specifically tuned for the ternary covering code problem K_3(7,1)
    # but will fall back to a simple construction for other parameters if needed.
    
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for the evaluator's 120s limit

    # If parameters don't match the specific task, use a fallback
    # The task is fixed: q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback: Direct sum or simple grid
        # For robustness, we return a valid but potentially suboptimal code
        # Simple construction: take every word where sum of coordinates mod 3 is 0? 
        # No, that's a code, not necessarily a covering code.
        # A safe fallback is the seed's logic or just returning a large set.
        # Since the task contract says "evaluator passes this task's alphabet size...", 
        # it implies q, n, radius are fixed for this specific task instance.
        # However, to be safe, we handle the general case poorly but correctly.
        if radius == 0:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
        
        # For general case, use a greedy approach too, but it might be slow.
        # Given the strict time limit, we assume the specific case q=3, n=7, r=1.
        # If not, we return a seed-like construction if possible.
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             # Small n, just brute force or simple
             return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific case: q=3, n=7, radius=1
    total_words = 3**7
    words = list(range(total_words))
    
    # Precompute neighbors for each word
    # A word x covers x and all words y such that dist(x, y) <= 1
    # dist <= 1 means they differ in at most 1 position.
    
    # To speed up neighbor generation, we can precompute the "sphere" for each index.
    # However, 2187 * 15 operations is tiny.
    
    def get_neighbors(x):
        # x is an integer representing the base-3 number
        # Convert to list of digits
        digits = [0] * n
        temp = x
        for i in range(n):
            digits[i] = temp % 3
            temp //= 3
        # Note: digits[0] is LSB.
        
        neighbors = [x]
        for i in range(n):
            orig = digits[i]
            for d in range(3):
                if d == orig:
                    continue
                # Change digit i to d
                # Calculate new value
                # x + (d - orig) * 3**i
                new_val = x + (d - orig) * (3 ** i)
                neighbors.append(new_val)
        return neighbors

    # Precompute all neighborhoods
    # This is a list of lists.
    # We also need to know, for a given word, which codewords cover it?
    # Or rather, we run a greedy set cover.
    
    # Greedy Set Cover Algorithm
    # 1. Start with all words uncovered.
    # 2. While uncovered words exist:
    #    a. Find the word w that covers the most uncovered words.
    #    b. Add w to the code.
    #    c. Mark all words in N(w) as covered.
    
    # To make step (a) efficient:
    # We maintain a count of uncovered words in the neighborhood of each candidate codeword.
    # Initially, count[w] = |N(w)| = 15 for all w.
    # When we select a codeword c, we decrement count[w] for all w such that c in N(w).
    # This is equivalent to: for all u in N(c), for all w in N(u) \cap N(c)? No.
    # If we select c, we cover N(c).
    # For any other candidate w, if w is in N(c), then w's neighborhood overlaps with N(c).
    # Specifically, the number of *newly* covered words by w decreases.
    # The number of words covered by w that are *also* covered by c is |N(w) \cap N(c)|.
    # So new_count[w] = old_count[w] - |N(w) \cap N(c)|?
    # Wait, the standard update is:
    # Let S be the set of uncovered words.
    # gain(w) = |N(w) \cap S|.
    # Initially S = AllWords, gain(w) = 15.
    # When we pick c, S becomes S \ N(c).
    # For any w, new_gain(w) = |N(w) \cap (S \ N(c))| = |N(w) \cap S| - |N(w) \cap N(c) \cap S|.
    # This is hard to update efficiently without knowing S.
    
    # Alternative: Use a Max-Heap with lazy deletion.
    # Push ( -gain, word ) into heap.
    # Pop max. If it's still valid (i.e., its calculated gain matches current uncovered count), select it.
    # Otherwise, discard and pop next.
    # To calculate "current uncovered count" for a word w, we need to check how many of its neighbors are in S.
    # This requires maintaining S as a set or bitset.
    
    # Given n=7, total words=2187.
    # We can represent S as a Python set of integers.
    # |S| starts at 2187.
    # Each step removes ~15 words.
    # ~2187/15 = 146 steps.
    # In each step, we might pop several invalid items from the heap.
    # We need to be able to calculate gain(w) quickly.
    # gain(w) = sum(1 for u in N(w) if u in S).
    # This takes O(15) time.
    # Heap size will grow as we push updates?
    # Actually, we can just re-calculate gain for the top item.
    # If top item's gain < stored_gain, it's stale. Pop it, recalc, push back?
    # Or just pop and if stale, discard. But we need to push the updated value?
    # Standard lazy deletion:
    # 1. Push (-initial_gain, w) for all w.
    # 2. Pop (-g, w).
    # 3. Check if g == current_gain(w).
    #    - If yes, select w. Update S.
    #    - If no, w is stale. Do we push a new entry?
    #      If we don't push a new entry, we might lose w entirely if its gain drops but it's still the best?
    #      No, if its gain drops, it's less attractive. But it might still be better than others.
    #      So we should push (-current_gain(w), w) back into the heap.
    
    # Let's implement this.
    
    S = set(range(total_words))
    
    # Precompute neighbors list for all words to avoid recalculating digits
    neighbors = [get_neighbors(i) for i in range(total_words)]
    
    # Initial heap
    heap = []
    for w in range(total_words):
        # Gain is 15 for all
        heapq.heappush(heap, (-15, w))
    
    code = []
    
    while S:
        if time.time() > deadline:
            break
            
        # Pop max gain
        neg_g, w = heapq.heappop(heap)
        g = -neg_g
        
        # Calculate current gain
        cur_gain = 0
        for u in neighbors[w]:
            if u in S:
                cur_gain += 1
        
        if cur_gain == 0:
            # Should not happen if S is not empty and we haven't covered everything
            # But if w covers nothing new, skip
            continue
            
        if g != cur_gain:
            # Stale entry. Push updated value.
            heapq.heappush(heap, (-cur_gain, w))
            continue
        
        # Valid max gain candidate
        code.append(w)
        
        # Update S
        for u in neighbors[w]:
            if u in S:
                S.remove(u)
                
        # Optimization: We don't strictly need to update the heap for all affected words immediately.
        # The lazy deletion handles it when they are popped.
        
    # Convert integer codes back to lists of digits
    result = []
    for c in code:
        digits = [0] * n
        temp = c
        for i in range(n):
            digits[i] = temp % 3
            temp //= 3
        result.append(digits)
        
    return result

# Note: The function construct is the entry point.
# The code above defines construct.
