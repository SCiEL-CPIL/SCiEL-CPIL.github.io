import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses a greedy set-cover approach with a time budget.
    """
    # Parameters for the specific problem: q=3, n=7, radius=1
    # The evaluator passes q, n, radius. We handle the general case but optimize for the known instance.
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Special case handling for the specific ternary 7-match problem
    # If q=3, n=7, radius=1, we can use specific heuristics.
    # The seed program uses a direct sum construction which yields 9 * 3^(7-4) = 243 words.
    # We want to do better.
    
    # General greedy approach:
    # 1. Start with an empty set of codewords.
    # 2. While there are uncovered words:
    #    a. Choose a candidate codeword that covers the maximum number of uncovered words.
    #    b. Add it to the set and mark its covered words.
    #
    # Optimization:
    # - Use a precomputed table of neighbors for speed? For 3^7=2187, it's small enough.
    # - We can represent words as integers 0..3^n-1.
    # - Precompute the coverage (ball of radius r) for each possible word?
    #   2187 balls of size 1 + 7*(2) = 15 words each. Total operations ~ 2187*15 = 32k. Very fast.
    
    total_words = q ** n
    if total_words > 1000000:
        # Fallback for larger spaces: use the seed's constructive method if possible, or random
        # For now, return the seed's direct sum construction if parameters match its assumptions
        if q == 3 and n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            import itertools
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # Generic fallback: random sampling
            codewords = []
            seen = set()
            while len(codewords) < 100:
                w = [random.randint(0, q-1) for _ in range(n)]
                if tuple(w) not in seen:
                    seen.add(tuple(w))
                    codewords.append(w)
            return codewords
    
    # Convert words to integers for efficient processing
    # word_to_int: tuple of digits -> int
    # int_to_word: int -> list of digits
    
    def to_int(word):
        val = 0
        for d in word:
            val = val * q + d
        return val
    
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word
    
    # Precompute all words as integers
    all_words = list(range(total_words))
    
    # Precompute the ball (neighbors within radius) for each word
    # ball[w] = list of words in the ball of radius 'radius' around w
    # To compute neighbors:
    # For each position i, and each symbol s != w[i], the word with s at i and same elsewhere is dist 1.
    # For radius 1, this is w itself + n*(q-1) neighbors.
    
    balls = [None] * total_words
    for w in all_words:
        word = to_word(w)
        neighbors = [w]
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for s in range(q):
                    if s == orig:
                        continue
                    word[pos] = s
                    neighbors.append(to_int(word))
                word[pos] = orig
        # For radius > 1, we would need to expand, but radius=1 is the target case.
        # If radius > 1, we'd need a BFS or combinatorial expansion.
        # Given the task is specifically K_3(7,1), we assume radius=1 primarily.
        # However, to be robust:
        if radius > 1:
             # Simple expansion: BFS from w up to depth 'radius'
             visited = {w}
             current = [w]
             for _ in range(radius):
                 next_frontier = []
                 for u in current:
                     u_word = to_word(u)
                     for pos in range(n):
                         orig = u_word[pos]
                         for s in range(q):
                             if s == orig:
                                 continue
                             u_word[pos] = s
                             v = to_int(u_word)
                             if v not in visited:
                                 visited.add(v)
                                 next_frontier.append(v)
                         u_word[pos] = orig
                 current = next_frontier
             neighbors = list(visited)
        
        balls[w] = neighbors
    
    # Greedy Set Cover
    uncovered = set(all_words)
    codewords = []
    
    # Precompute coverage counts for each candidate?
    # Initially, every candidate covers len(balls[c]) words.
    # As we add codewords, the "new" coverage decreases.
    
    # To speed up:
    # Maintain a count for each word: how many selected codewords cover it?
    # Or, simply iterate: for each candidate, check intersection with uncovered.
    # With 2187 words, and ~180 codewords, and ball size 15:
    # 180 * 2187 * 15 ~ 5.9 million operations. Very fast in Python.
    
    # Optimization: 
    # 1. Compute initial coverage for all candidates.
    # 2. Pick best.
    # 3. Remove covered words from `uncovered`.
    # 4. Update scores? 
    #    Simple approach: re-evaluate all candidates each step.
    #    186 steps * 2187 candidates * 15 avg ball size = ~6M checks. 
    #    Each check is `if w in uncovered`. Set lookup is O(1).
    #    This should run in < 1s.
    
    while uncovered:
        best_candidate = -1
        best_count = -1
        
        # Optional: Deadline check
        if time.time() > deadline:
            break
        
        # Evaluate all remaining candidates (or all candidates, as re-selecting a codeword is useless but harmless if we don't re-add)
        # We should not re-add a codeword already in the list.
        # Let's keep a set of selected_ints.
        
        # Iterate over all possible codewords
        for w in all_words:
            # If already selected, skip? Or just check if it adds value.
            # If we pick the same one again, new_count will be 0.
            # To avoid wasting time on selected ones, we can check membership.
            # But `codewords` is a list of ints. 
            # Let's use a set `selected_set`.
            
            # Count how many words in balls[w] are in uncovered
            # Optimization: if we've already selected w, new_count is 0.
            # Let's assume we don't re-select.
            
            cnt = 0
            for u in balls[w]:
                if u in uncovered:
                    cnt += 1
            
            if cnt > best_count:
                best_count = cnt
                best_candidate = w
            
        if best_candidate == -1 or best_count == 0:
            # Should not happen if uncovered is not empty and balls cover the space
            break
            
        codewords.append(best_candidate)
        # Remove covered words from uncovered
        for u in balls[best_candidate]:
            if u in uncovered:
                uncovered.remove(u)
        
        if time.time() > deadline:
            break
    
    # Convert back to list of lists
    result = [to_word(w) for w in codewords]
    
    # Ensure distinctness (they are distinct by construction of greedy set cover on unique candidates)
    return result
