import heapq
import itertools
import time

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # For this specific task, q=3, n=8, radius=1
    # But we should handle general inputs if possible, or at least q=3, n=8.
    # The prompt says "The evaluator passes this task's alphabet size q, length n and covering radius."
    # It implies we might get different parameters?
    # "Keep the function names and signatures of the seed program."
    # "Each evaluation must finish within 120 s."
    # If n is large, this greedy approach might be too slow or memory heavy.
    # However, the task is specifically "Football pool problem for 8 matches... K_3(8,1)".
    # It's safe to assume q=3, n=8, radius=1.
    # If n is different, we should fall back to a simpler method or handle it generally.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback to the original direct sum method if parameters differ
        # This ensures we don't crash on unexpected inputs, though the task is fixed.
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # Greedy Set Cover for q=3, n=8, R=1
    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage
    # codeword_to_words: list of lists of word indices
    # word_to_codewords: list of lists of codeword indices
    
    codeword_to_words = []
    word_to_codewords = [[] for _ in range(total_words)]
    
    # Helper to get neighbors
    def get_neighbors(w, q, n, r):
        neighbors = [w]
        if r > 0:
            for i in range(n):
                for val in range(q):
                    if val == w[i]: continue
                    new_w_list = list(w)
                    new_w_list[i] = val
                    new_w = tuple(new_w_list)
                    neighbors.append(new_w)
        return neighbors

    # For R=1, each codeword covers 1 + n*(q-1) = 1 + 8*2 = 17 words.
    # We can hardcode this logic for speed or use the generic function.
    # Given the small size, generic is fine, but let's optimize slightly.
    
    for w in all_words:
        idx = word_to_idx[w]
        covered_words = []
        # Center
        covered_words.append(idx)
        # Radius 1 neighbors
        w_list = list(w)
        for i in range(n):
            original_val = w_list[i]
            for val in range(q):
                if val == original_val: continue
                w_list[i] = val
                covered_words.append(word_to_idx[tuple(w_list)])
            w_list[i] = original_val
            
        codeword_to_words.append(covered_words)
        for cw_idx in covered_words:
            word_to_codewords[cw_idx].append(idx)

    # Initialize
    current_counts = [17] * total_words  # Each covers 17 words initially
    covered = [False] * total_words
    heap = []
    for i in range(total_words):
        heapq.heappush(heap, (-current_counts[i], i))
        
    result = []
    
    while True:
        if time.time() > deadline:
            break
            
        neg_count, c_idx = heapq.heappop(heap)
        count = -neg_count
        
        # Check for stale entry
        if count != current_counts[c_idx]:
            continue
            
        if count == 0:
            break
            
        # Select this codeword
        result.append(all_words[c_idx])
        
        # Update coverage
        for w_idx in codeword_to_words[c_idx]:
            if covered[w_idx]:
                continue
            covered[w_idx] = True
            
            # Update counts for all codewords covering this word
            for cp_idx in word_to_codewords[w_idx]:
                current_counts[cp_idx] -= 1
                heapq.heappush(heap, (-current_counts[cp_idx], cp_idx))
                
    # Convert to list of lists as required
    return [list(w) for w in result]
