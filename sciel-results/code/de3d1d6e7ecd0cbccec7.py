import random
import time
import itertools

def construct(q, n, radius):
    """
    Constructs a ternary covering code K_q(n, radius) using a greedy set-cover approach.
    """
    start_time = time.time()
    deadline = 115  # Safety margin
    
    total_words = q ** n
    
    # Map integer to word (list of q-ary digits)
    def int_to_word(idx):
        word = [0] * n
        for i in range(n):
            word[n - 1 - i] = idx % q
            idx //= q
        return word

    # Precompute coverage sets for all words
    # coverage[w] is a set of indices covered by word w
    coverage = [None] * total_words
    
    # Helper to get neighbors within radius
    # Since radius is small (1), we can just iterate
    # For general radius, we'd need a more general method, but task specifies radius 1 for this specific case mostly.
    # However, construct takes radius as arg.
    
    # Efficient neighbor generation for small radius
    def get_neighbors(idx, q, n, radius):
        # Convert to list
        word = [0] * n
        temp = idx
        for i in range(n):
            word[n - 1 - i] = temp % q
            temp //= q
            
        neighbors = {idx}
        if radius >= 1:
            # Change one position
            for pos in range(n):
                original = word[pos]
                for new_val in range(q):
                    if new_val == original:
                        continue
                    word[pos] = new_val
                    # Convert back to int
                    new_idx = 0
                    for i in range(n):
                        new_idx = new_idx * q + word[i]
                    neighbors.add(new_idx)
        # Note: For radius > 1, this is incomplete. 
        # But the task is specifically K_3(6,1). 
        # If radius > 1 is passed, we might need a BFS. 
        # Given the context "Football pool problem... radius 1", we assume radius=1 is the main case.
        # To be safe, if radius > 1, we can fall back to a slower method or just handle radius 1 optimally.
        # Let's implement a generic BFS for neighbors if radius > 1, though unlikely.
        
        if radius > 1:
            # BFS from idx
            visited = {idx}
            queue = [(idx, 0)]
            while queue:
                cur_idx, dist = queue.pop(0)
                if dist == radius:
                    continue
                # Get immediate neighbors of cur_idx (distance 1)
                cur_word = [0] * n
                temp = cur_idx
                for i in range(n):
                    cur_word[n - 1 - i] = temp % q
                    temp //= q
                    
                for pos in range(n):
                    original = cur_word[pos]
                    for new_val in range(q):
                        if new_val == original:
                            continue
                        cur_word[pos] = new_val
                        new_idx = 0
                        for i in range(n):
                            new_idx = new_idx * q + cur_word[i]
                        if new_idx not in visited:
                            visited.add(new_idx)
                            queue.append((new_idx, dist + 1))
            neighbors = visited
        return neighbors

    # Precompute coverages
    for i in range(total_words):
        coverage[i] = get_neighbors(i, q, n, radius)

    best_codewords = None
    best_size = float('inf')
    
    # Run greedy multiple times with random tie-breaking
    # 100 iterations should be plenty and fast
    for _ in range(200):
        if time.time() > deadline:
            break
            
        uncovered = set(range(total_words))
        codewords = []
        
        while uncovered:
            best_w = -1
            best_count = 0
            candidates = []
            
            for w in range(total_words):
                # Optimization: if w is already in codewords, skip? 
                # No, we can pick the same w again? No, codewords must be distinct.
                # So we should track chosen codewords.
                pass
            
            # Better: iterate all, find max overlap with uncovered
            # If tie, add to candidates
            for w in range(total_words):
                # Skip if already chosen
                # Checking membership in list is slow. Use a set for chosen.
                # But we just need to pick a new one.
                # Actually, if we pick w, we remove its coverage. 
                # If we pick w again, it covers 0 new words.
                # So we can just check overlap.
                
                overlap = len(coverage[w] & uncovered)
                if overlap > best_count:
                    best_count = overlap
                    best_w = w
                    candidates = [w]
                elif overlap == best_count and overlap > 0:
                    candidates.append(w)
            
            if not candidates:
                break # Should not happen if uncovered is not empty and coverage is valid
                
            # Pick randomly from candidates
            w = random.choice(candidates)
            codewords.append(w)
            uncovered -= coverage[w]
            
        if len(codewords) < best_size:
            best_size = len(codewords)
            best_codewords = codewords.copy()
            
            # If we hit the theoretical lower bound or known optimum, we can stop?
            # Known optimum for K_3(6,1) is 73.
            # If we find 73, it's likely optimal.
            if best_size == 73:
                break
                
    if best_codewords is None:
        # Fallback: return all words? Or a simple construction?
        # Simple construction: direct sum of tetracode
        if n >= 4 and q == 3:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
             return [int_to_word(i) for i in range(total_words)]
             
    return [int_to_word(w) for w in best_codewords]
