import time
import itertools
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Handle trivial or small cases
    if n == 0:
        return [[]]
        
    # Specific optimization for the known hard case: q=3, n=6, radius=1
    # The problem statement highlights this case specifically.
    if q == 3 and n == 6 and radius == 1:
        # 1. Generate Direct Sum Code (Tetracode x {0,1,2}^2)
        # Tetracode: (a, b, a+b, a+2b) mod 3
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        
        init_code = []
        for h in tetracode:
            for t1 in range(3):
                for t2 in range(3):
                    init_code.append(tuple(h + [t1, t2]))
        
        # Convert to indices for fast processing
        # Word w -> index idx = w[0]*3^5 + w[1]*3^4 + ... + w[5]
        def word_to_idx(w):
            idx = 0
            for i in range(n):
                idx = idx * 3 + w[i]
            return idx

        def idx_to_word(idx):
            w = [0] * n
            for i in range(n - 1, -1, -1):
                w[i] = idx % 3
                idx //= 3
            return w

        # Precompute neighbors (ball of radius 1) for all 729 words
        # neighbors[idx] = list of indices covered by this word
        all_indices = list(range(729))
        neighbors = [[] for _ in range(729)]
        for idx in all_indices:
            w = idx_to_word(idx)
            ball = [idx]
            for i in range(n):
                orig = w[i]
                for v in range(3):
                    if v == orig:
                        continue
                    nw = w[:]
                    nw[i] = v
                    ball.append(word_to_idx(nw))
            neighbors[idx] = ball

        # Helper to calculate coverage counts
        def get_coverage(code_indices):
            counts = [0] * 729
            for c_idx in code_indices:
                for nb in neighbors[c_idx]:
                    counts[nb] += 1
            return counts

        # Greedy Removal
        # Start with the direct sum code
        current_code_indices = [word_to_idx(c) for c in init_code]
        
        def try_remove_redundant(code_indices):
            changed = True
            while changed:
                if time.time() > deadline:
                    return code_indices
                changed = False
                counts = get_coverage(code_indices)
                # Find removable codewords
                removable = []
                for c_idx in code_indices:
                    # Check if all neighbors have count > 1
                    can_remove = True
                    for nb in neighbors[c_idx]:
                        if counts[nb] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        removable.append(c_idx)
                
                if removable:
                    # Remove one randomly to avoid getting stuck in a specific order
                    # Or remove all? Removing one by one is safer for local minima escape later.
                    # Actually, removing all redundant at once is faster and deterministic.
                    # Let's remove all redundant in one pass.
                    to_remove = set(removable)
                    code_indices = [c for c in code_indices if c not in to_remove]
                    changed = True
            return code_indices

        current_code_indices = try_remove_redundant(current_code_indices)
        best_size = len(current_code_indices)
        best_code = current_code_indices[:]

        # Local Search: Swap
        # If we are stuck at > 73, try swapping a codeword with a random other word.
        # If the swap allows the code to remain valid (or improves removal potential), accept.
        # Actually, a valid swap must maintain coverage.
        # Strategy:
        # 1. Try to remove more.
        # 2. If stuck, pick a codeword C to remove.
        # 3. Find a word W not in code such that adding W covers the "holes" created by removing C.
        #    This is complex.
        
        # Simpler Local Search:
        # 1. Compute coverage.
        # 2. Identify "critical" words (count == 1).
        # 3. If no critical words, we can remove any codeword. (Already handled by greedy).
        # 4. If critical words exist, we are stuck in a local minimum for removal.
        #    We need to change the code.
        #    Try: Remove C, Add W.
        #    Condition: Coverage of (Code - C + W) is valid.
        #    This is expensive to check for all W.
        
        # Alternative: Random Restart with Perturbation
        # Since the space is small (729), we can try a few random restarts of the greedy process
        # starting from slightly perturbed initial codes.
        
        # However, the known optimal is 73. The direct sum is 81.
        # Greedy removal usually gets to 74 or 75.
        # To get 73, we need a specific structure.
        
        # Let's try a more aggressive local search:
        # 1. Get coverage counts.
        # 2. Find codewords with "low utility" (cover many words that are covered by others).
        # 3. Try to swap them.
        
        # Given the time limit, a robust approach is:
        # 1. Greedy remove (done).
        # 2. If size > 73, try random swaps for a fixed number of iterations.
        
        if len(current_code_indices) > 73:
            for _ in range(50): # Try a few restarts/swaps
                if time.time() > deadline - 5:
                    break
                
                # Perturb: Remove a random codeword, add a random word, then greedy remove
                test_code = current_code_indices[:]
                if not test_code:
                    break
                
                # Remove a random codeword
                idx_to_remove = random.randint(0, len(test_code) - 1)
                c_removed = test_code[idx_to_remove]
                test_code.pop(idx_to_remove)
                
                # Add a random word not in code
                # We want to add a word that might help cover the holes.
                # But random is okay for diversity.
                new_word_idx = random.randint(0, 728)
                if new_word_idx not in test_code:
                    test_code.append(new_word_idx)
                
                # Greedy remove on the new code
                test_code = try_remove_redundant(test_code)
                
                if len(test_code) < len(current_code_indices):
                    current_code_indices = test_code
                    best_size = len(current_code_indices)
                    best_code = current_code_indices[:]
                    
                # Early exit if we hit the known lower bound or best known
                if best_size <= 73:
                    break

        # Convert back to lists of ints
        return [idx_to_word(idx) for idx in best_code]

    # General Fallback for other q, n, radius
    # Greedy Set Cover
    # 1. Generate all words
    # 2. While not covered, pick word covering most uncovered
    
    all_words = list(itertools.product(range(q), repeat=n))
    
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)
    
    def get_ball_general(w):
        ball = [w]
        for i in range(n):
            orig = w[i]
            for v in range(q):
                if v == orig:
                    continue
                nw = list(w)
                nw[i] = v
                ball.append(tuple(nw))
        return ball

    # Precompute balls for all words if space is small
    if q ** n <= 10000:
        balls = {w: get_ball_general(w) for w in all_words}
    else:
        balls = None

    code = []
    uncovered = set(all_words)
    
    while uncovered and time.time() < deadline:
        best_word = None
        best_cover = 0
        # Sample a subset of words if space is large to save time
        candidates = all_words if len(all_words) < 1000 else random.sample(all_words, 100)
        
        for w in candidates:
            if w in code:
                continue
            if balls:
                b = balls[w]
            else:
                b = get_ball_general(w)
            
            cover_count = 0
            for u in b:
                if u in uncovered:
                    cover_count += 1
            
            if cover_count > best_cover:
                best_cover = cover_count
                best_word = w
        
        if best_word is None or best_cover == 0:
            break
            
        code.append(best_word)
        if balls:
            b = balls[best_word]
        else:
            b = get_ball_general(best_word)
        
        for u in b:
            uncovered.discard(u)
            
    return [list(w) for w in code]
