import itertools
import time
import random

def construct(q, n, radius):
    # This problem is specific to q=3, n=8, radius=1.
    # The seed solution is 486.
    # We attempt to improve it using local search with swaps.
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (though task specifies this one)
        # Use a simple greedy or direct sum if applicable
        if n == 4 and q == 3 and radius == 1:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return tetracode
        # Generic fallback: return seed-like construction if possible
        # For now, just return the seed for the specific case, else empty or error?
        # The task contract says "evaluator passes this task's alphabet size...".
        # So we assume q=3, n=8, radius=1.
        pass

    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Generate the initial code (Direct Sum of Tetracode)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Map words to indices 0..6560
    # Index = w0 * 3^7 + w1 * 3^6 + ... + w7
    def word_to_idx(w):
        idx = 0
        for v in w:
            idx = idx * 3 + v
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % 3
            idx //= 3
        return w

    # Precompute stars for all 6561 words
    # star[i] is a list of indices at distance <= 1 from i
    total_words = 3 ** n
    stars = []
    for i in range(total_words):
        w = idx_to_word(i)
        star = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(3):
                if val != orig:
                    w[pos] = val
                    star.append(word_to_idx(w))
            w[pos] = orig
        stars.append(star)
        
    # Initial code
    initial_code_indices = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = h + list(tail)
            initial_code_indices.add(word_to_idx(w))
            
    current_code = set(initial_code_indices)
    
    # Initialize cover counts
    cover_count = [0] * total_words
    for c in current_code:
        for p in stars[c]:
            cover_count[p] += 1
            
    best_code_size = len(current_code)
    best_code_set = set(current_code)
    
    # Local Search
    # We try to swap words to find redundancies, or just remove redundancies if they appear.
    # Since the initial code has no single-word redundancies, we need swaps.
    
    # Optimization: 
    # A swap (remove c_old, add c_new) is valid if:
    # 1. c_new not in code
    # 2. All points p where cover_count[p] == 1 (and c_old is the cover) are in stars[c_new]
    
    # To make this faster, we can identify "critical" points for each codeword.
    # But calculating this dynamically is expensive.
    
    # Simple approach: Random swaps.
    # If a swap is valid, we perform it. Then we check for any removable words.
    
    random.seed(42)
    
    # List of all indices for random selection
    all_indices = list(range(total_words))
    
    # To speed up "is c_new in code", use the set.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        if not current_code:
            break
            
        # Pick a random codeword to remove
        c_old = random.choice(list(current_code))
        
        # Identify points uniquely covered by c_old
        # These are points p in stars[c_old] where cover_count[p] == 1
        # If there are any such points, c_new MUST cover them.
        
        # Instead of iterating all stars[c_old] (17 items), we can just try a random c_new
        # and check validity.
        
        # Pick a random c_new
        c_new = random.choice(all_indices)
        if c_new in current_code:
            continue
            
        # Check validity of swap
        # We need to ensure that for all p in stars[c_old], 
        # (cover_count[p] - 1 + (1 if p in stars[c_new] else 0)) >= 1
        # i.e., cover_count[p] >= 1 - (1 if p in stars[c_new] else 0)
        # If cover_count[p] == 1, we need p in stars[c_new].
        # If cover_count[p] > 1, it's fine.
        
        valid = True
        for p in stars[c_old]:
            if cover_count[p] == 1:
                # p is uniquely covered by c_old.
                # c_new must cover p.
                if p not in stars[c_new]:
                    valid = False
                    break
        
        if valid:
            # Perform swap
            current_code.remove(c_old)
            current_code.add(c_new)
            
            # Update cover counts
            for p in stars[c_old]:
                cover_count[p] -= 1
            for p in stars[c_new]:
                cover_count[p] += 1
                
            # Try to remove redundant words
            # A word c is redundant if for all p in stars[c], cover_count[p] > 1
            # We can scan the current code for such a word.
            # To avoid O(N) scan every time, we can just check a few random ones or scan all if small.
            # N is ~486. Scanning all is fast.
            
            removed_any = False
            for c in list(current_code):
                is_redundant = True
                for p in stars[c]:
                    if cover_count[p] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove it
                    current_code.remove(c)
                    for p in stars[c]:
                        cover_count[p] -= 1
                    removed_any = True
                    break # Restart redundancy check
            
            if len(current_code) < best_code_size:
                best_code_size = len(current_code)
                best_code_set = set(current_code)
                
    # Convert best_code_set back to list of lists
    result = [idx_to_word(i) for i in best_code_set]
    
    # Ensure distinctness (set guarantees it)
    # Ensure validity? 
    # The local search maintains validity by construction (only valid swaps/removals).
    # Initial code is valid.
    
    return result

