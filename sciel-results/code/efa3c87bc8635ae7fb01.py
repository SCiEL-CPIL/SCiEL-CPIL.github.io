import numpy as np
import itertools
import time

# Precomputation for q=3, n=6
Q = 3
N = 6
NUM_WORDS = Q ** N

# Generate all words
all_words = np.array(list(itertools.product(range(Q), repeat=N)), dtype=np.int32)

# Precompute a mapping from word tuple to index
word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}

# Precompute balls: for each codeword index, store the indices of words it covers
# ball_indices[c] is a list/array of word indices covered by codeword c
ball_indices = []
for i in range(NUM_WORDS):
    c = all_words[i]
    ball = [i]
    for pos in range(N):
        for val in range(Q):
            if val == c[pos]:
                continue
            neighbor = c.copy()
            neighbor[pos] = val
            idx = word_to_idx[tuple(neighbor)]
            ball.append(idx)
    ball_indices.append(np.array(ball, dtype=np.int32))

# Convert to list for faster access in loops if needed, but NumPy arrays are good
# We will use ball_indices as a list of NumPy arrays

def get_initial_code():
    """
    Generate an initial code. 
    We use a greedy approach with randomization to get a diverse starting point,
    or use the tetracode construction (81 words) which is a known valid cover.
    Here we use a randomized greedy to potentially get a smaller starting point,
    or just use the tetracode as a safe baseline.
    """
    # Tetracode construction: 81 words
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=N - 4)]
    return code

def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        if q is None or n is None or radius is None:
            return []
        # Simple greedy for general case
        all_w = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
        nw = q ** n
        wti = {tuple(w): i for i, w in enumerate(all_w)}
        # Build ball indices
        balls = []
        for i in range(nw):
            c = all_w[i]
            ball = [i]
            for pos in range(n):
                for val in range(q):
                    if val == c[pos]:
                        continue
                    nb = c.copy()
                    nb[pos] = val
                    ball.append(wti[tuple(nb)])
            balls.append(np.array(ball, dtype=np.int32))
        
        covered = np.zeros(nw, dtype=bool)
        code = []
        while not np.all(covered):
            # Greedy: pick word covering most uncovered
            scores = np.array([np.sum(covered[balls[i]] == False) for i in range(nw)])
            c = int(np.argmax(scores))
            if scores[c] == 0:
                break
            code.append(c)
            covered[balls[c]] = True
        return [list(all_w[c]) for c in code]

    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    # Start with tetracode (81 words)
    best_code = get_initial_code()
    best_size = len(best_code)
    
    # Convert best_code to list of indices for internal processing
    # We work with indices into all_words
    best_code_idx = [word_to_idx[tuple(w)] for w in best_code]
    
    # Main search loop
    # We use a simulated annealing-like or random restart local search
    # Here we use a single long run with random restarts of the local search state if stuck
    
    current_code_idx = best_code_idx.copy()
    
    # Precompute a set for fast membership check? 
    # Since NUM_WORDS is 729, a boolean array is faster.
    
    def reduce_code(code_idx):
        """Remove redundant codewords. Returns new list of indices."""
        if not code_idx:
            return []
        # Count coverage
        cov = np.zeros(NUM_WORDS, dtype=np.int32)
        for c in code_idx:
            cov[ball_indices[c]] += 1
        
        new_code = []
        # Try to remove each codeword
        # We iterate and check if removing c leaves any word with cov 0
        # To be efficient, we can just try removing in random order
        indices_to_check = list(range(len(code_idx)))
        np.random.shuffle(indices_to_check)
        
        # We need to be careful: removing one might make another redundant
        # Simple approach: loop until no change
        changed = True
        while changed:
            changed = False
            # Recalculate cov if we removed something? 
            # Better: maintain cov.
            # But cov depends on the current set.
            # Let's just recalculate cov at the start of each 'changed' loop iteration?
            # Or update incrementally.
            
            # Incremental update is better.
            # However, the 'while changed' loop with full recalculation is simpler and 
            # 729 words is small.
            
            pass # We will implement a simpler version below
        
        # Simpler reduction:
        # 1. Compute cov counts for current set.
        # 2. Iterate through codewords. If removing c_i keeps all counts > 0, remove it.
        # 3. Repeat until no more removals.
        
        code_list = list(code_idx)
        while True:
            cov = np.zeros(NUM_WORDS, dtype=np.int32)
            for c in code_list:
                cov[ball_indices[c]] += 1
            
            removed_any = False
            # Try removing in random order to break symmetry
            order = list(range(len(code_list)))
            np.random.shuffle(order)
            
            for i in order:
                c = code_list[i]
                # Check if any word in ball(c) has cov 1
                # If min(cov[ball_indices[c]]) > 1, we can remove c
                if np.min(cov[ball_indices[c]]) > 1:
                    code_list.pop(i)
                    removed_any = True
                    break # Restart the loop with new list
            
            if not removed_any:
                break
        
        return code_list

    # Local Search Step: 1-for-1 and 1-for-2 swaps
    def local_search(code_idx):
        """
        Perform local search on code_idx.
        Returns improved code_idx and best size found during this call.
        """
        current = reduce_code(code_idx)
        best_in_ls = len(current)
        
        max_iterations = 500 # Cap iterations per LS call to ensure progress
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # 1. Try 1-for-1 swap
            if len(current) > 0:
                # Pick a random codeword to replace
                i = np.random.randint(len(current))
                c_old = current[i]
                
                # Pick a random new codeword not in current
                # To be fast, pick random index, check if in current
                # Since current is small (<100), linear scan is fine
                in_current = set(current)
                
                c_new = np.random.randint(NUM_WORDS)
                if c_new in in_current:
                    continue
                
                # Test swap
                test_code = current.copy()
                test_code[i] = c_new
                
                # Check if test_code is a valid cover
                # We can check by seeing if union of balls covers all words
                # Efficient check: 
                # cov = sum of balls
                # min(cov) > 0
                
                cov = np.zeros(NUM_WORDS, dtype=np.int32)
                for c in test_code:
                    cov[ball_indices[c]] += 1
                
                if np.min(cov) > 0:
                    # Valid cover
                    # Reduce it
                    reduced = reduce_code(test_code)
                    if len(reduced) <= len(current):
                        current = reduced
                        if len(current) < best_in_ls:
                            best_in_ls = len(current)
                    # Note: Even if len doesn't decrease, we might be in a different basin.
                    # But for this heuristic, we only accept if size doesn't increase (after reduction).
                    # Actually, accepting equal size moves allows exploration.
            
            # 2. Try 1-for-2 swap
            if len(current) > 0:
                i = np.random.randint(len(current))
                c_old = current[i]
                
                # Pick two new codewords
                in_current = set(current)
                c_new1 = np.random.randint(NUM_WORDS)
                c_new2 = np.random.randint(NUM_WORDS)
                
                if c_new1 in in_current or c_new2 in in_current or c_new1 == c_new2:
                    continue
                
                # Test swap: remove c_old, add c_new1, c_new2
                test_code = current.copy()
                test_code.pop(i)
                test_code.append(c_new1)
                test_code.append(c_new2)
                
                # Check coverage
                cov = np.zeros(NUM_WORDS, dtype=np.int32)
                for c in test_code:
                    cov[ball_indices[c]] += 1
                
                if np.min(cov) > 0:
                    # Valid cover
                    reduced = reduce_code(test_code)
                    # Accept if size is smaller, or equal (to explore)
                    if len(reduced) <= len(current):
                        current = reduced
                        if len(current) < best_in_ls:
                            best_in_ls = len(current)
        
        return current, best_in_ls

    # Main loop
    current_code_idx = best_code_idx.copy()
    
    # Run local search for a while
    # We can also do random restarts if we get stuck
    restarts = 0
    max_restarts = 50
    
    while time.time() < deadline:
        # Perform local search
        current_code_idx, ls_best = local_search(current_code_idx)
        
        if len(current_code_idx) < best_size:
            best_size = len(current_code_idx)
            best_code_idx = current_code_idx.copy()
            # If we hit 73, we can stop or try to go lower
            if best_size <= 73:
                # We found the best known upper bound. 
                # We can continue to see if we can find 72 or 71, but it's unlikely.
                # For now, we accept 73 as a strong result.
                pass
        
        # Check for stagnation? 
        # If we don't improve for a while, restart from a random perturbation of best
        # Simple strategy: just keep going. The randomization in LS handles it.
        
        # Occasionally, try to start from a different initial code?
        # If best_size is still 81, maybe try a greedy random start.
        if best_size > 75 and restarts < max_restarts:
             # Generate a random greedy code
             covered = np.zeros(NUM_WORDS, dtype=bool)
             random_code = []
             while not np.all(covered):
                 # Pick random word, add it
                 # Or pick word covering most uncovered? 
                 # Let's pick random to diversify
                 c = np.random.randint(NUM_WORDS)
                 random_code.append(c)
                 covered[ball_indices[c]] = True
             
             # Reduce it
             random_code = reduce_code(random_code)
             
             # If this random code is better or equal, use it as new starting point for LS
             if len(random_code) <= best_size:
                 current_code_idx = random_code
                 restarts += 1

    # Convert best_code_idx back to list of lists
    final_code = [list(all_words[c]) for c in best_code_idx]
    return final_code
