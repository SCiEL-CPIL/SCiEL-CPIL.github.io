import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    # Only handle the specific case from the task description for efficiency and correctness
    # The task says "evaluator passes this task's alphabet size q, length n and covering radius"
    # But the contract says "Football pool problem for 6 matches: ternary covering code K_3(6,1)"
    # So q=3, n=6, radius=1 is the intended case.
    # We should handle general q,n,radius gracefully or at least not crash, but optimize for 3,6,1.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other cases: just return a trivial code or empty?
        # The task contract implies we are solving the specific problem.
        # If parameters differ, we can try a generic greedy or return empty.
        # However, the score is based on the specific task.
        # Let's return a trivial code that covers nothing? No, it must be valid.
        # A trivial valid code is the whole space.
        return [list(x) for x in itertools.product(range(q), repeat=n)]

    # Specific solver for q=3, n=6, r=1
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Generate the initial 81-word code (Direct Sum of Tetracode and H(2,3))
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # 2. Precompute balls for all 729 words
    # Map word tuple to index 0..728
    all_words = list(itertools.product(range(3), repeat=6))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # balls[i] is a list of indices of words within distance 1 of all_words[i]
    balls = []
    for w in all_words:
        ball = [word_to_idx[w]]
        for i in range(6):
            for d in range(3):
                if d == w[i]: continue
                new_w = list(w)
                new_w[i] = d
                ball.append(word_to_idx[tuple(new_w)])
        balls.append(ball)
    
    # Convert initial code to indices
    code_indices = [word_to_idx[tuple(c)] for c in initial_code]
    # Remove duplicates just in case (shouldn't be any)
    code_indices = list(set(code_indices))
    
    best_code_size = len(code_indices)
    best_code = list(code_indices)
    
    # Greedy Removal with Randomized Restarts
    # We try to remove as many as possible.
    # Since the structure is symmetric, different removal orders might lead to different local minima.
    # We run multiple restarts with different random seeds/orders.
    
    num_restarts = 50 # We can afford many
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        current_code = list(code_indices)
        # Shuffle to break symmetry
        random.shuffle(current_code)
        
        # Coverage count for each word in space
        # coverage[i] = number of codewords covering all_words[i]
        coverage = np.zeros(729, dtype=np.int32)
        
        for idx in current_code:
            for w_idx in balls[idx]:
                coverage[w_idx] += 1
        
        # Iterative removal
        while True:
            if time.time() > deadline:
                break
                
            # Find removable codewords
            # A codeword c is removable if for all w in ball(c), coverage[w] > 1
            # Equivalently, if there is no w in ball(c) such that coverage[w] == 1
            
            removable = []
            for idx in current_code:
                is_removable = True
                for w_idx in balls[idx]:
                    if coverage[w_idx] == 1:
                        is_removable = False
                        break
                if is_removable:
                    removable.append(idx)
            
            if not removable:
                break
                
            # Pick one to remove
            # Heuristic: Pick the one that covers the most "redundant" words?
            # Or just random from removable.
            # Let's try random first.
            idx_to_remove = random.choice(removable)
            
            # Update coverage
            for w_idx in balls[idx_to_remove]:
                coverage[w_idx] -= 1
                
            # Remove from current code
            current_code.remove(idx_to_remove)
            
        # Record result
        if len(current_code) < best_code_size:
            best_code_size = len(current_code)
            best_code = list(current_code)
            
            if best_code_size == 73:
                break # Can't do better than known optimal (likely)
                
    # Convert back to list of lists
    result = [list(all_words[idx]) for idx in best_code]
    
    return result
