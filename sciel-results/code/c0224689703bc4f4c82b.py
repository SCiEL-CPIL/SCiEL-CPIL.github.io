import time
import random
import numpy as np

def construct():
    # Deadline: 55 seconds to be safe
    deadline = time.time() + 55.0
    
    # Target sums: 128 to 256 inclusive
    MIN_SUM = 128
    MAX_SUM = 256
    NUM_SUMS = MAX_SUM - MIN_SUM + 1
    
    # Helper to calculate multiplicities for a given set A
    # Returns a numpy array of length NUM_SUMS
    def calc_mult(A):
        M = np.zeros(NUM_SUMS, dtype=np.int32)
        A_list = sorted(A)
        n = len(A_list)
        for i in range(n):
            a = A_list[i]
            # a + a
            s = 2 * a
            if MIN_SUM <= s <= MAX_SUM:
                M[s - MIN_SUM] += 1
            # a + b for b > a
            for j in range(i + 1, n):
                b = A_list[j]
                s = a + b
                if MIN_SUM <= s <= MAX_SUM:
                    M[s - MIN_SUM] += 1
        return M
    
    # Helper: remove element x from A and update M incrementally
    def remove_from_A(x, A, M):
        """Remove x from A, update M. Returns (new_A, new_M) or (None, None) if invalid."""
        new_M = M.copy()
        # For each a in A (including x itself for 2x case):
        # If a != x: sum = a + x, decrement
        # If a == x: sum = 2*x, decrement
        for a in A:
            s = a + x
            if MIN_SUM <= s <= MAX_SUM:
                new_M[s - MIN_SUM] -= 1
        # Check validity
        if np.min(new_M) < 1:
            return None, None
        new_A = A - {x}
        return new_A, new_M
    
    # Helper: add element x to A and update M incrementally
    def add_to_A(x, A, M):
        """Add x to A, update M. Returns (new_A, new_M)."""
        new_M = M.copy()
        for a in A:
            s = a + x
            if MIN_SUM <= s <= MAX_SUM:
                new_M[s - MIN_SUM] += 1
        s = 2 * x
        if MIN_SUM <= s <= MAX_SUM:
            new_M[s - MIN_SUM] += 1
        new_A = A | {x}
        return new_A, new_M
    
    # Better initial set: use a structured approach
    # Start with elements that cover the range well
    # Try multiple initial configurations
    initial_sets = [
        set(range(64, 129)),  # Original
        set(range(32, 97)),   # Shifted
        set(range(0, 129, 2)),  # Every other
        set(range(64, 129)) | set(range(0, 32, 2)),  # Mixed
    ]
    
    best_A = None
    best_score = float('inf')
    
    for init_set in initial_sets:
        M_init = calc_mult(init_set)
        if np.min(M_init) < 1:
            continue  # Skip invalid initial sets
        
        A = set(init_set)
        M = M_init
        current_score = int(np.max(M))
        
        if current_score < best_score:
            best_score = current_score
            best_A = set(A)
    
    if best_A is None:
        # Fallback to original
        A = set(range(64, 129))
        M = calc_mult(A)
        best_A = set(A)
        best_score = int(np.max(M))
    
    # Simulated Annealing with improved moves
    T = 5.0
    cooling = 0.9995
    ALL_ELEMENTS = list(range(0, 129))
    
    # Precompute for each element x, which sums it contributes to
    # This helps with faster updates
    
    while time.time() < deadline:
        # Choose a move type with bias toward removal/swap
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove an element
            if len(A) <= 10:  # Don't make set too small
                T *= cooling
                continue
            
            # Prefer removing elements that contribute to high-multiplicity sums
            A_list = list(A)
            x = random.choice(A_list)
            
            new_A, new_M = remove_from_A(x, A, M)
            if new_M is not None:
                score_new = int(np.max(new_M))
                delta = score_new - current_score
                if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                    A = new_A
                    M = new_M
                    current_score = score_new
                    if score_new < best_score:
                        best_score = score_new
                        best_A = set(A)
        
        elif move_type < 0.8:
            # Swap: remove x, add y
            if len(A) == 0:
                T *= cooling
                continue
            
            x = random.choice(list(A))
            # Choose y not in A
            candidates = [y for y in ALL_ELEMENTS if y not in A]
            if not candidates:
                T *= cooling
                continue
            y = random.choice(candidates)
            
            # Simulate: remove x then add y
            temp_A, temp_M = remove_from_A(x, A, M)
            if temp_M is not None:
                new_A, new_M = add_to_A(y, temp_A, temp_M)
                score_new = int(np.max(new_M))
                delta = score_new - current_score
                if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                    A = new_A
                    M = new_M
                    current_score = score_new
                    if score_new < best_score:
                        best_score = score_new
                        best_A = set(A)
        
        else:
            # Add an element (less common)
            candidates = [y for y in ALL_ELEMENTS if y not in A]
            if not candidates:
                T *= cooling
                continue
            y = random.choice(candidates)
            
            new_A, new_M = add_to_A(y, A, M)
            score_new = int(np.max(new_M))
            delta = score_new - current_score
            # Adding usually increases max, so only accept with low probability
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 0.01)) * 0.5:
                A = new_A
                M = new_M
                current_score = score_new
        
        # Cool down
        T *= cooling
        if T < 0.05:
            T = 0.05
    
    # Final polish: try to remove more elements
    removed = True
    while removed and time.time() < deadline:
        removed = False
        A_list = list(A)
        random.shuffle(A_list)
        for x in A_list:
            if x not in A:
                continue
            new_A, new_M = remove_from_A(x, A, M)
            if new_M is not None:
                score_new = int(np.max(new_M))
                if score_new <= best_score:
                    A = new_A
                    M = new_M
                    current_score = score_new
                    if score_new < best_score:
                        best_score = score_new
                        best_A = set(A)
                    removed = True
    
    return list(best_A)

# EVOLVE-BLOCK-START
# The code above is the implementation. 
# Note: The task asks to keep the function names and signatures.
# I will place the code in the EVOLVE-BLOCK.
# EVOLVE-BLOCK-END
