"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def compute_score_fast(U):
    """
    Fast computation of the score using NumPy.
    Score = 1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)
    """
    if len(U) < 2:
        return 0.0
    U = np.array(sorted(set(U)), dtype=np.int64)
    maxU = U[-1]
    if maxU == 0:
        return 0.0
    
    # Compute sum and difference sets
    # Using broadcasting for N up to ~100 is fast enough if done carefully
    # sums = U[:, None] + U[None, :]
    # diffs = U[:, None] - U[None, :]
    
    # For very large N, this creates N^2 arrays. 
    # If N=100, 10,000 elements. Very fast.
    # If N=1000, 1,000,000 elements. Still okay for one eval, but slow for many.
    # We will limit N to 100 in the search.
    
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    size_sum = len(np.unique(sums))
    size_diff = len(np.unique(diffs))
    
    if size_sum == 0:
        return 0.0
        
    log_ratio = math.log(size_diff / size_sum)
    log_range = math.log(2 * maxU + 1)
    
    return 1.0 + log_ratio / log_range

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Initial candidates: 
    # 1. Dense: {0, 1, ..., N-1}
    # 2. Sparse: Powers of 2
    # 3. Semi-sparse: Mix
    
    candidates = []
    # Try various sizes
    for n in range(3, 20):
        # Dense
        candidates.append(list(range(n)))
        # Powers of 2
        candidates.append([2**i for i in range(n)])
        # Fibonacci
        fibs = [0, 1]
        while len(fibs) < n:
            fibs.append(fibs[-1] + fibs[-2])
        candidates.append(fibs)
        
    best_U = None
    best_score = -1.0
    
    for c in candidates:
        s = compute_score_fast(c)
        if s > best_score:
            best_score = s
            best_U = c
            
    current_U = list(best_U)
    current_score = best_score
    
    # Local Search with Fast Evaluation
    # We will perform moves:
    # 1. Add a new element x
    # 2. Remove an element x
    # 3. Replace x with y
    
    # To speed up, we will batch evaluate some candidates or use a faster heuristic for pruning.
    # However, compute_score_fast is already quite fast for N<100.
    
    stuck_count = 0
    
    while time.time() < deadline:
        improved = False
        current_max = max(current_U) if current_U else 0
        
        # Generate candidate moves
        moves = []
        
        # 1. Additions
        # Try small increments near existing elements
        # Try geometric growth
        for x in current_U:
            for delta in [1, 2, 3, 5, 10]:
                nx = x + delta
                if nx > 0 and nx not in current_U:
                    moves.append(('add', nx))
        
        # Try large jumps (geometric)
        if current_max > 0:
            for mult in [2, 3]:
                nx = current_max * mult
                if nx not in current_U and nx < 100000: # Cap to keep log denominator manageable
                    moves.append(('add', nx))
                    
        # 2. Removals
        for x in current_U:
            if x != 0:
                moves.append(('remove', x))
                
        # 3. Replacements
        # Try replacing x with x +/- delta
        for x in current_U:
            if x == 0: continue
            for delta in [-2, -1, 1, 2, 5]:
                y = x + delta
                if y > 0 and y not in current_U:
                    moves.append(('replace', (x, y)))
        
        # Shuffle moves to avoid bias
        random.shuffle(moves)
        
        best_move_score = current_score
        best_move = None
        
        # Evaluate moves
        # Optimization: If we have too many moves, sample a subset
        if len(moves) > 50:
            moves = random.sample(moves, 50)
            
        for move in moves:
            if time.time() > deadline:
                break
                
            new_U = list(current_U)
            if move[0] == 'add':
                new_U.append(move[1])
            elif move[0] == 'remove':
                new_U.remove(move[1])
            elif move[0] == 'replace':
                x, y = move[1]
                new_U.remove(x)
                new_U.append(y)
                
            s = compute_score_fast(new_U)
            if s > best_move_score:
                best_move_score = s
                best_move = new_U
                
        if best_move:
            current_U = best_move
            current_score = best_move_score
            if current_score > best_score:
                best_score = current_score
                best_U = list(current_U)
            stuck_count = 0
            improved = True
        else:
            stuck_count += 1
            
            # If stuck, try a random restart or larger perturbation
            if stuck_count > 5:
                # Random perturbation
                if random.random() < 0.5:
                    # Add a random large number
                    nx = random.randint(10, 1000)
                    if nx not in current_U:
                        new_U = current_U + [nx]
                        s = compute_score_fast(new_U)
                        if s > current_score:
                            current_U = new_U
                            current_score = s
                            stuck_count = 0
                else:
                    # Remove a random element
                    if len(current_U) > 3:
                        idx = random.randint(1, len(current_U)-1) # Don't remove 0
                        new_U = current_U[:idx] + current_U[idx+1:]
                        s = compute_score_fast(new_U)
                        if s > current_score:
                            current_U = new_U
                            current_score = s
                            stuck_count = 0
                stuck_count = 0 # Reset to allow more attempts
                
    return best_U

# EVOLVE-BLOCK-END
