import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (evaluator kills at 150s, leave buffer)
    
    def time_left():
        return deadline - time.time()
    
    def compute_score(U):
        """Direct computation for small sets. Fast and accurate."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        # Direct set construction
        # Optimization: Use set comprehensions or loops
        sumset = set()
        diffset = set()
        
        # For very small n, nested loops are fine.
        # n is typically < 30 in this regime.
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    best_score = 0
    best_set = [0, 1, 3]
    
    # Evaluate seed
    seed_score = compute_score(best_set)
    if seed_score > best_score:
        best_score = seed_score

    # Generate initial diverse candidates
    candidates = []
    
    # 1. Geometric progressions
    for B in range(2, 20):
        for k in range(2, 12):
            U = [0]
            val = 1
            for _ in range(k-1):
                val *= B
                U.append(val)
            if max(U) < 5000: # Keep max U reasonable to keep log base small
                candidates.append(U)
                
    # 2. Random sparse sets
    for _ in range(50):
        n = random.randint(3, 15)
        max_val = random.randint(5, 500)
        U = {0}
        for _ in range(n-1):
            U.add(random.randint(1, max_val))
        candidates.append(sorted(U))
        
    # 3. Specific known good structures (e.g. from literature or previous runs)
    candidates.append([0, 1, 3])
    candidates.append([0, 1, 2, 5])
    candidates.append([0, 1, 3, 7])
    
    # Evaluate all candidates
    scored_candidates = []
    for cand in candidates:
        if time_left() < 5:
            break
        s = compute_score(cand)
        scored_candidates.append((s, cand))
        if s > best_score:
            best_score = s
            best_set = list(cand)
            
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Local Search
    # Refine top candidates
    num_to_refine = min(20, len(scored_candidates))
    
    for idx in range(num_to_refine):
        if time_left() < 10:
            break
            
        current_set = list(scored_candidates[idx][1])
        current_score = scored_candidates[idx][0]
        
        no_improve = 0
        max_no_improve = 100
        
        while no_improve < max_no_improve and time_left() > 5:
            improved = False
            n = len(current_set)
            max_val = max(current_set)
            
            # Generate moves
            moves = []
            
            # Add elements
            if n < 25:
                # Try values near existing ones
                for x in current_set:
                    for d in range(1, 6):
                        for sign in [1, -1]:
                            nv = x + sign * d
                            if nv > 0 and nv not in current_set:
                                moves.append(('add', nv))
                # Try random values
                for _ in range(20):
                    # Bias towards smaller values to keep max U small
                    if random.random() < 0.5:
                        nv = random.randint(1, max_val)
                    else:
                        nv = random.randint(1, max_val * 2)
                    if nv not in current_set:
                        moves.append(('add', nv))
                        
            # Remove elements
            if n > 3:
                for i in range(n):
                    if current_set[i] != 0:
                        moves.append(('remove', i))
                        
            # Replace elements
            if n > 3:
                for i in range(n):
                    if current_set[i] == 0:
                        continue
                    # Small perturbations
                    for d in range(-3, 4):
                        if d == 0:
                            continue
                        nv = current_set[i] + d
                        if nv > 0 and nv not in current_set:
                            moves.append(('replace', i, nv))
                    # Random replacement
                    if random.random() < 0.1:
                        nv = random.randint(1, max_val * 3)
                        if nv not in current_set:
                            moves.append(('replace', i, nv))
            
            if not moves:
                no_improve += 1
                continue
                
            random.shuffle(moves)
            
            # Try a subset of moves
            for move in moves[:50]:
                if time_left() < 5:
                    break
                    
                new_set = list(current_set)
                
                if move[0] == 'add':
                    new_set.append(move[1])
                elif move[0] == 'remove':
                    new_set.pop(move[1])
                elif move[0] == 'replace':
                    new_set[move[1]] = move[2]
                    
                # Ensure validity
                new_set = sorted(set(new_set))
                if 0 not in new_set:
                    continue
                if len(new_set) != len(set(new_set)):
                    continue
                    
                new_score = compute_score(new_set)
                
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_set)
                    improved = True
                    no_improve = 0
                    break # Restart search with new state
                    
            if not improved:
                no_improve += 1
                
    return list(best_set)
