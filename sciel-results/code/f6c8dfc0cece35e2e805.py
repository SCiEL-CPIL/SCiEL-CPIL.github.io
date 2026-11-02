import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin before 150s kill
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters
        # Simple greedy or exhaustive for small n
        all_words = list(product(range(q), repeat=n))
        # For small n, just return all words or a simple grid
        return [list(w) for w in all_words]
    
    # Specific solution for K_3(6,1)
    # Space size 729
    all_words = np.array(list(product(range(3), repeat=6)), dtype=np.int8)
    N = 729
    
    # Precompute balls
    # balls[i] is a list of indices of words covered by word i
    balls = []
    for i in range(N):
        w = all_words[i]
        ball = [i]
        for pos in range(6):
            for val in range(3):
                if val != w[pos]:
                    new_w = w.copy()
                    new_w[pos] = val
                    # Find index of new_w
                    # Since we have all_words, we can map it
                    # But doing this in a loop is slow. 
                    # Better to precompute a mapping from tuple to index.
                    pass
        
    # Precompute mapping
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    balls = []
    for i in range(N):
        w = all_words[i]
        ball_set = {i}
        for pos in range(6):
            orig_val = w[pos]
            for val in range(3):
                if val == orig_val:
                    continue
                new_w = w.copy()
                new_w[pos] = val
                ball_set.add(word_to_idx[tuple(new_w)])
        balls.append(list(ball_set))
    
    # Initial Code: 81 words from perfect code of length 4
    # Perfect code of length 4:
    # We need 9 words such that balls of radius 1 partition the 81 words of length 4.
    # One such code is the tetracode? No, tetracode is length 9.
    # The ternary Hamming code of length 4?
    # Actually, $K_3(4,1)=9$.
    # We can construct it.
    # Words:
    # 0000
    # 0011, 0022? No.
    # Let's just use a known set or generate one.
    # A simple way:
    # The code consists of words (a, b, c, d) such that a + b + c + d = 0 mod 3?
    # No, that's a linear code.
    # Let's check if the set {(a,b,c,d) | a+b+c+d=0} works.
    # Size 27. Too big.
    # We need size 9.
    # The code is:
    # 0000
    # 0111, 0122?
    # Actually, we can just use the first 9 words of a greedy cover for n=4?
    # Or just use the 81-word direct product as the start for SA, 
    # but initialize with the 81 words explicitly.
    
    # Let's generate the 9 words for n=4.
    # We can find them by greedy set cover on n=4.
    # Space 81. Ball size 13 (1 + 4*2).
    # 81/13 ~ 6.2. So 9 is plausible.
    
    # For simplicity and robustness, let's start with the 81 words 
    # formed by taking 9 "heads" and all 9 "tails".
    # We need 9 heads that cover all 81 heads.
    # Let's find them quickly.
    
    heads_space = list(product(range(3), repeat=4))
    head_to_idx = {h: i for i, h in enumerate(heads_space)}
    head_balls = []
    for h in heads_space:
        ball = {h}
        for pos in range(4):
            for val in range(3):
                if val != h[pos]:
                    nh = list(h)
                    nh[pos] = val
                    ball.add(tuple(nh))
        head_balls.append(ball)
        
    # Greedy for heads
    uncovered_heads = set(range(81))
    heads = []
    while uncovered_heads:
        best_h = None
        best_cov = -1
        for h in range(81):
            cov = len(head_balls[h] & {head_to_idx[hh] for hh in uncovered_heads})
            # Optimization: just count intersection
            # This is small (81), so it's fine
            if cov > best_cov:
                best_cov = cov
                best_h = h
        if best_h is None:
            break
        heads.append(heads_space[best_h])
        uncovered_heads -= head_balls[best_h]
        
    # Now construct the 81 codewords for n=6
    initial_code_indices = []
    tails = list(product(range(3), repeat=2))
    for h in heads:
        for t in tails:
            cw = list(h) + list(t)
            initial_code_indices.append(word_to_idx[tuple(cw)])
            
    # State for SA
    # code_set: set of indices
    # cover_count: np.array of size 729, counts how many codewords cover each word
    # uncovered_count: int
    
    def init_state(indices):
        cover_count = np.zeros(N, dtype=np.int32)
        for idx in indices:
            for j in balls[idx]:
                cover_count[j] += 1
        uncovered_count = np.sum(cover_count == 0)
        return set(indices), cover_count, uncovered_count
        
    code_set, cover_count, uncovered_count = init_state(initial_code_indices)
    best_code = list(code_set)
    best_size = len(best_code)
    best_uncovered = uncovered_count
    
    # SA Parameters
    T = 100.0
    T_min = 0.1
    alpha = 0.999
    cost_weight = 1000.0
    
    current_cost = len(code_set) * cost_weight + uncovered_count
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 2:
        # Propose move
        move_type = random.choice(['remove', 'add', 'swap'])
        
        delta_cost = 0
        accepted = False
        new_code_set = code_set
        new_cover_count = cover_count
        new_uncovered = uncovered_count
        
        if move_type == 'remove':
            if not code_set: continue
            c = random.choice(list(code_set))
            # Calculate delta
            delta_uncovered = 0
            for j in balls[c]:
                if cover_count[j] == 1:
                    delta_uncovered += 1
                # else cover_count[j] > 1, remains covered
            delta_cost = -cost_weight + delta_uncovered
            
            if delta_cost < 0 or (delta_cost <= 50 and random.random() < np.exp(-delta_cost / T)):
                # Accept
                code_set.remove(c)
                for j in balls[c]:
                    cover_count[j] -= 1
                    if cover_count[j] == 0:
                        uncovered_count += 1
                current_cost += delta_cost
                accepted = True
                
        elif move_type == 'add':
            # Pick a word not in code_set
            # To be efficient, pick a word that covers many uncovered
            # Or just random
            # Random is fine for SA
            c = random.randint(0, N-1)
            if c in code_set:
                continue
            
            delta_uncovered = 0
            for j in balls[c]:
                if cover_count[j] == 0:
                    delta_uncovered -= 1
            delta_cost = cost_weight + delta_uncovered
            
            if delta_cost < 0 or (delta_cost <= 50 and random.random() < np.exp(-delta_cost / T)):
                # Accept
                code_set.add(c)
                for j in balls[c]:
                    if cover_count[j] == 0:
                        uncovered_count -= 1
                    cover_count[j] += 1
                current_cost += delta_cost
                accepted = True
                
        else: # swap
            if not code_set: continue
            c1 = random.choice(list(code_set))
            c2 = random.randint(0, N-1)
            if c1 == c2:
                continue
            if c2 in code_set:
                continue
                
            # Remove c1, Add c2
            delta_uncovered = 0
            # Words covered by c1 but not c2: might become uncovered
            # Words covered by c2 but not c1: might become covered
            # Words covered by both: no change in uncovered status (count changes but >0)
            # Words covered by neither: no change
            
            # We need to check intersection with current cover counts.
            # This is complex to do analytically without iterating.
            # Let's just iterate balls[c1] and balls[c2].
            
            # Simulate
            # We can't easily undo, so let's calculate delta carefully.
            # For each j in balls[c1]:
            #   if cover_count[j] == 1 and j not in balls[c2]: delta_uncovered += 1
            # For each j in balls[c2]:
            #   if cover_count[j] == 0 and j not in balls[c1]: delta_uncovered -= 1
            
            ball_c1 = set(balls[c1])
            ball_c2 = set(balls[c2])
            
            for j in ball_c1:
                if cover_count[j] == 1 and j not in ball_c2:
                    delta_uncovered += 1
            
            for j in ball_c2:
                if cover_count[j] == 0 and j not in ball_c1:
                    delta_uncovered -= 1
                    
            delta_cost = delta_uncovered # Size doesn't change
            
            if delta_cost < 0 or (delta_cost <= 50 and random.random() < np.exp(-delta_cost / T)):
                # Accept
                code_set.remove(c1)
                code_set.add(c2)
                
                for j in ball_c1:
                    cover_count[j] -= 1
                    if cover_count[j] == 0:
                        uncovered_count += 1
                        
                for j in ball_c2:
                    if cover_count[j] == 0:
                        uncovered_count -= 1
                    cover_count[j] += 1
                    
                current_cost += delta_cost
                accepted = True

        if accepted:
            if uncovered_count == 0 and len(code_set) < best_size:
                best_size = len(code_set)
                best_code = list(code_set)
                best_uncovered = 0
                no_improve_steps = 0
            else:
                no_improve_steps += 1
                
            T *= alpha
            if T < T_min:
                T = T_min
                
        else:
            no_improve_steps += 1
            
        steps += 1
        
        # Periodic peeling if valid
        if uncovered_count == 0 and steps % 1000 == 0:
            # Try to remove redundant words
            changed = True
            while changed and time.time() < deadline - 2:
                changed = False
                to_remove = []
                for c in list(code_set):
                    # Check if c is redundant
                    # i.e., all j in balls[c] are covered by others
                    redundant = True
                    for j in balls[c]:
                        if cover_count[j] == 1:
                            redundant = False
                            break
                    if redundant:
                        to_remove.append(c)
                if to_remove:
                    for c in to_remove:
                        code_set.remove(c)
                        for j in balls[c]:
                            cover_count[j] -= 1
                            if cover_count[j] == 0:
                                uncovered_count += 1
                        current_cost -= cost_weight
                    changed = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
                        best_uncovered = 0
                        no_improve_steps = 0

    # Final peeling
    if uncovered_count == 0:
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            to_remove = []
            for c in list(code_set):
                redundant = True
                for j in balls[c]:
                    if cover_count[j] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(c)
            if to_remove:
                for c in to_remove:
                    code_set.remove(c)
                    for j in balls[c]:
                        cover_count[j] -= 1
                        if cover_count[j] == 0:
                            uncovered_count += 1
                    current_cost -= cost_weight
                changed = True
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
                    best_uncovered = 0

    # If SA failed to find a valid code (uncovered > 0), use best valid found
    # If no valid code found, fallback to initial 81 words
    if best_uncovered > 0:
        # Reconstruct initial
        result_indices = initial_code_indices
    else:
        result_indices = best_code

    result = [list(all_words[i]) for i in result_indices]
    
    # Ensure distinctness
    seen = set()
    unique_result = []
    for cw in result:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_result.append(cw)
            
    return unique_result
