import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _score_with_diffs(s):
    """
    Compute the score (number of distinct d > 0 with 3-AP) and the set of such differences.
    s: set of integers
    """
    s_list = sorted(s)
    s_set = set(s)
    diffs = set()
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            d = b - a
            c = b + d
            if c in s_set:
                diffs.add(d)
    return len(diffs), diffs

def _generate_initial_set():
    """
    Generate a diverse initial set.
    Strategy: Mix of intervals, arithmetic progressions, and random.
    """
    strategies = []
    
    # 1. Simple Interval
    strategies.append(set(range(1, 33)))
    
    # 2. Random
    strategies.append(set(random.sample(range(1, 513), 32)))
    
    # 3. Two Blocks
    s = set()
    b1_start = random.randint(1, 100)
    b2_start = random.randint(200, 400)
    for i in range(16):
        s.add(b1_start + i)
        s.add(b2_start + i)
    if len(s) == 32:
        strategies.append(s)
        
    # 4. Arithmetic Progression with small step
    d = random.choice([2, 3, 4])
    a = random.randint(1, 512 - 31 * d)
    s = set(a + i * d for i in range(32))
    if all(1 <= x <= 512 for x in s):
        strategies.append(s)
        
    # 5. "Sparse" Random - pick 32 from 1..512
    strategies.append(set(random.sample(range(1, 513), 32)))
    
    # Pick the best one
    best_s = None
    best_score = -1
    for s in strategies:
        if len(s) != 32: continue
        sc, _ = _score_with_diffs(s)
        if sc > best_score:
            best_score = sc
            best_s = s
            
    return best_s if best_s else set(random.sample(range(1, 513), 32))

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    current_set = _generate_initial_set()
    current_score, current_diffs = _score_with_diffs(current_set)
    
    best_set = set(current_set)
    best_score = current_score
    
    # Hill Climbing with First Improvement
    no_improve_count = 0
    max_no_improve = 100 # After 100 failed attempts to find any improvement, kick
    
    while time.time() < deadline:
        improved = False
        s_list = sorted(current_set)
        s_set = set(current_set)
        candidates_y = [y for y in range(1, 513) if y not in s_set]
        
        # Shuffle candidates to avoid bias towards low numbers
        random.shuffle(candidates_y)
        
        for x in s_list:
            new_s_set = set(s_set)
            new_s_set.remove(x)
            
            for y in candidates_y:
                new_s_set.add(y)
                
                # Score new_s_set
                sc, _ = _score_with_diffs(new_s_set)
                
                if sc > current_score:
                    # Apply improvement
                    current_set = new_s_set
                    current_score = sc
                    current_diffs, _ = _score_with_diffs(current_set)
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
                    
                    improved = True
                    break
            
            if improved:
                break
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                # Kick: Randomly change 2 elements
                for _ in range(2):
                    x = random.choice(list(current_set))
                    y = random.randint(1, 512)
                    if y not in current_set:
                        current_set.remove(x)
                        current_set.add(y)
                
                current_score, current_diffs = _score_with_diffs(current_set)
                no_improve_count = 0
                
    return list(best_set)

# EVOLVE-BLOCK-END
