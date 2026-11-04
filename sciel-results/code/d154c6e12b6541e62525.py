"""
Combined approach: Dynamic candidate generation + local search with geometric seeds
Focus on maximizing |U-U|/|U+U| ratio while keeping max(U) bounded.
"""
import time
import random
from math import log
from itertools import combinations

def score_set(U):
    """Compute score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)"""
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return -1.0
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1.0
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1.0
        
    max_u = U[-1]
    D = 2 * max_u + 1
    if D <= 1:
        return -1.0
        
    return 1 + log(ratio) / log(D)

def generate_structural_candidates():
    """Generate structured candidate sets with good difference/sum properties."""
    candidates = []
    
    # Powers of 2 (classic good structure)
    for n in range(3, 12):
        U = [2**i for i in range(n)]
        candidates.append(U)
    
    # Modified powers of 2 with small perturbations
    base = [2**i for i in range(6)]
    for n in range(3, len(base)+1):
        sub = base[:n]
        candidates.append(sub[:])
        # Try replacing last element with nearby values
        last = sub[-1]
        for delta in [-2, -1, 1, 2, 3, 4, 5, 6, 8, 10]:
            new_last = last + delta
            if new_last > 0 and new_last not in sub[:-1]:
                candidates.append(sub[:-1] + [new_last])
    
    # Geometric with base 3
    for n in range(3, 8):
        U = [3**i for i in range(n)]
        candidates.append(U)
    
    # Squares
    for n in range(3, 10):
        U = [i*i for i in range(n)]
        candidates.append(U)
    
    # Triangular numbers
    for n in range(3, 8):
        U = [i*(i+1)//2 for i in range(n)]
        candidates.append(U)
    
    # Known good small sets
    known_good = [
        [0, 1, 3, 7],
        [0, 1, 3, 8],
        [0, 1, 4, 10],
        [0, 1, 4, 9],
        [0, 1, 5, 10],
        [0, 1, 2, 5],
        [0, 1, 2, 6],
        [0, 1, 3, 6],
        [0, 1, 4, 7],
        [0, 1, 5, 9],
        [0, 1, 2, 7],
        [0, 1, 3, 9],
        [0, 1, 4, 11],
        [0, 1, 5, 12],
        [0, 1, 2, 8],
        [0, 1, 3, 10],
        [0, 1, 4, 12],
        [0, 1, 5, 13],
        [0, 1, 2, 9],
        [0, 1, 3, 11],
        [0, 1, 4, 13],
        [0, 1, 5, 14],
        [0, 1, 2, 10],
        [0, 1, 3, 12],
        [0, 1, 4, 14],
        [0, 1, 5, 15],
        [0, 1, 2, 11],
        [0, 1, 3, 13],
        [0, 1, 4, 15],
        [0, 1, 5, 16],
        [0, 1, 4, 9, 16],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 8, 17],
        [0, 1, 4, 10, 21],
        [0, 1, 5, 11, 23],
        [0, 2, 5, 11, 23],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 4, 9, 16, 25],
    ]
    candidates.extend(known_good)
    
    # Random small sets for diversity
    random.seed(12345)
    for _ in range(50):
        n = random.randint(3, 8)
        max_val = random.randint(15, 60)
        other_elements = random.sample(range(1, max_val), n - 1)
        cand = sorted([0] + other_elements)
        candidates.append(cand)
    
    return candidates

def generate_move_candidates(U, max_val_limit=200):
    """Generate candidate elements for moves based on sums and differences."""
    U_set = set(U)
    candidates = set()
    
    for a in U:
        for b in U:
            s = a + b
            d = abs(a - b)
            if 0 < s <= max_val_limit and s not in U_set:
                candidates.add(s)
            if 0 < d <= max_val_limit and d not in U_set:
                candidates.add(d)
            # Also try a-b and b-a (already covered by abs, but ensure non-neg)
    
    # Also add some random values near existing elements
    for a in U:
        if a == 0:
            continue
        for delta in [-3, -2, -1, 1, 2, 3, 4, 5, 7, 10]:
            nv = a + delta
            if 0 < nv <= max_val_limit and nv not in U_set:
                candidates.add(nv)
    
    return list(candidates)

def local_search(U, deadline, max_val_limit=200, max_n=20):
    """
    Local search with dynamic candidate generation.
    Moves: add, remove, replace.
    """
    U = sorted(set(U))
    current_score = score_set(U)
    if current_score < 0:
        return U, -1.0
    
    best_score = current_score
    best_set = U[:]
    
    candidates = generate_move_candidates(U, max_val_limit)
    
    no_improvement = 0
    max_no_improvement = 40
    
    while time.time() < deadline - 3:
        if no_improvement > max_no_improvement:
            break
            
        improved = False
        move_type = random.random()
        
        # Refresh candidates periodically
        if not candidates or random.random() < 0.05:
            current_max = max(U)
            new_limit = max(max_val_limit, int(current_max * 1.5) + 20)
            candidates = generate_move_candidates(U, new_limit)
        
        if move_type < 0.45:
            # ADD an element
            if len(U) < max_n and candidates:
                random.shuffle(candidates)
                for c in candidates[:15]:
                    if c in U:
                        continue
                    new_set = sorted(U + [c])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        U = new_set
                        current_score = new_score
                        improved = True
                        candidates = generate_move_candidates(U, max(max_val_limit, max(U)*2+10))
                        if current_score > best_score:
                            best_score = current_score
                            best_set = U[:]
                        break
        elif move_type < 0.65:
            # REMOVE an element
            if len(U) > 3:
                removable = [x for x in U if x != 0]
                if removable:
                    random.shuffle(removable)
                    for c in removable[:5]:
                        new_set = [x for x in U if x != c]
                        new_score = score_set(new_set)
                        if new_score > current_score:
                            U = new_set
                            current_score = new_score
                            improved = True
                            candidates = generate_move_candidates(U, max_val_limit)
                            if current_score > best_score:
                                best_score = current_score
                                best_set = U[:]
                            break
        else:
            # REPLACE an element
            if len(U) > 2 and candidates:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                random.shuffle(candidates)
                for new_val in candidates[:15]:
                    if new_val == old_val or new_val in U:
                        continue
                    new_set = U.copy()
                    new_set[idx] = new_val
                    new_set = sorted(new_set)
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        U = new_set
                        current_score = new_score
                        improved = True
                        candidates = generate_move_candidates(U, max(max_val_limit, max(U)*2+10))
                        if current_score > best_score:
                            best_score = current_score
                            best_set = U[:]
                        break
        
        if improved:
            no_improvement = 0
        else:
            no_improvement += 1
            
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # 1. Generate and score structural candidates
    candidates = generate_structural_candidates()
    
    best_score = -1
    best_set = [0, 1, 3, 7]  # Fallback
    
    scored_candidates = []
    for c in candidates:
        if time.time() > deadline - 10:
            break
        s = score_set(c)
        if s > -0.5:  # Only consider valid scores
            scored_candidates.append((s, c))
            if s > best_score:
                best_score = s
                best_set = c[:]
    
    # Sort by score descending
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Local search from top candidates
    num_restarts = min(5, len(scored_candidates))
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        s, c = scored_candidates[i]
        if s <= best_score and i > 0:
            continue  # Skip if not competitive
        
        refined_set, refined_score = local_search(c, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined_set
    
    # 3. If time remains, try a few random restarts
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 15:
        random.seed(99999)
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            n = random.randint(4, 10)
            max_val = random.randint(20, 80)
            other_elements = random.sample(range(1, max_val), n - 1)
            cand = sorted([0] + other_elements)
            s = score_set(cand)
            if s > -0.5:
                refined_set, refined_score = local_search(cand, deadline)
                if refined_score > best_score:
                    best_score = refined_score
                    best_set = refined_set
    
    return best_set
