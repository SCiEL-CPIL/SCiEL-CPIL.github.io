# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    def compute_score_exact(U):
        """Compute score using Python sets for exactness."""
        U = sorted(set(U))
        if not U or U[0] != 0:
            return -1
        if len(U) == 1:
            return 0.0
        max_u = U[-1]
        if max_u == 0:
            return 0.0
        
        sums = set()
        diffs = set()
        for a in U:
            for b in U:
                sums.add(a + b)
                diffs.add(a - b)
        
        n_sum = len(sums)
        n_diff = len(diffs)
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    best_score = 0
    best_set = [0, 1, 3]
    best_score = compute_score_exact([0, 1, 3])
    
    def greedy_construction(max_steps=100, max_candidate_range=50):
        """
        Greedily add elements to maximize score.
        """
        U = [0]
        sums = set([0])
        diffs = set([0])
        current_max = 0
        
        for _ in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            best_x = -1
            best_new_score = -1
            
            # Generate candidates
            candidates = set()
            # Small candidates
            for x in range(1, current_max + 10):
                candidates.add(x)
            # Medium candidates
            for x in range(current_max + 1, current_max + max_candidate_range):
                candidates.add(x)
            # Random large candidates to escape local optima
            for _ in range(5):
                candidates.add(random.randint(current_max + 1, current_max * 2 + 10))
            
            for x in candidates:
                if x <= 0:
                    continue
                
                # Estimate impact
                new_diffs_count = 0
                new_sums_count = 0
                
                for u in U:
                    d1 = x - u
                    d2 = u - x
                    s1 = x + u
                    
                    if d1 not in diffs:
                        new_diffs_count += 1
                    if d2 not in diffs:
                        new_diffs_count += 1
                    if s1 not in sums:
                        new_sums_count += 1
                
                # Calculate new score
                new_max = max(current_max, x)
                new_n_diff = len(diffs) + new_diffs_count
                new_n_sum = len(sums) + new_sums_count
                
                if new_n_sum == 0:
                    continue
                    
                base = 2 * new_max + 1
                ratio = new_n_diff / new_n_sum
                if ratio <= 1:
                    continue
                
                score = 1 + math.log(ratio) / math.log(base)
                
                if score > best_new_score:
                    best_new_score = score
                    best_x = x
            
            if best_x == -1:
                break
            
            # Update U, sums, diffs
            U.append(best_x)
            U.sort()
            for u in U:
                s = best_x + u
                if s not in sums:
                    sums.add(s)
                
                d1 = best_x - u
                if d1 not in diffs:
                    diffs.add(d1)
                
                d2 = u - best_x
                if d2 not in diffs:
                    diffs.add(d2)
            
            current_max = max(current_max, best_x)
            
            # Check if we improved the global best
            if best_new_score > best_score:
                best_score = best_new_score
                best_set = list(U)
        
        return best_set

    # Run multiple greedy constructions with different random seeds
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        random.seed()
        U_candidate = greedy_construction(max_steps=50, max_candidate_range=20)
        s = compute_score_exact(U_candidate)
        if s > best_score:
            best_score = s
            best_set = U_candidate

    # Final local search on the best set found
    if time.time() < deadline - 5:
        current = list(best_set)
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < deadline - 5 and no_improve < max_no_improve:
            improved = False
            
            # Try adding an element
            if len(current) < 100:
                max_cur = max(current)
                candidates = set()
                for x in current:
                    for dx in [-3, -2, -1, 1, 2, 3]:
                        candidates.add(x + dx)
                candidates.update(range(max_cur + 1, max_cur + 10))
                candidates.discard(0)
                
                for new_el in sorted(candidates):
                    if new_el in current or new_el < 0:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score_exact(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            # Try replacing an element
            if len(current) >= 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    for new_el in range(max(1, current[i] - 5), current[i] + 6):
                        if new_el in current or new_el < 0:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score_exact(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0

    return list(best_set)
# EVOLVE-BLOCK-END
