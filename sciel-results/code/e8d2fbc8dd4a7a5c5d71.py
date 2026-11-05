import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    def pairwise_coprime(vals):
        distinct = list(set(vals))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def score_pair(A, B):
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if not pairwise_coprime(distinct):
            return 0
        return len(distinct)
    
    # Initial best from seed
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_pair(best_A, best_B)
    
    # Current state for search
    cur_A = best_A[:]
    cur_B = best_B[:]
    cur_score = best_score
    
    # Local search + Random restarts
    # We will try to improve the score by modifying elements
    
    def random_valid_set():
        return sorted(random.sample(range(1, 129), 3))
    
    # Heuristic: Try to maximize distinct sums first, then check coprimality
    # But score is 0 if not coprime, so we need to jump to valid regions.
    
    # Strategy:
    # 1. Explore neighborhood of current best.
    # 2. If stuck, random restart.
    
    while time.time() < deadline:
        # Local search step
        improved = False
        for _ in range(100): # 100 local moves per iteration
            if time.time() > deadline:
                break
            
            # Choose which set to modify
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old_val = cur_A[idx]
                # Try a random new value
                new_val = random.randint(1, 128)
                if new_val in cur_A:
                    continue
                cur_A[idx] = new_val
                cur_A.sort()
                new_score = score_pair(cur_A, cur_B)
                if new_score > cur_score:
                    cur_score = new_score
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_A = cur_A[:]
                        best_B = cur_B[:]
                else:
                    # Revert if not improved (greedy hill climbing)
                    # Or accept with probability for simulated annealing? 
                    # Let's stick to greedy + random restarts for simplicity and speed
                    cur_A[idx] = old_val
                    cur_A.sort()
            else:
                idx = random.randint(0, 2)
                old_val = cur_B[idx]
                new_val = random.randint(1, 128)
                if new_val in cur_B:
                    continue
                cur_B[idx] = new_val
                cur_B.sort()
                new_score = score_pair(cur_A, cur_B)
                if new_score > cur_score:
                    cur_score = new_score
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_A = cur_A[:]
                        best_B = cur_B[:]
                else:
                    cur_B[idx] = old_val
                    cur_B.sort()
        
        # If we improved, we stay in this neighborhood.
        # If we didn't improve in the last 100 steps, we might be stuck.
        # Let's do a random restart periodically or if stuck.
        if not improved:
            # Random restart
            cur_A = random_valid_set()
            cur_B = random_valid_set()
            cur_score = score_pair(cur_A, cur_B)
            if cur_score > best_score:
                best_score = cur_score
                best_A = cur_A[:]
                best_B = cur_B[:]
    
    return [best_A, best_B]
