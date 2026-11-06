"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best_set = None
    best_score = -1
    
    def score_set(s):
        """Count the number of sum collisions in the set.
        Higher collisions = harder to find large Sidon subset."""
        sums = {}
        for i in range(len(s)):
            for j in range(i, len(s)):
                s_val = s[i] + s[j]
                if s_val in sums:
                    sums[s_val] += 1
                else:
                    sums[s_val] = 1
        # Count total collisions (pairs sharing a sum)
        collisions = 0
        for cnt in sums.values():
            if cnt > 1:
                collisions += cnt * (cnt - 1) // 2
        return collisions
    
    def is_valid_set(s):
        """Check if set has exactly 12 distinct integers in [0,128]."""
        if len(s) != 12:
            return False
        if len(set(s)) != 12:
            return False
        if any(x < 0 or x > 128 for x in s):
            return False
        return True
    
    # Initial candidate: try structured patterns
    # Pattern 1: Arithmetic progressions create many sum collisions
    candidates = []
    
    # Try various structured sets
    # Even numbers: 0,2,4,...,22
    candidates.append(list(range(0, 24, 2)))
    # Odd numbers: 1,3,5,...,23
    candidates.append(list(range(1, 24, 2)))
    # Multiples of 2 with offset
    candidates.append([2*i + 1 for i in range(12)])
    # Try to create sum collisions: pairs that sum to same value
    # e.g., (0,10), (1,9), (2,8), (3,7), (4,6), (5,11) all sum to 10
    # But we need 12 distinct, so let's be more creative
    
    # Greedy construction: build set to maximize sum collisions
    def greedy_build():
        """Build a set of 12 integers maximizing sum collisions."""
        best = None
        best_coll = -1
        
        # Try multiple starting points
        for start in range(0, 128, 3):
            current = [start]
            used = {start}
            
            for _ in range(11):
                best_next = None
                best_next_coll = -1
                
                for candidate in range(0, 129):
                    if candidate in used:
                        continue
                    
                    new_set = current + [candidate]
                    coll = score_set(new_set)
                    
                    if coll > best_next_coll:
                        best_next_coll = coll
                        best_next = candidate
                
                if best_next is not None:
                    current.append(best_next)
                    used.add(best_next)
            
            if best is None or best_coll < score_set(current):
                best = current
                best_coll = score_set(current)
            
            if time.time() > deadline:
                break
        
        return best
    
    # Run greedy
    greedy_result = greedy_build()
    if greedy_result and is_valid_set(greedy_result):
        s = score_set(greedy_result)
        if s > best_score:
            best_score = s
            best_set = greedy_result
    
    # Local search: try to improve by swapping elements
    if best_set:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current = list(best_set)
            current_score = score_set(current)
            
            for i in range(12):
                for new_val in range(0, 129):
                    if new_val in current and new_val != current[i]:
                        continue
                    if new_val == current[i]:
                        continue
                    
                    old_val = current[i]
                    current[i] = new_val
                    if len(set(current)) == 12 and all(0 <= x <= 128 for x in current):
                        new_score = score_set(current)
                        if new_score > current_score:
                            best_set = list(current)
                            best_score = new_score
                            improved = True
                    current[i] = old_val
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break
    
    # Random restarts if time remains
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Random set with structure
            base = random.randint(0, 116)
            s = []
            used = set()
            for _ in range(12):
                for attempt in range(20):
                    val = base + random.randint(-10, 10)
                    if 0 <= val <= 128 and val not in used:
                        s.append(val)
                        used.add(val)
                        break
                else:
                    # Fill remaining
                    for val in range(0, 129):
                        if val not in used:
                            s.append(val)
                            used.add(val)
                            break
            
            if len(s) == 12 and is_valid_set(s):
                sc = score_set(s)
                if sc > best_score:
                    best_score = sc
                    best_set = s
    
    if best_set is None or not is_valid_set(best_set):
        # Fallback: simple set
        best_set = list(range(12))
    
    return best_set
# EVOLVE-BLOCK-END
