"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    def _elapsed():
        return time.time() - start_time
    
    def _score_set(U):
        """Compute the score for a set U of nonnegative integers containing 0."""
        U = sorted(set(U))
        if 0 not in U:
            return -1.0
        if len(U) < 2 or max(U) == 0:
            return -1.0
        
        # Compute sumset and difference set
        n = len(U)
        plus_set = set()
        minus_set = set()
        for i in range(n):
            for j in range(n):
                plus_set.add(U[i] + U[j])
                minus_set.add(U[i] - U[j])
        
        ratio = len(minus_set) / len(plus_set)
        if ratio <= 1:
            return -1.0
        denom = math.log2(2 * max(U) + 1)
        if denom <= 0:
            return -1.0
        return 1 + math.log2(ratio) / denom
    
    def _golomb_ruler(n, max_val=100000):
        """Generate a Golomb-like ruler of n marks using greedy approach."""
        marks = [0]
        for i in range(1, n):
            best_pos = None
            best_min_diff = -1
            # Try positions from 1 to some limit
            for pos in range(1, min(max_val, 50000)):
                if pos in marks:
                    continue
                # Check if adding pos creates duplicate differences
                diffs = set()
                valid = True
                for m in marks:
                    d = abs(pos - m)
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if valid:
                    # Prefer smaller positions to keep max(U) small
                    best_pos = pos
                    break
            if best_pos is None:
                # Fallback: just pick a large prime-like gap
                best_pos = marks[-1] + i * i + 1
            marks.append(best_pos)
        return marks
    
    def _local_search(U, max_iter=5000):
        """Local search to improve the score."""
        U = sorted(set(U))
        if 0 not in U:
            U = sorted(set(U | {0}))
        current_score = _score_set(U)
        if current_score < 0:
            return U, current_score
        
        n = len(U)
        best_U = U[:]
        best_score = current_score
        
        for iteration in range(max_iter):
            if _elapsed() > deadline - 5:
                break
            
            # Random perturbation
            U_new = U[:]
            
            # Choose a random operation
            op = random.random()
            
            if op < 0.3 and len(U) > 2:
                # Remove a random element (keep 0)
                candidates = [i for i in range(n) if U[i] != 0]
                if candidates:
                    idx = random.choice(candidates)
                    U_new = [U[i] for i in range(n) if i != idx]
            elif op < 0.6:
                # Add a random element
                max_u = max(U) if U else 1
                new_val = random.randint(1, max_u * 2 + 100)
                U_new.append(new_val)
            elif op < 0.8:
                # Swap: remove one, add another
                if len(U) > 2:
                    candidates = [i for i in range(n) if U[i] != 0]
                    if candidates:
                        idx = random.choice(candidates)
                        U_new = [U[i] for i in range(n) if i != idx]
                        max_u = max(U) if U else 1
                        new_val = random.randint(1, max_u * 2 + 100)
                        U_new.append(new_val)
            else:
                # Shift a random element
                candidates = [i for i in range(n) if U[i] != 0]
                if candidates:
                    idx = random.choice(candidates)
                    delta = random.randint(-max(1, U[idx]//10), max(1, U[idx]//10) + 10)
                    new_val = U[idx] + delta
                    if new_val > 0:
                        U_new[idx] = new_val
                    elif new_val == 0 and 0 in U_new and idx != 0:
                        # Would create duplicate 0, skip
                        continue
            
            U_new = sorted(set(U_new))
            if 0 not in U_new or len(U_new) < 2:
                continue
            
            new_score = _score_set(U_new)
            if new_score > best_score:
                best_score = new_score
                best_U = U_new[:]
                U = U_new[:]
                n = len(U)
            elif new_score > current_score - 0.01:
                # Accept with some probability (simulated annealing-like)
                if random.random() < 0.3:
                    U = U_new[:]
                    n = len(U)
                    current_score = new_score
        
        return best_U, best_score
    
    # Initial candidates
    best_score = -1.0
    best_U = [0, 1, 3]
    
    # Try Golomb rulers of various sizes
    for n in range(3, 15):
        if _elapsed() > deadline - 10:
            break
        U = _golomb_ruler(n)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Try geometric sets
    for b in range(2, 30):
        for n in range(2, 12):
            if _elapsed() > deadline - 10:
                break
            U = [0]
            val = 1
            for i in range(1, n):
                val *= b
                U.append(val)
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Try small structured sets
    for a in range(2, 15):
        for b in range(a + 1, 50):
            if _elapsed() > deadline - 10:
                break
            U = [0, 1, a, b]
            s = _score_set(U)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Local search from best found so far
    if _elapsed() < deadline - 20:
        best_U, best_score = _local_search(best_U, max_iter=3000)
    
    # Try local search from Golomb rulers
    for n in range(4, 12):
        if _elapsed() > deadline - 15:
            break
        U = _golomb_ruler(n)
        s = _score_set(U)
        if s > best_score:
            # Refine with local search
            U, s = _local_search(U, max_iter=2000)
            if s > best_score:
                best_score = s
                best_U = U[:]
    
    # Final local search polish
    if _elapsed() < deadline - 5:
        best_U, best_score = _local_search(best_U, max_iter=2000)
    
    return best_U
# EVOLVE-BLOCK-END
