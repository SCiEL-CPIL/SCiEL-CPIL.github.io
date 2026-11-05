import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    # Precompute reciprocal weights as floats for fast comparison, 
    # but use Fraction for final exact calculation if needed? 
    # The task says "score is its floating serialization", so float is fine for internal logic.
    weights = [1.0 / i for i in range(N + 1)]
    
    def is_valid(S):
        """Check if set S contains no 3-AP."""
        S_list = list(S)
        S_set = set(S_list)
        # Optimization: if size is small, brute force pairs
        for i, a in enumerate(S_list):
            for b in S_list[i+1:]:
                # Check if 2*b - a is in S and is not a or b (nonconstant)
                c = 2 * b - a
                if c in S_set and c != a and c != b:
                    return False
        return True

    def can_add(x, S):
        """Check if x can be added to S without creating a 3-AP. O(|S|)"""
        S_set = set(S)
        for a in S:
            # Case 1: x is the middle term. a, x, c => c = 2x - a
            c = 2 * x - a
            if c in S_set and c != a and c != x:
                return False
            # Case 2: x is an end term. a, b, x or x, b, a
            # b = (a + x) / 2
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in S_set and b != a and b != x:
                    return False
        return True

    def score(S):
        """Calculate the reciprocal sum."""
        return sum(weights[x] for x in S)

    # 1. Greedy Initialization
    # Try to build a good initial set by adding elements in order of decreasing weight (increasing index? No, decreasing value -> increasing reciprocal)
    # Smaller numbers have higher reciprocal weight. So we prioritize 1, 2, 3...
    S = set()
    for x in range(1, N + 1):
        if time.time() > deadline - 5.0: break
        if can_add(x, S):
            S.add(x)
    
    current_score = score(S)
    best_S = set(S)
    best_score = current_score
    
    # 2. Simulated Annealing
    # Parameters
    T_start = 0.05  # Initial temperature (scale of score differences)
    T_end = 0.001   # Final temperature
    alpha = 0.999   # Cooling factor
    
    T = T_start
    no_improve_count = 0
    max_no_improve = 5000
    
    # Shuffle candidates for randomness in moves
    candidates_add = [x for x in range(1, N + 1) if x not in S]
    candidates_remove = list(S)
    
    while time.time() < deadline - 5.0 and T > T_end:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random element
            if not candidates_add:
                T *= alpha
                continue
            idx = random.randint(0, len(candidates_add) - 1)
            x = candidates_add[idx]
            if can_add(x, S):
                S.add(x)
                new_score = current_score + weights[x]
                delta = new_score - current_score
                # Always accept if better, else accept with prob exp(-delta/T)
                if delta > 0 or random.random() < pow(2.718281828, -delta / T):
                    current_score = new_score
                    candidates_add.pop(idx)
                    candidates_remove.append(x)
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
        elif move_type < 0.7:
            # Remove a random element
            if not candidates_remove:
                T *= alpha
                continue
            idx = random.randint(0, len(candidates_remove) - 1)
            x = candidates_remove[idx]
            S.remove(x)
            new_score = current_score - weights[x]
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, -delta / T):
                current_score = new_score
                candidates_remove.pop(idx)
                candidates_add.append(x)
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Revert
                S.add(x)
                candidates_remove.insert(idx, x)
                candidates_add.remove(x)
                no_improve_count += 1
                
        else:
            # Swap: Remove x, Add y
            if not candidates_remove or not candidates_add:
                T *= alpha
                continue
            idx_r = random.randint(0, len(candidates_remove) - 1)
            idx_a = random.randint(0, len(candidates_add) - 1)
            x = candidates_remove[idx_r]
            y = candidates_add[idx_a]
            
            # Try swap
            S.remove(x)
            # Check if y can be added to S (which no longer has x)
            if can_add(y, S):
                S.add(y)
                new_score = current_score - weights[x] + weights[y]
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, -delta / T):
                    current_score = new_score
                    candidates_remove.pop(idx_r)
                    candidates_remove.append(y)
                    candidates_add.pop(idx_a)
                    candidates_add.append(x)
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Revert
                    S.remove(y)
                    S.add(x)
                    candidates_remove.insert(idx_r, x)
                    candidates_remove.remove(y)
                    candidates_add.insert(idx_a, y)
                    candidates_add.remove(x)
                    no_improve_count += 1
            else:
                # Revert
                S.add(x)
                no_improve_count += 1
        
        T *= alpha
        
        if no_improve_count > max_no_improve:
            # Restart with a slightly perturbed best solution or just continue
            # Let's just reset counter to allow more exploration
            no_improve_count = 0

    # 3. Final Local Search Polish on Best Solution
    S = set(best_S)
    current_score = best_score
    
    # Try to add any remaining elements that fit
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        for x in range(1, N + 1):
            if x in S:
                continue
            if can_add(x, S):
                S.add(x)
                current_score += weights[x]
                improved = True
    
    # Try swaps to improve score
    improved = True
    while improved and time.time() < deadline - 0.5:
        improved = False
        # Try removing large elements and adding smaller ones
        for x in sorted(S, reverse=True):
            if time.time() > deadline - 0.5: break
            S.remove(x)
            old_score = current_score - weights[x]
            # Try adding any missing y
            for y in range(1, x):
                if y in S:
                    continue
                if can_add(y, S):
                    S.add(y)
                    new_score = old_score + weights[y]
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        break
                    else:
                        S.remove(y)
            
            if not improved:
                S.add(x)
                current_score += weights[x]
            else:
                # We broke out of y loop with improvement, so x is already removed
                # But we need to update the loop context? 
                # Actually, if we improved, we added y and removed x. 
                # The outer loop is iterating over a static list of S? No, S is being modified.
                # This logic is a bit messy. Let's simplify the polish.
                break

    # Final Verification
    S_list = sorted(S)
    valid = True
    S_set = set(S_list)
    for i, a in enumerate(S_list):
        for b in S_list[i+1:]:
            c = 2 * b - a
            if c in S_set and c != a and c != b:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to greedy
        S = set()
        for x in range(1, N + 1):
            if can_add(x, S):
                S.add(x)

    return list(S)
# EVOLVE-BLOCK-END
