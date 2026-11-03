"""
Greedy constructive heuristic for maximizing sumset/difference-set exponent.
Strategy: Sequentially add elements to U (starting with 0) that maximize the
ratio |U-U|/|U+U| while keeping max(U) controlled. This avoids the structural
constraints of mixed-radix representations and explores the space of all
finite sets of nonnegative integers.
"""
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def score_set(U):
        U_set = set(U)
        max_U = max(U)
        if max_U == 0:
            return 0.0
        n = len(U_set)
        if n == 0:
            return 0.0

        # Compute sumset and difference set
        sumset = set()
        diffset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)

        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0

        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return exponent

    def greedy_construct(max_steps=30, max_val_limit=200, restarts=20):
        best_set = [0]
        best_score = 0.0
        
        for r in range(restarts):
            if time.time() > deadline - 10:
                break
                
            current_set = {0}
            current_sumset = {0}
            current_diffset = {0}
            current_max = 0
            current_score = 0.0
            
            # Greedy steps
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                    
                best_candidate = None
                best_marginal_gain = -1.0
                
                # Candidate range: 1 to max_val_limit
                # We prioritize smaller values to keep max(U) low, but explore up to limit
                # To speed up, we can limit the candidate search space based on current max
                search_upper = min(max_val_limit, current_max + 20 if step > 0 else 50)
                
                # We should consider candidates that are not already in the set
                # and are non-negative
                candidates = []
                for x in range(1, search_upper + 1):
                    if x not in current_set:
                        candidates.append(x)
                
                if not candidates:
                    break
                    
                # Shuffle to avoid bias in ties
                random.shuffle(candidates)
                
                # Evaluate each candidate
                for x in candidates:
                    # New sumset: old_sumset + {x + a for a in current_set}
                    new_sumset = set(current_sumset)
                    for a in current_set:
                        new_sumset.add(x + a)
                        
                    # New diffset: old_diffset + {x - a for a in current_set} + {a - x for a in current_set}
                    new_diffset = set(current_diffset)
                    for a in current_set:
                        new_diffset.add(x - a)
                        new_diffset.add(a - x)
                        
                    new_max = max(current_max, x)
                    if new_max == 0:
                        continue
                        
                    new_ratio = len(new_diffset) / len(new_sumset)
                    if new_ratio <= 0:
                        continue
                        
                    try:
                        new_score = 1 + math.log(new_ratio) / math.log(2 * new_max + 1)
                    except (ValueError, OverflowError, ZeroDivisionError):
                        continue
                        
                    # We want to maximize the score.
                    # However, adding an element always increases the denominator (max_U) potentially.
                    # We compare the new score against the current score.
                    if new_score > current_score:
                        # To break ties or prefer smaller max, we can add a small penalty for larger max
                        # Or just take the first one that improves.
                        # Let's take the one with the highest new_score.
                        if new_score > best_marginal_gain:
                            best_marginal_gain = new_score
                            best_candidate = x
                            best_new_sumset = new_sumset
                            best_new_diffset = new_diffset
                            best_new_max = new_max
                            best_new_score = new_score

                if best_candidate is None:
                    break
                    
                # Commit the best candidate
                current_set.add(best_candidate)
                current_sumset = best_new_sumset
                current_diffset = best_new_diffset
                current_max = best_new_max
                current_score = best_new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = sorted(current_set)
                    
            # If we didn't improve, we might try a different random seed or break
            # But the restarts loop handles different random shuffles
            
        return best_set, best_score

    # Initial seed
    best_set, best_score = greedy_construct(max_steps=20, max_val_limit=100, restarts=5)
    
    # Refine with more steps and larger limit if time permits
    if time.time() < deadline - 20:
        # Try to extend the best set found so far
        current_set = set(best_set)
        current_max = max(current_set) if current_set else 0
        
        # Recompute sum/diff sets for the current best
        current_sumset = set()
        current_diffset = set()
        for a in current_set:
            for b in current_set:
                current_sumset.add(a + b)
                current_diffset.add(a - b)
                
        current_score = best_score
        
        # Try adding more elements greedily
        for step in range(20):
            if time.time() > deadline - 5:
                break
                
            best_candidate = None
            best_new_score = current_score
            best_new_sumset = None
            best_new_diffset = None
            best_new_max = current_max
            
            search_upper = min(300, current_max + 50)
            
            candidates = []
            for x in range(1, search_upper + 1):
                if x not in current_set:
                    candidates.append(x)
            
            if not candidates:
                break
                
            random.shuffle(candidates)
            
            for x in candidates:
                new_sumset = set(current_sumset)
                for a in current_set:
                    new_sumset.add(x + a)
                    
                new_diffset = set(current_diffset)
                for a in current_set:
                    new_diffset.add(x - a)
                    new_diffset.add(a - x)
                    
                new_max = max(current_max, x)
                if new_max == 0:
                    continue
                    
                new_ratio = len(new_diffset) / len(new_sumset)
                if new_ratio <= 0:
                    continue
                    
                try:
                    new_score = 1 + math.log(new_ratio) / math.log(2 * new_max + 1)
                except (ValueError, OverflowError, ZeroDivisionError):
                    continue
                    
                if new_score > best_new_score:
                    best_new_score = new_score
                    best_candidate = x
                    best_new_sumset = new_sumset
                    best_new_diffset = new_diffset
                    best_new_max = new_max

            if best_candidate is None:
                break
                
            current_set.add(best_candidate)
            current_sumset = best_new_sumset
            current_diffset = best_new_diffset
            current_max = best_new_max
            current_score = best_new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = sorted(current_set)

    return best_set

if __name__ == "__main__":
    # Simple test
    U = construct()
    print(f"Constructed set: {U}")
    print(f"Size: {len(U)}")
    if U:
        print(f"Max: {max(U)}")
        s = score_set(U)
        print(f"Score: {s}")
