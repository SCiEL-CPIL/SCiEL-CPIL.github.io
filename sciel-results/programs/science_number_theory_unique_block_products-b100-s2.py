import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin for 90s limit
    
    def is_valid_addition(seq, x, products_set):
        """
        Check if x can be appended to seq.
        Returns (is_valid, new_tails)
        """
        new_tails = [x]
        prod = x
        for a in reversed(seq):
            prod *= a
            new_tails.append(prod)
        
        if products_set.intersection(new_tails):
            return False, None
            
        return True, new_tails

    def greedy_construct(start_val=2, end_val=128, skip_set=None, seed=None):
        if seed is not None:
            random.seed(seed)
            
        seq = []
        products_set = set()
        
        for x in range(start_val, end_val + 1):
            if time.time() > deadline:
                break
                
            if skip_set and x in skip_set:
                continue

            valid, new_tails = is_valid_addition(seq, x, products_set)
            
            if valid:
                seq.append(x)
                products_set.update(new_tails)
                
        return seq

    def backtrack_search(seq, products_set, start_idx, max_depth=3):
        """
        Try to improve the sequence by backtracking.
        Returns improved sequence or original if no improvement.
        """
        if time.time() > deadline:
            return seq
        
        if not seq or len(seq) < 2:
            return seq
        
        # Try removing the last element and trying alternatives
        best = list(seq)
        
        # Try backtracking up to max_depth positions
        for back_depth in range(1, min(max_depth, len(seq)) + 1):
            if time.time() > deadline:
                break
                
            # Remove last back_depth elements
            truncated = seq[:-back_depth] if back_depth < len(seq) else []
            
            # Recompute products_set for truncated
            truncated_products = set()
            for i in range(len(truncated)):
                prod = 1
                for j in range(i, len(truncated)):
                    prod *= truncated[j]
                    truncated_products.add(prod)
            
            # Try to fill remaining positions with better choices
            remaining_positions = range(truncated[-1] + 1 if truncated else 2, 129)
            new_seq = list(truncated)
            new_products = set(truncated_products)
            
            for x in remaining_positions:
                if time.time() > deadline:
                    break
                
                valid, new_tails = is_valid_addition(new_seq, x, new_products)
                if valid:
                    new_seq.append(x)
                    new_products.update(new_tails)
            
            if len(new_seq) > len(best):
                best = new_seq
        
        return best

    best_seq = []
    
    # 1. Standard Greedy (Seed behavior)
    seq = greedy_construct(2)
    if len(seq) > len(best_seq):
        best_seq = seq
    
    # 2. Try different starting values
    for sv in range(3, 20):
        if time.time() > deadline:
            break
        seq = greedy_construct(sv)
        if len(seq) > len(best_seq):
            best_seq = seq
    
    # 3. Randomized trials with skipping
    trials = 0
    while time.time() < deadline - 5.0:  # Leave 5s for backtracking
        trials += 1
        start_val = random.randint(2, 15)
        
        # Randomly skip some small numbers
        skip_set = set()
        for x in range(2, 20):
            if random.random() < 0.3:
                skip_set.add(x)
        
        seq = greedy_construct(start_val, 128, skip_set)
        
        # Try to improve with backtracking
        if len(seq) >= len(best_seq) - 1:  # Only backtrack if promising
            # Recompute products_set
            products_set = set()
            for i in range(len(seq)):
                prod = 1
                for j in range(i, len(seq)):
                    prod *= seq[j]
                    products_set.add(prod)
            
            improved = backtrack_search(seq, products_set, len(seq), max_depth=4)
            if len(improved) > len(best_seq):
                best_seq = improved
        
        if len(seq) > len(best_seq):
            best_seq = seq
    
    # 4. Final backtracking pass on best_seq
    if time.time() < deadline - 2.0 and len(best_seq) > 5:
        products_set = set()
        for i in range(len(best_seq)):
            prod = 1
            for j in range(i, len(best_seq)):
                prod *= best_seq[j]
                products_set.add(prod)
        
        improved = backtrack_search(best_seq, products_set, len(best_seq), max_depth=6)
        if len(improved) > len(best_seq):
            best_seq = improved
    
    return best_seq
