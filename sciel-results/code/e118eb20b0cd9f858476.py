import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    def time_left():
        return deadline - time.time()
    
    def compute_all_products(seq):
        """Compute all non-empty consecutive block products."""
        products = set()
        n = len(seq)
        for i in range(n):
            prod = 1
            for j in range(i, n):
                prod *= seq[j]
                products.add(prod)
        return products
    
    def check_valid_addition(seq, x, products_set):
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
    
    def greedy_construct(start_val=2, end_val=128, skip_set=None, seed=None, 
                         future_look=0, weight_future=0.0):
        """
        Greedy construction with optional future collision awareness.
        """
        if seed is not None:
            random.seed(seed)
            
        seq = []
        products_set = set()
        
        for x in range(start_val, end_val + 1):
            if time_left() < 5.0:
                break
                
            if skip_set and x in skip_set:
                continue

            valid, new_tails = check_valid_addition(seq, x, products_set)
            
            if valid:
                # Optional: check future collision risk
                if future_look > 0 and weight_future > 0:
                    # Count how many future candidates would collide
                    future_collisions = 0
                    for y in range(x + 1, min(x + 1 + future_look, end_val + 1)):
                        # Check if y would collide with seq + [x]
                        temp_tails = [y]
                        prod = y
                        for a in reversed(seq + [x]):
                            prod *= a
                            temp_tails.append(prod)
                        if (products_set | set(new_tails)).intersection(temp_tails):
                            future_collisions += 1
                    
                    # Skip if too many future collisions (probabilistic)
                    if future_collisions > 2 and random.random() < weight_future:
                        continue
                
                seq.append(x)
                products_set.update(new_tails)
                
        return seq
    
    def local_search_polish(seq, max_iterations=200):
        """
        Try single-element swaps and insertions to improve the sequence.
        """
        if not seq or len(seq) < 2:
            return seq
        
        best_seq = list(seq)
        best_len = len(best_seq)
        
        for iteration in range(max_iterations):
            if time_left() < 5.0:
                break
            
            improved = False
            
            # Try removing an element and trying alternatives
            for pos in range(len(best_seq)):
                if time_left() < 5.0:
                    break
                
                # Remove element at pos
                temp_seq = best_seq[:pos] + best_seq[pos+1:]
                
                # Recompute products for temp_seq
                temp_products = compute_all_products(temp_seq)
                
                # Try to insert a new element at pos
                prev_val = temp_seq[pos-1] if pos > 0 else 1
                next_val = temp_seq[pos] if pos < len(temp_seq) else 129
                
                for x in range(prev_val + 1, next_val):
                    if x == best_seq[pos]:
                        continue
                    
                    valid, new_tails = check_valid_addition(temp_seq[:pos], x, temp_products)
                    if valid:
                        # Check the rest of the sequence after insertion
                        full_seq = temp_seq[:pos] + [x] + temp_seq[pos:]
                        full_products = compute_all_products(full_seq)
                        
                        if len(full_products) == len(full_seq) * (len(full_seq) + 1) // 2:
                            # All products are distinct
                            if len(full_seq) > best_len:
                                best_seq = full_seq
                                best_len = len(best_seq)
                                improved = True
                                break
                            elif len(full_seq) == best_len:
                                # Try to see if this is "better" (e.g., smaller elements)
                                if sum(full_seq) < sum(best_seq):
                                    best_seq = full_seq
                                    improved = True
                                    break
            
            if not improved:
                break
        
        return best_seq
    
    def branch_and_bound_search(seq, products_set, start_idx, max_depth, best_len):
        """
        Branch and bound: try to extend the sequence by choosing from remaining candidates.
        Uses pruning based on upper bound on possible extensions.
        """
        if time_left() < 5.0:
            return seq, products_set
        
        if max_depth <= 0 or len(seq) + (128 - start_idx + 1) <= best_len:
            return seq, products_set
        
        best = list(seq)
        best_products = set(products_set)
        best_len_found = len(best)
        
        # Try each candidate from start_idx
        for x in range(start_idx, 129):
            if time_left() < 5.0:
                break
            
            valid, new_tails = check_valid_addition(best, x, best_products)
            if valid:
                candidate_seq = best + [x]
                candidate_products = best_products | set(new_tails)
                
                # Upper bound: can we possibly beat best_len?
                if len(candidate_seq) + (128 - x) > best_len_found:
                    # Recurse
                    improved_seq, improved_products = branch_and_bound_search(
                        candidate_seq, candidate_products, x + 1, max_depth - 1, best_len_found
                    )
                    if len(improved_seq) > best_len_found:
                        best = improved_seq
                        best_products = improved_products
                        best_len_found = len(best)
        
        return best, best_products
    
    def iterative_deepening_search(seq, products_set, max_iterations=50):
        """
        Iterative deepening: gradually increase the search depth.
        """
        best_seq = list(seq)
        best_products = set(products_set)
        
        for depth in range(1, min(8, 128 - seq[-1] + 1) if seq else 8):
            if time_left() < 5.0:
                break
            
            improved_seq, improved_products = branch_and_bound_search(
                best_seq, best_products, best_seq[-1] + 1 if best_seq else 2, 
                depth, len(best_seq)
            )
            
            if len(improved_seq) > len(best_seq):
                best_seq = improved_seq
                best_products = improved_products
        
        return best_seq
    
    # Phase 1: Multiple greedy strategies
    best_seq = []
    
    # Standard greedy
    seq = greedy_construct(2)
    if len(seq) > len(best_seq):
        best_seq = seq
    
    # Try different starting values
    for sv in range(3, 25):
        if time_left() < 10.0:
            break
        seq = greedy_construct(sv)
        if len(seq) > len(best_seq):
            best_seq = seq
    
    # Randomized greedy with skipping
    random.seed(42)
    for trial in range(30):
        if time_left() < 10.0:
            break
        
        start_val = random.randint(2, 18)
        skip_set = set()
        for x in range(2, 25):
            if random.random() < 0.25:
                skip_set.add(x)
        
        seq = greedy_construct(start_val, 128, skip_set, seed=trial)
        
        if len(seq) > len(best_seq):
            best_seq = seq
    
    # Phase 2: Local search polish
    if len(best_seq) > 3 and time_left() > 15.0:
        polished = local_search_polish(best_seq, max_iterations=100)
        if len(polished) > len(best_seq):
            best_seq = polished
    
    # Phase 3: Branch and bound from current best
    if len(best_seq) > 3 and time_left() > 20.0:
        products_set = compute_all_products(best_seq)
        improved = iterative_deepening_search(best_seq, products_set, max_iterations=30)
        if len(improved) > len(best_seq):
            best_seq = improved
    
    # Phase 4: Final local search polish
    if len(best_seq) > 3 and time_left() > 10.0:
        polished = local_search_polish(best_seq, max_iterations=150)
        if len(polished) > len(best_seq):
            best_seq = polished
    
    # Final validation
    if best_seq:
        products = compute_all_products(best_seq)
        expected_count = len(best_seq) * (len(best_seq) + 1) // 2
        if len(products) != expected_count:
            # Fallback: find the longest valid prefix
            for cut in range(len(best_seq), 0, -1):
                prefix = best_seq[:cut]
                p = compute_all_products(prefix)
                if len(p) == cut * (cut + 1) // 2:
                    best_seq = prefix
                    break
    
    return best_seq
