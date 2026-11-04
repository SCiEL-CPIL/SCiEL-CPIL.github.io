import time
import random
import numpy as np
import itertools

# Precomputation for the specific problem instance (q=3, n=6, radius=1)
# However, the function signature is generic, so we must handle generic q, n, radius.
# But the task is specific. We can optimize for q=3, n=6, radius=1 if detected, 
# or just write a generic fast version.
# Given the 120s limit, a generic version with bitsets/sets is fast enough for n=6.

def _precompute_balls(q, n, radius):
    """
    Precompute the set of indices covered by each possible word.
    Returns a list of frozensets, where balls[i] is the set of indices 
    within Hamming distance 'radius' of the word corresponding to index i.
    """
    total = q ** n
    balls = [None] * total
    
    # Precompute all words as tuples of digits
    words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    for idx, word in enumerate(words):
        covered = set()
        # Generate all words within Hamming distance 'radius'
        # For radius 1, this is just the word itself and single-position changes.
        # For general radius, we can use combinations of positions.
        # Since n=6, radius=1 is the main case, let's optimize for that but keep it general.
        
        if radius == 0:
            covered.add(idx)
        elif radius == 1:
            covered.add(idx)
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = list(word)
                    new_word[pos] = val
                    new_idx = int(np.ravel_multi_index(tuple(new_word), (q,) * n))
                    covered.add(new_idx)
        else:
            # General case: iterate over all combinations of positions to change
            # This is expensive for large n, but n=6 is small.
            # We can just iterate over all words and check distance, but that's O(total^2).
            # Better: generate neighbors.
            # For simplicity and given n=6, radius=1 is the target. 
            # If radius > 1, we might need a more generic neighbor generation.
            # Let's assume radius is small.
            # We can use a BFS or just check distance for all words if total is small.
            # For n=6, q=3, total=729. Checking distance for all 729 words for each of 729 is ~500k ops. Fast.
            for other_idx in range(total):
                other_word = words[other_idx]
                dist = sum(1 for a, b in zip(word, other_word) if a != b)
                if dist <= radius:
                    covered.add(other_idx)
        
        balls[idx] = frozenset(covered)
        
    return balls, words

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Precompute balls and words
    balls, words = _precompute_balls(q, n, radius)
    total_points = q ** n
    all_indices = set(range(total_points))
    
    # 2. Greedy Set Cover Initialization
    # We want to pick words that cover the most uncovered points.
    uncovered = set(all_indices)
    selected_indices = []
    
    # To speed up greedy, we can pre-sort or just iterate. 
    # With 729 words, iterating 729 times is fine.
    while uncovered and time.time() < deadline - 10:
        best_idx = -1
        best_count = -1
        # Optimization: Only check indices not already selected? 
        # No, a selected index might cover more *new* points if we consider interactions? 
        # No, greedy set cover adds one set at a time. The marginal gain of adding a set 
        # depends only on the current uncovered set.
        # We can skip indices already in selected_indices to avoid duplicates, 
        # but we might want to re-add? No, set cover doesn't re-add.
        for idx in range(total_points):
            if idx in selected_indices:
                continue
            # Count how many points in 'uncovered' are in balls[idx]
            # Using set intersection
            count = len(balls[idx] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1 or best_count == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered -= balls[best_idx]
        
    current_code = selected_indices[:]
    
    # 3. Local Search
    # Objective: Minimize len(current_code)
    # Heuristic for swaps: Minimize total overlap (sum of coverage counts)
    
    def calc_total_overlap(code):
        # Returns sum of number of codewords covering each point
        # This is a proxy for "efficiency". Lower is better.
        # We can compute this fast using an array.
        counts = np.zeros(total_points, dtype=np.int32)
        for idx in code:
            # Add 1 to all points in balls[idx]
            # Using numpy might be faster if balls were arrays, but they are sets.
            # For small sets, python loops are okay, or we can precompute balls as numpy arrays of indices.
            # Let's precompute balls as numpy arrays of indices for faster overlap calc?
            # Actually, for the swap decision, we just need to know if the code is valid.
            # Validity check: Union of balls covers all points.
            # Overlap calc: Sum of sizes of intersections?
            # Let's stick to a simple validity check and a simple "uniqueness" metric.
            pass
        return 0 # Placeholder, we will implement a better metric if needed.
        
    # Better Local Search Strategy:
    # 1. Try to remove any codeword.
    # 2. If no removal possible, try to swap a codeword with another to enable a removal.
    #    A swap is "good" if it reduces the "essentiality" of the remaining codewords.
    #    Metric: For each codeword c, let U(c) be the set of points covered ONLY by c.
    #    If U(c) is empty, c can be removed.
    #    We want to make U(c) empty for some c.
    #    Swapping c with c' changes the coverage.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Pass 1: Removal
        for i in range(len(current_code)):
            if time.time() > deadline - 5:
                break
            # Try removing current_code[i]
            remaining = current_code[:i] + current_code[i+1:]
            # Check if remaining covers all points
            # Union of balls
            covered = set()
            for idx in remaining:
                covered |= balls[idx]
            if len(covered) == total_points:
                current_code = remaining
                improved = True
                break # Restart loop as indices changed
        
        if time.time() > deadline - 5:
            break
            
        # Pass 2: Swap to enable removal
        # For each codeword, try replacing it with a non-selected word.
        # Accept if the new code is valid.
        # To guide the search, we can prefer swaps that increase the number of 
        # "removable" codewords or reduce overlap.
        # Simple heuristic: Accept any valid swap, then re-run removal.
        # To avoid cycling, we can keep a history or just rely on the removal pass 
        # to drive the size down.
        
        # Optimization: Only try swaps for codewords that are "critical" 
        # (i.e., their removal breaks coverage).
        
        for i in range(len(current_code)):
            if time.time() > deadline - 5:
                break
            
            c_i = current_code[i]
            
            # Check if c_i is removable
            remaining = current_code[:i] + current_code[i+1:]
            covered_without = set()
            for idx in remaining:
                covered_without |= balls[idx]
            
            if len(covered_without) == total_points:
                # It is removable, but Pass 1 should have caught it. 
                # If not, maybe Pass 1 broke early? 
                # In any case, if we are here, it wasn't removed in Pass 1.
                # We should have removed it. Let's force it.
                current_code = remaining
                improved = True
                break
                
            # c_i is essential. Points only covered by c_i:
            essential_points = balls[c_i] - covered_without
            
            # Try to find a replacement c' such that:
            # 1. (Code \ {c_i}) U {c'} is valid.
            # 2. This might allow removal of another codeword later.
            
            # Heuristic: Look for c' that covers many of the essential_points 
            # AND minimizes overlap with others?
            # Actually, if we replace c_i with c', the new essential points for c' 
            # will be different.
            
            # Let's try all candidates c' not in current_code.
            # To speed up, we can filter candidates that cover at least one essential point.
            # If c' doesn't cover any essential point, the code becomes invalid 
            # (unless another codeword covers them, but they were essential to c_i, 
            # so no other codeword covers them).
            # So c' MUST cover all essential_points.
            
            valid_replacements = []
            for c_prime in range(total_points):
                if c_prime in current_code:
                    continue
                if not essential_points.issubset(balls[c_prime]):
                    continue
                valid_replacements.append(c_prime)
                
            if not valid_replacements:
                continue
                
            # Pick the best replacement.
            # Metric: Minimize the "new essential points" for c_prime?
            # Or minimize the impact on other codewords?
            # Let's pick the one that minimizes the total overlap of the new code.
            # Or simply pick the first one and let the next iteration handle it.
            # To avoid getting stuck, let's pick the one that minimizes 
            # len(balls[c_prime] & (Union of others)).
            
            others_covered = covered_without
            
            best_c_prime = None
            min_overlap = float('inf')
            
            for c_prime in valid_replacements:
                # Overlap with others
                overlap = len(balls[c_prime] & others_covered)
                if overlap < min_overlap:
                    min_overlap = overlap
                    best_c_prime = c_prime
                    
            if best_c_prime is not None:
                # Perform swap
                current_code[i] = best_c_prime
                improved = True
                break # Restart to check for removals
                
    return [list(words[idx]) for idx in current_code]
