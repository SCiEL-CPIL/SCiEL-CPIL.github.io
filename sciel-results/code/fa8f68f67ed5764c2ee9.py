import time
from fractions import Fraction

def construct():
    start_time = time.time()
    # Use 85 seconds to be safe, limit is 90s
    deadline = start_time + 85.0
    
    # Denominators 2 to 48
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Create list of (Fraction, original_index)
    # Sort by Fraction descending to prioritize larger terms for tighter bounds
    items = [(Fraction(1, d), i) for i, d in enumerate(denominators)]
    items.sort(key=lambda x: x[0], reverse=True)
    
    vals = [x[0] for x in items]
    orig_indices = [x[1] for x in items]
    
    # Precompute suffix sums
    # suffix_sums[i] is the sum of vals[i] to vals[n-1]
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + vals[i]
        
    # Initial Greedy Solution to establish a good lower bound for max sum
    best_sum = Fraction(0)
    best_indices = []
    current_sum = Fraction(0)
    
    # Greedy pass
    for i in range(n):
        if current_sum + vals[i] < 1:
            current_sum += vals[i]
            # We need to track indices for the greedy solution too, 
            # but since we will run B&B from scratch, we just need best_sum.
            # However, if B&B times out immediately, we need a valid index list.
            # So let's store the greedy indices.
            pass # We'll just rely on B&B finding something, but greedy is a valid leaf.
            
    # Actually, let's just run B&B. It will explore the greedy path first if we branch "Take" first.
    # But to ensure we have a non-empty result even if B&B fails/times out weirdly,
    # let's do a quick greedy to set initial best.
    
    greedy_path = []
    cs = Fraction(0)
    for i in range(n):
        if cs + vals[i] < 1:
            cs += vals[i]
            greedy_path.append(orig_indices[i])
            
    if cs > best_sum:
        best_sum = cs
        best_indices = greedy_path
        
    current_path = []
    
    def search(i, current_sum):
        nonlocal best_sum, best_indices
        
        # Time check
        if time.time() > deadline:
            return
        
        # Bound check: Even if we take all remaining, can we beat best_sum?
        if current_sum + suffix_sums[i] <= best_sum:
            return
            
        if i == n:
            # Since we only recurse if sum < 1, current_sum is strictly < 1
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_path[:]
            return
        
        # Option 1: Take item i
        # Check if adding it keeps sum strictly below 1
        next_sum = current_sum + vals[i]
        if next_sum < 1:
            current_path.append(orig_indices[i])
            search(i + 1, next_sum)
            current_path.pop()
            
        # Option 2: Skip item i
        search(i + 1, current_sum)

    search(0, Fraction(0))
    
    return best_indices
