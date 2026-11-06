import numpy as np
import time
import random

N = 128
LO, HI = 128, 256
SUMS = list(range(LO, HI + 1))

def compute_peak_and_coverage(ord_counts, A):
    # ord_counts: array of length 257
    # A: set of ints
    # Returns (peak, covered)
    max_count = 0
    covered = True
    for s in SUMS:
        # Unordered count logic
        # ordered = ord_counts[s]
        # diag = 1 if s % 2 == 0 and s // 2 in A else 0
        # unord = (ordered + diag) // 2
        
        # Optimization: 
        # If s is odd, diag=0, unord = ordered // 2
        # If s is even, diag=1 if s/2 in A, else 0.
        
        ordered = ord_counts[s]
        if s % 2 == 0:
            if (s // 2) in A:
                unord = (ordered + 1) // 2
            else:
                unord = ordered // 2
        else:
            unord = ordered // 2
            
        if unord == 0:
            covered = False
        elif unord > max_count:
            max_count = unord
            
    return max_count, covered

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best_A = None
    best_peak = float('inf')
    
    # We will try a few different initial sizes and restarts
    # The optimal size is likely around 40-60.
    # Let's try multiple restarts.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize
        # Try different sizes
        size = random.randint(35, 55)
        A = set(random.sample(range(N + 1), size))
        
        # Initialize ord_counts
        ord_counts = np.zeros(N * 2 + 1, dtype=np.int64)
        
        # Fill ord_counts
        for a in A:
            for b in A:
                ord_counts[a + b] += 1
                
        # Ensure coverage
        peak, covered = compute_peak_and_coverage(ord_counts, A)
        
        # If not covered, add elements greedily
        # We need to find elements that cover the most uncovered sums.
        attempts = 0
        while not covered and attempts < 100:
            # Find uncovered sums
            uncovered_sums = []
            for s in SUMS:
                ordered = ord_counts[s]
                if s % 2 == 0:
                    if (s // 2) in A:
                        unord = (ordered + 1) // 2
                    else:
                        unord = ordered // 2
                else:
                    unord = ordered // 2
                if unord == 0:
                    uncovered_sums.append(s)
            
            if not uncovered_sums:
                break
                
            # Try adding a random element
            candidates = [i for i in range(N + 1) if i not in A]
            if not candidates:
                break
            
            # Score candidates by how many uncovered sums they help cover
            best_candidate = None
            best_score = -1
            for c in candidates:
                score = 0
                for s in uncovered_sums:
                    # If we add c, does it create a pair for s?
                    # We need b in A such that c + b = s => b = s - c
                    # Or if c + c = s (and c not in A yet, so this pair is new)
                    # Wait, if we add c, new pairs are (c, b) for b in A, and (c, c).
                    # So s is covered if (s-c in A) or (2c == s).
                    # Note: if s-c == c, it's the same as 2c==s.
                    if (s - c) in A or 2 * c == s:
                        # Check if it actually increases the count from 0
                        # If (s-c) in A, we add 2 to ordered (c,b) and (b,c).
                        # If 2c==s, we add 1 to ordered (c,c).
                        # In either case, if it was 0, it becomes >0.
                        score += 1
                if score > best_score:
                    best_score = score
                    best_candidate = c
                    
            if best_candidate is None or best_score == 0:
                # Random pick if no improvement
                best_candidate = random.choice(candidates)
                
            # Add best_candidate
            c = best_candidate
            A.add(c)
            # Update ord_counts
            for b in A:
                if b != c:
                    ord_counts[c + b] += 2
            ord_counts[2 * c] += 1
            
            peak, covered = compute_peak_and_coverage(ord_counts, A)
            attempts += 1
            
        if not covered:
            continue
            
        # Local Search
        current_peak = peak
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Swaps
            # Find best swap
            best_swap = None
            best_swap_peak = current_peak
            best_swap_covered = False
            
            A_list = list(A)
            not_in_A = [i for i in range(N + 1) if i not in A]
            
            for x in A_list:
                if time.time() > deadline:
                    break
                
                # Temporarily remove x
                for b in A:
                    if b != x:
                        ord_counts[x + b] -= 2
                ord_counts[2 * x] -= 1
                A.remove(x)
                
                for y in not_in_A:
                    # Temporarily add y
                    for b in A:
                        if b != y:
                            ord_counts[y + b] += 2
                    ord_counts[2 * y] += 1
                    A.add(y)
                    
                    p, c = compute_peak_and_coverage(ord_counts, A)
                    
                    if c and p < best_swap_peak:
                        best_swap_peak = p
                        best_swap = (x, y)
                        best_swap_covered = True
                        
                    # Undo add y
                    A.remove(y)
                    for b in A:
                        if b != y:
                            ord_counts[y + b] -= 2
                    ord_counts[2 * y] -= 1
                    
                # Undo remove x
                A.add(x)
                for b in A:
                    if b != x:
                        ord_counts[x + b] += 2
                ord_counts[2 * x] += 1
                
            if best_swap:
                x, y = best_swap
                # Perform swap
                # Remove x
                for b in A:
                    if b != x:
                        ord_counts[x + b] -= 2
                ord_counts[2 * x] -= 1
                A.remove(x)
                
                # Add y
                for b in A:
                    if b != y:
                        ord_counts[y + b] += 2
                ord_counts[2 * y] += 1
                A.add(y)
                
                current_peak = best_swap_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = set(A)
                continue # Restart loop to re-evaluate all swaps
            
            # Phase 2: Removals
            # Try to remove elements if it doesn't increase peak and keeps coverage
            best_removal = None
            best_removal_peak = current_peak
            
            A_list = list(A)
            for x in A_list:
                if time.time() > deadline:
                    break
                
                # Temporarily remove x
                for b in A:
                    if b != x:
                        ord_counts[x + b] -= 2
                ord_counts[2 * x] -= 1
                A.remove(x)
                
                p, c = compute_peak_and_coverage(ord_counts, A)
                
                if c and p <= best_removal_peak:
                    # Prefer removal if peak decreases, or if peak same (to shrink set)
                    # But shrinking set might make it harder to cover later? 
                    # If peak is same, it's a "sideways" move. 
                    # Let's only accept if peak decreases, or if peak is same and we want to try smaller sets.
                    # Actually, if peak is same, we might get stuck. 
                    # Let's only accept if p < best_removal_peak.
                    if p < best_removal_peak:
                        best_removal_peak = p
                        best_removal = x
                        
                # Undo remove x
                A.add(x)
                for b in A:
                    if b != x:
                        ord_counts[x + b] += 2
                ord_counts[2 * x] += 1
                
            if best_removal is not None:
                x = best_removal
                # Perform removal
                for b in A:
                    if b != x:
                        ord_counts[x + b] -= 2
                ord_counts[2 * x] -= 1
                A.remove(x)
                
                current_peak = best_removal_peak
                improved = True
                if current_peak < best_peak:
                    best_peak = current_peak
                    best_A = set(A)
                continue

    if best_A is None:
        # Fallback
        A = set(range(64, N + 1))
        return list(A)
        
    return list(best_A)
