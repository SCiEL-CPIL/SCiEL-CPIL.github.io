import time
import random
import numpy as np

def construct():
    base = 51975
    deadline = time.time() + 115  # Safety margin
    
    # 1. Get divisors
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    # 2. Helper to score a set of classes (for initial comparison if needed, 
    #    but we will maintain state during search)
    
    def greedy_init(order):
        """
        Greedy construction.
        Returns: classes list, final uncovered count, final counts array
        """
        counts = np.zeros(base, dtype=np.int8)
        classes = []
        uncovered = base
        
        for m in order:
            best_r = 0
            best_gain = -1
            # Try all residues
            for r in range(m):
                # Count how many currently uncovered residues are in this class
                # counts[r::m] gives the counts for residues r, r+m, ...
                # We want to add 1 to these.
                # A residue is 'newly covered' if its count was 0.
                gain = np.sum(counts[r::m] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            classes.append([m, best_r])
            counts[best_r::m] += 1
            uncovered -= best_gain
            
            if uncovered == 0:
                break
                
        return classes, uncovered, counts

    def local_search(classes, initial_counts):
        """
        Local search using incremental updates.
        Returns: best classes, best uncovered count
        """
        classes = [list(c) for c in classes]
        counts = initial_counts.copy()
        uncovered = base - np.sum(counts > 0) # Recalculate to be safe? 
        # Actually, we can track uncovered from the greedy step if we pass it, 
        # but recalculating is O(N) which is fine.
        
        improved = True
        passes = 0
        max_passes = 1000 # Safety
        
        while improved and passes < max_passes:
            if time.time() > deadline - 5:
                break
            improved = False
            passes += 1
            
            for i in range(len(classes)):
                m, r_old = classes[i]
                
                # 1. Remove current class
                counts[r_old::m] -= 1
                # Note: uncovered count changes. 
                # Let's just recalculate uncovered at the end of the loop for this modulus?
                # Or track it?
                # If we remove, some residues might become uncovered.
                # If we add, some might become covered.
                
                best_r = r_old
                best_gain = 0 # Gain relative to state without class i
                
                # 2. Try all residues
                for r_new in range(m):
                    # If we add r_new to the current counts (which lacks class i),
                    # how many new residues are covered?
                    # Those are residues in S(m, r_new) where counts[x] == 0.
                    gain = np.sum(counts[r_new::m] == 0)
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r_new
                
                # 3. Apply best residue
                counts[best_r::m] += 1
                
                # 4. Update uncovered count
                # The change in uncovered is:
                # - (number of residues that became covered)
                # + (number of residues that became uncovered when we removed old class)
                # Wait, we removed old class first.
                # Let's just recalculate uncovered for simplicity? 
                # O(N) per modulus is O(D*N) per pass. 
                # D=50, N=52k -> 2.6M ops. 
                # If we do 100 passes, 260M ops. Might be slow in Python/NumPy overhead.
                
                # Better: Track uncovered.
                # Before removal: U_old
                # After removal of (m, r_old): 
                #   Residues in S(m, r_old) that had count 1 become count 0.
                #   Let L = sum(counts[r_old::m] == 0) AFTER decrement? No.
                #   Before decrement, count was C. After, C-1.
                #   If C-1 == 0, it became uncovered. So C == 1.
                #   So losses = sum(counts[r_old::m] == 1) BEFORE decrement.
                
                # This is getting complicated to track perfectly without bugs.
                # Let's use the simple recalculation if it's fast enough.
                # np.sum(counts == 0) is very fast (C-level).
                # Let's assume it's fast enough for ~100 passes.
                
                if best_r != r_old:
                    classes[i] = [m, best_r]
                    improved = True
                    uncovered = np.sum(counts == 0)
                else:
                    # If no change, we still need to update uncovered?
                    # No, if we didn't change the class, the state is the same as before the removal?
                    # No, we removed it and added it back. State is identical.
                    # So uncovered is unchanged.
                    pass
        
        return classes, uncovered

    # --- Main Execution ---
    
    best_classes = None
    best_uncovered = base
    
    # 1. Greedy with increasing order
    classes, unc, counts = greedy_init(divisors)
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = classes
        best_counts = counts
    
    # 2. Greedy with decreasing order
    classes, unc, counts = greedy_init(divisors[::-1])
    if unc < best_uncovered:
        best_uncovered = unc
        best_classes = classes
        best_counts = counts
        
    # 3. Random orders
    random.seed(42)
    start_time = time.time()
    
    # Run random greedy inits until time is low
    while time.time() < deadline - 30: # Leave 30s for local search
        order = divisors[:]
        random.shuffle(order)
        classes, unc, counts = greedy_init(order)
        if unc < best_uncovered:
            best_uncovered = unc
            best_classes = classes
            best_counts = counts
            if best_uncovered == 0:
                break
    
    # 4. Local Search on the best found
    if best_classes is not None and time.time() < deadline - 10:
        best_classes, best_uncovered = local_search(best_classes, best_counts)
        
    return best_classes
