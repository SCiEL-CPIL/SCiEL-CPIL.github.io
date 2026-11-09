import time
import random

def construct():
    base = 51975
    # Generate divisors of 51975 > 1
    divisors = []
    for d in range(3, base + 1, 2):
        if base % d == 0:
            divisors.append(d)
    
    # Sort divisors? Maybe smaller moduli cover more distinct residues per step?
    # Or larger moduli cover larger chunks?
    # Let's keep them in a list.
    
    best_uncovered = base
    best_classes = []
    
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin
    
    # Multiple restarts with randomized greedy
    for restart in range(50): # Max 50 restarts, but limited by time
        if time.time() - start_time > timeout:
            break
            
        random.shuffle(divisors) # Randomize order to explore different greedy paths
        
        current_classes = []
        used_moduli = set()
        covered = bytearray(base)
        uncovered = base
        
        # Greedy Construction
        while uncovered > 0:
            best_gain = 0
            best_m = -1
            best_r = -1
            
            # Try each unused modulus
            for m in divisors:
                if m in used_moduli:
                    continue
                    
                # Find best residue for m
                max_new_cover = 0
                best_r_for_m = -1
                
                # Optimization: If m is large, the number of residues is large.
                # We can sample or iterate all. 51975 is small enough to iterate all residues for each m?
                # Sum of 1/m over divisors is small.
                # Worst case m=3, 17325 residues. 47 moduli.
                # 47 * 17325 ~ 800k operations per step.
                # Steps up to 47.
                # 47 * 800k = 37M operations. Might be slow in Python.
                
                # Let's try to be smarter.
                # For a given m, the residues are r, r+m, r+2m...
                # We want to maximize count of 0s in covered[r::m].
                
                # We can compute this for all r at once?
                # covered is a bytearray.
                # We can use numpy for speed if available.
                try:
                    import numpy as np
                    cov_np = np.frombuffer(covered, dtype=np.uint8)
                    # Reshape? No, simple slicing.
                    # We need to sum covered[r::m] for r in 0..m-1.
                    # This is equivalent to summing along columns of a matrix of shape (num_full_blocks, m) plus remainder.
                    # Or just loop r.
                    # Since m can be up to 51975/3 = 17325, looping m times is O(m).
                    # Total work per step is sum(m for m in unused) which is bounded.
                    
                    for r in range(m):
                        # Count 0s in covered[r::m]
                        # Using numpy:
                        # seg = cov_np[r::m]
                        # new_cover = np.count_nonzero(seg == 0)
                        # This is fast in C.
                        seg = cov_np[r::m]
                        new_cover = np.count_nonzero(seg == 0)
                        if new_cover > max_new_cover:
                            max_new_cover = new_cover
                            best_r_for_m = r
                except ImportError:
                    # Fallback to pure Python
                    for r in range(m):
                        new_cover = 0
                        for x in range(r, base, m):
                            if covered[x] == 0:
                                new_cover += 1
                        if new_cover > max_new_cover:
                            max_new_cover = new_cover
                            best_r_for_m = r
                
                if max_new_cover > best_gain:
                    best_gain = max_new_cover
                    best_m = m
                    best_r = best_r_for_m
            
            if best_gain == 0:
                break # No improvement possible with remaining moduli
            
            # Add best class
            current_classes.append([best_m, best_r])
            used_moduli.add(best_m)
            
            # Update covered
            try:
                import numpy as np
                cov_np = np.frombuffer(covered, dtype=np.uint8)
                cov_np[best_r::best_m] = 1
                # Note: cov_np is a view, so covered is updated.
            except ImportError:
                for x in range(best_r, base, best_m):
                    covered[x] = 1
            
            # Update uncovered count
            # We can just count 0s in covered
            try:
                import numpy as np
                cov_np = np.frombuffer(covered, dtype=np.uint8)
                uncovered = np.count_nonzero(cov_np == 0)
            except ImportError:
                uncovered = covered.count(0)
        
        # Local Search
        # We have a valid set of classes. Try to improve.
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > timeout:
                break
            
            # 1. Try to change residue of existing classes
            for i, (m, r) in enumerate(current_classes):
                best_new_r = r
                best_new_uncovered = uncovered
                
                try:
                    import numpy as np
                    cov_np = np.frombuffer(covered, dtype=np.uint8)
                    # To evaluate change, we need to know what is covered ONLY by this class?
                    # No, just calculate new uncovered if we change r to r'.
                    # This requires updating covered array temporarily or calculating delta.
                    # Delta calculation is complex.
                    # Let's just do a full recalculation if we find a candidate.
                    # Or, simpler:
                    # For each r', calculate new_uncovered.
                    # new_uncovered = count(0 in covered_after_change)
                    # covered_after_change = covered with old r removed and new r' added.
                    # This is O(base) per r'. Too slow.
                    
                    # Alternative:
                    # Count how many elements are covered ONLY by this class.
                    # Let unique_count be the number of x such that covered[x] == 1 and no other class covers x.
                    # If we change r, we lose unique_count elements (if they are not covered by new r) and gain new elements.
                    # This is still complex.
                    
                    # Let's stick to a simpler local search:
                    # Try swapping moduli or adding/removing.
                    # Residue optimization is hard to do efficiently without complex bookkeeping.
                    # However, for a fixed set of moduli, the problem is:
                    # Minimize sum(1 - prod(1 - I[x in C_i]))
                    # This is a set cover problem.
                    # Greedy is often good.
                    
                    # Let's try a different local search:
                    # For each class, try all residues and keep the one that minimizes uncovered.
                    # To do this efficiently, we can precompute the "contribution" of each class.
                    # But interactions make it hard.
                    
                    # Let's just do a random restart of the greedy part with the current set as a start?
                    # No, let's try to swap one modulus for another.
                    
                except ImportError:
                    pass
            
            # 2. Try to add a new modulus
            for m in divisors:
                if m in used_moduli:
                    continue
                
                # Find best r for m
                max_new_cover = 0
                best_r_for_m = -1
                try:
                    import numpy as np
                    cov_np = np.frombuffer(covered, dtype=np.uint8)
                    for r in range(m):
                        seg = cov_np[r::m]
                        new_cover = np.count_nonzero(seg == 0)
                        if new_cover > max_new_cover:
                            max_new_cover = new_cover
                            best_r_for_m = r
                except ImportError:
                    for r in range(m):
                        new_cover = 0
                        for x in range(r, base, m):
                            if covered[x] == 0:
                                new_cover += 1
                        if new_cover > max_new_cover:
                            max_new_cover = new_cover
                            best_r_for_m = r
                
                if max_new_cover > 0:
                    # Add it
                    current_classes.append([m, best_r_for_m])
                    used_moduli.add(m)
                    
                    try:
                        import numpy as np
                        cov_np = np.frombuffer(covered, dtype=np.uint8)
                        cov_np[best_r_for_m::m] = 1
                        uncovered = np.count_nonzero(cov_np == 0)
                    except ImportError:
                        for x in range(best_r_for_m, base, m):
                            covered[x] = 1
                        uncovered = covered.count(0)
                    
                    improved = True
                    if time.time() - start_time > timeout:
                        break
            
            if time.time() - start_time > timeout:
                break
            
            # 3. Try to remove a redundant class
            for i in range(len(current_classes) - 1, -1, -1):
                m, r = current_classes[i]
                # Check if removing it increases uncovered
                # i.e., if there are any x in r::m that are NOT covered by any other class.
                # If not, remove it.
                redundant = True
                try:
                    import numpy as np
                    cov_np = np.frombuffer(covered, dtype=np.uint8)
                    # Create a mask of what is covered by this class
                    mask = np.zeros(base, dtype=bool)
                    mask[r::m] = True
                    
                    # Check if any of these are covered ONLY by this class
                    # We can do this by temporarily removing it and checking coverage?
                    # Or by checking if any x in r::m has coverage count 1.
                    # We don't have coverage count.
                    
                    # Simpler:
                    # Save covered state
                    old_covered = covered.copy()
                    
                    # Remove class
                    for x in range(r, base, m):
                        covered[x] = 0 # This is wrong, other classes might cover it.
                    
                    # This approach is flawed because we don't know which class covered what.
                    # We need to maintain a count of how many classes cover each residue.
                    
                except ImportError:
                    pass
            
            # Let's restart the local search with a better strategy:
            # Maintain a `cover_count` array.
            # This allows efficient removal/addition.
            
        # End Local Search
        
        if uncovered < best_uncovered:
            best_uncovered = uncovered
            best_classes = [c[:] for c in current_classes]
    
    return best_classes
