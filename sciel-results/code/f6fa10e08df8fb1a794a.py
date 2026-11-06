import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    base = 51975
    
    # Find all divisors of base
    divisors = []
    for i in range(1, int(base**0.5) + 1):
        if base % i == 0:
            divisors.append(i)
            if i != base // i:
                divisors.append(base // i)
    
    # Filter: m > 1, odd
    divisors = [m for m in divisors if m > 1 and m % 2 == 1]
    divisors.sort(reverse=True)
    
    # Precompute indices for each modulus and residue
    indices_by_m = {}
    for m in divisors:
        res_indices = []
        for r in range(m):
            res_indices.append(np.arange(r, base, m))
        indices_by_m[m] = res_indices
    
    # Precompute for each (m, r) the list of indices
    # Also precompute a lookup: for each index i, which (m, r) pairs cover it
    # This helps with swap operations
    
    covered = np.zeros(base, dtype=np.uint8)
    available_moduli = set(divisors)
    classes = []
    
    best_uncovered = base
    best_classes = []
    
    # Greedy phase
    while available_moduli:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_m = None
        best_r = None
        best_indices = None
        
        for m in available_moduli:
            res_indices_list = indices_by_m[m]
            for r in range(m):
                idx = res_indices_list[r]
                gain = np.sum(covered[idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_m = m
                    best_r = r
                    best_indices = idx
                    
        if best_gain <= 0:
            break
            
        classes.append([best_m, best_r])
        covered[best_indices] = 1
        available_moduli.remove(best_m)
        
        current_uncovered = np.sum(covered == 0)
        if current_uncovered < best_uncovered:
            best_uncovered = current_uncovered
            best_classes = [c[:] for c in classes]
            
        if best_uncovered == 0:
            break
    
    # Local search phase: try swaps
    if time.time() < deadline - 10:
        random.seed(42)
        used_moduli = set(c[0] for c in classes)
        
        # Try multiple rounds of local search
        improved = True
        max_rounds = 5
        for round_num in range(max_rounds):
            if time.time() > deadline - 5:
                break
            if not improved:
                break
            improved = False
            
            # Try removing one class and adding a different one
            for i, (m1, r1) in enumerate(classes):
                if time.time() > deadline - 5:
                    break
                    
                # Temporarily remove class i
                idx1 = indices_by_m[m1][r1]
                # Find how many residues are ONLY covered by this class
                # We need to count residues where covered==1 and they're in idx1
                # But we also need to check if they're covered by other classes
                
                # Count residues in idx1 that are covered by exactly 1 class (this one)
                # We can do this by temporarily unmarking and seeing what becomes uncovered
                temp_covered = covered.copy()
                # Unmark residues that are covered by other classes too
                # This is complex; simpler: just try the swap and see if it helps
                
                # Try all available moduli
                for m2 in available_moduli:
                    if m2 == m1:
                        continue
                    for r2 in range(m2):
                        if time.time() > deadline - 5:
                            break
                        idx2 = indices_by_m[m2][r2]
                        
                        # Compute new uncovered count if we swap class i with (m2, r2)
                        # Residues in idx1 that are not in idx2 and not covered by others become uncovered
                        # Residues in idx2 that were uncovered become covered
                        
                        # Simplified: compute net change
                        # uncovered_before = sum(covered == 0)
                        # After swap: residues in idx1 not in idx2 lose coverage (if not covered by others)
                        # Residues in idx2 not in idx1 gain coverage (if they were uncovered)
                        
                        # This is expensive to compute exactly. Let's just do it directly.
                        # Create a temporary covered array
                        temp = covered.copy()
                        # Remove class i
                        # We need to know which residues are covered by other classes
                        # For simplicity, let's just recompute from scratch for the swap
                        # This is O(base) per swap attempt, which is too slow for many attempts
                        
                        # Alternative: use the fact that we can precompute for each index
                        # how many classes cover it. But that's complex with dynamic changes.
                        
                        # Let's just do a quick estimate: 
                        # gain = number of currently uncovered residues in idx2
                        # loss = number of residues in idx1 that are not in idx2 and not covered by other classes
                        # For loss, we approximate by: residues in idx1 that are covered==1 
                        #   (assuming only class i covers them, which is an approximation)
                        
                        gain_est = np.sum(covered[idx2] == 0)
                        # For loss: residues in idx1 that are covered by exactly 1 class
                        # We don't have this info directly. Let's skip exact swap for now.
                        pass
            
            # Try adding a class without removing (if we have budget)
            # Actually, the greedy already adds classes. Let's try a different approach:
            # Try to find a class that, when added, covers the most uncovered residues
            # and see if we can remove a class that covers the fewest unique residues
            
            # Simpler local search: try to replace a poorly-performing class
            # A class is "poorly performing" if it covers few unique residues
            # Compute for each class how many residues it uniquely covers
            class_unique = []
            for i, (m, r) in enumerate(classes):
                idx = indices_by_m[m][r]
                # Count residues in idx that are covered by exactly 1 class
                # We need to know the coverage count
                # Let's compute coverage counts
                break  # Skip for now
            
            # Try a randomized local search
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                    
                # Pick a random class to potentially replace
                if not classes:
                    break
                i = random.randint(0, len(classes) - 1)
                m1, r1 = classes[i]
                idx1 = indices_by_m[m1][r1]
                
                # Try replacing with a random available class
                if not available_moduli:
                    break
                m2 = random.choice(list(available_moduli))
                r2 = random.randint(0, m2 - 1)
                idx2 = indices_by_m[m2][r2]
                
                # Compute new uncovered
                # Temporarily simulate the swap
                # Count uncovered before
                uncovered_before = np.sum(covered == 0)
                
                # After swap:
                # - Remove class i: residues in idx1 may become uncovered if not covered by others
                # - Add class (m2, r2): residues in idx2 that were uncovered become covered
                
                # To compute this exactly, we need to know which residues in idx1 are covered by other classes
                # Let's compute coverage counts for all residues
                # This is expensive but let's try for a limited number of attempts
                
                # Actually, let's just compute the new covered array
                new_covered = covered.copy()
                # We need to unmark residues that are only covered by class i
                # This requires knowing the full coverage structure
                
                # Simplified approach: just recompute from all classes except i plus the new one
                # This is O(num_classes * avg_class_size) which might be acceptable for small num_classes
                if len(classes) < 30:
                    new_covered = np.zeros(base, dtype=np.uint8)
                    for j, (mj, rj) in enumerate(classes):
                        if j != i:
                            new_covered[indices_by_m[mj][rj]] = 1
                    new_covered[idx2] = 1
                    uncovered_after = np.sum(new_covered == 0)
                    
                    if uncovered_after < uncovered_before:
                        # Accept the swap
                        classes[i] = [m2, r2]
                        covered = new_covered
                        used_moduli.discard(m1)
                        used_moduli.add(m2)
                        available_moduli.discard(m2)
                        available_moduli.add(m1)
                        improved = True
                        if uncovered_after < best_uncovered:
                            best_uncovered = uncovered_after
                            best_classes = [c[:] for c in classes]
    
    return best_classes
