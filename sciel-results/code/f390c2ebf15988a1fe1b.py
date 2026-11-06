"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    base = 51975
    
    # Generate all divisors of base greater than 1
    divisors = []
    for d in range(2, base + 1):
        if base % d == 0:
            divisors.append(d)
    
    # Sort by size descending for Greedy Phase
    divisors_desc = sorted(divisors, reverse=True)
    
    covered_count = np.zeros(base, dtype=np.int32)
    classes = []
    
    # Greedy Phase: Large moduli first
    for m in divisors_desc:
        # Vectorized gain calculation
        # Reshape to (base//m, m) so that columns correspond to residues
        matrix = covered_count.reshape(base // m, m)
        gains = np.sum(matrix == 0, axis=0)
        
        best_r = np.argmax(gains)
        best_gain = gains[best_r]
        
        if best_gain > 0:
            classes.append([m, best_r])
            indices = np.arange(best_r, base, m)
            covered_count[indices] += 1
            
    # Local Search Phase
    # Sort classes by modulus ascending to prioritize small moduli for gap filling
    # as per directive "swap residues of smaller moduli to fill gaps"
    # We will iterate through classes in this order
    
    max_iterations = 50
    for _ in range(max_iterations):
        improved = False
        
        # Iterate in ascending order of modulus
        # We need to sort indices based on modulus
        # classes is a list of [m, r]
        # We can just sort a copy of indices? Or just iterate and pick?
        # Since number of classes is small (< 48), sorting is cheap.
        class_indices = sorted(range(len(classes)), key=lambda i: classes[i][0])
        
        for i in class_indices:
            m, r = classes[i]
            old_indices = np.arange(r, base, m)
            
            # Calculate Loss: residues that will become uncovered if we remove this class
            # These are residues where covered_count == 1
            Loss = np.sum(covered_count[old_indices] == 1)
            
            # Temporarily remove this class
            covered_count[old_indices] -= 1
            
            # Calculate gains for all residues with the modified covered_count
            matrix = covered_count.reshape(base // m, m)
            gains = np.sum(matrix == 0, axis=0)
            
            best_r_new = np.argmax(gains)
            best_gain = gains[best_r_new]
            
            if best_gain > Loss:
                # Perform swap
                new_indices = np.arange(best_r_new, base, m)
                covered_count[new_indices] += 1
                classes[i] = [m, best_r_new]
                improved = True
            else:
                # Revert
                covered_count[old_indices] += 1
                
        if not improved:
            break
            
    return classes
# EVOLVE-BLOCK-END
