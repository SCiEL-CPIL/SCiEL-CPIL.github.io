import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    # Precompute spheres as lists of indices
    # For q=3, n=7, radius=1, sphere size is 1 + 7*(3-1) = 15
    
    def index_to_digits(idx):
        digits = [0] * n
        for k in range(n):
            digits[k] = idx % q
            idx //= q
        return digits

    def digits_to_index(digits):
        idx = 0
        for k in range(n - 1, -1, -1):
            idx = idx * q + digits[k]
        return idx

    # Precompute all digits
    all_digits = [index_to_digits(i) for i in range(total_words)]
    
    # Precompute spheres
    # sphere[i] is a list of indices j such that dist(i, j) <= radius
    spheres = []
    for i in range(total_words):
        base = all_digits[i]
        neighbors = [i]
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == base[pos]:
                        continue
                    new_digits = base[:]
                    new_digits[pos] = val
                    neighbors.append(digits_to_index(new_digits))
        # If radius > 1, we would need to extend this, but for this task radius=1 is typical.
        # The task says "radius" is passed. We should handle general radius if possible, 
        # but for 7 matches and q=3, radius 1 is the standard hard case.
        # If radius is larger, the sphere size grows. 
        # For now, let's assume radius is small (1 or 2).
        # To be safe, we can just compute distance for all j if radius is large, 
        # but for r=1 it's fast.
        
        # Let's implement a generic neighbor generator for small radius
        # If radius is 1, the above is correct.
        # If radius is 2, we need to add distance 2 neighbors.
        # Given the context "Football pool", it's almost certainly radius 1.
        # But to be robust:
        if radius > 1:
             # Fallback to distance check for larger radii, as sphere generation explodes
             # But N=2187 is small, so O(N^2) precomputation is 4.7M * 7 ops ~ 30M ops.
             # This is acceptable in Python if done once.
             spheres = []
             for i in range(total_words):
                 base = all_digits[i]
                 neighbors = []
                 for j in range(total_words):
                     dist = sum(1 for k in range(n) if base[k] != all_digits[j][k])
                     if dist <= radius:
                         neighbors.append(j)
                 spheres.append(neighbors)
             # Break out of the loop if we used the fallback
             # We need to restructure to avoid double computation
             pass
        else:
             spheres.append(neighbors)

    # If radius > 1, we already computed spheres in the loop above? 
    # No, the logic above is messy. Let's clean it up.
    
    # Reset spheres
    spheres = []
    if radius == 1:
        for i in range(total_words):
            base = all_digits[i]
            neighbors = [i]
            for pos in range(n):
                for val in range(q):
                    if val != base[pos]:
                        new_digits = base[:]
                        new_digits[pos] = val
                        neighbors.append(digits_to_index(new_digits))
            spheres.append(neighbors)
    else:
        # Generic O(N^2) precomputation
        for i in range(total_words):
            base = all_digits[i]
            neighbors = []
            for j in range(total_words):
                dist = 0
                for k in range(n):
                    if base[k] != all_digits[j][k]:
                        dist += 1
                        if dist > radius:
                            break
                if dist <= radius:
                    neighbors.append(j)
            spheres.append(neighbors)

    # Greedy Set Cover
    uncovered = np.ones(total_words, dtype=bool)
    codewords = []
    
    # To speed up, we can keep a list of active candidates
    # But simply scanning all is O(N^2 * R) which is fine for N=2187.
    
    for _ in range(total_words):
        if not uncovered.any():
            break
            
        best_idx = -1
        best_count = -1
        
        for i in range(total_words):
            # Count how many uncovered words are in sphere i
            # Using numpy sum on a boolean array slice is fast
            # But spheres[i] is a list of indices.
            # sum(uncovered[spheres[i]])
            # This creates a temporary array. 
            # Alternative: loop in Python? 
            # sum(1 for j in spheres[i] if uncovered[j])
            # Let's try numpy indexing.
            count = np.sum(uncovered[spheres[i]])
            
            if count > best_count:
                best_count = count
                best_idx = i
                
        if best_count == 0:
            break
            
        codewords.append(best_idx)
        uncovered[spheres[best_idx]] = False

    # Convert indices back to lists
    result = [all_digits[idx] for idx in codewords]
    return result
