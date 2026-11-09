import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    def check_validity(n, orientation):
        # Build adjacency masks
        masks = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                bit = orientation[idx]
                if bit == 0:
                    masks[i] |= (1 << j)
                else:
                    masks[j] |= (1 << i)
                idx += 1
        
        # Check all 4-subsets with early termination
        for i in range(n):
            for j in range(i+1, n):
                m_ij = masks[i] & masks[j]
                if m_ij == 0:
                    continue
                for k in range(j+1, n):
                    m_ijk = m_ij & masks[k]
                    if m_ijk == 0:
                        continue
                    for l in range(k+1, n):
                        if (m_ijk & masks[l]) == 0:
                            return False
        return True

    def get_paley_orientation(p):
        squares = {x * x % p for x in range(1, p)}
        return [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]

    def count_violations(n, orientation):
        """Count the number of 4-sets without a common dominator."""
        masks = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                bit = orientation[idx]
                if bit == 0:
                    masks[i] |= (1 << j)
                else:
                    masks[j] |= (1 << i)
                idx += 1
        
        violations = 0
        for i in range(n):
            for j in range(i+1, n):
                m_ij = masks[i] & masks[j]
                if m_ij == 0:
                    # All 4-sets containing i,j are violations
                    remaining = n - j - 1
                    violations += remaining * (remaining - 1) // 2
                    continue
                for k in range(j+1, n):
                    m_ijk = m_ij & masks[k]
                    if m_ijk == 0:
                        remaining = n - k - 1
                        violations += remaining
                        continue
                    for l in range(k+1, n):
                        if (m_ijk & masks[l]) == 0:
                            violations += 1
        return violations

    def get_edge_index(n, i, j):
        """Get the index in the orientation list for edge (i,j) with i<j."""
        if i > j:
            i, j = j, i
        # Index = sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
        # = i*(n-1) - i*(i-1)//2 + (j-i-1)
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    # First, try Paley 67
    p = 67
    if p % 4 == 3:
        orient = get_paley_orientation(p)
        if check_validity(p, orient):
            return {'n': p, 'orientation': orient}
    
    # Try local search from random or Paley seeds
    n = 67
    best_violations = float('inf')
    best_orient = None
    
    # Try several random seeds
    for seed_attempt in range(20):
        if time.time() > deadline:
            break
        
        if seed_attempt == 0:
            # Start from Paley
            orient = get_paley_orientation(n)
        else:
            # Random start
            orient = [random.randint(0, 1) for _ in range(n * (n-1) // 2)]
        
        # Local search: flip edges to reduce violations
        current_violations = count_violations(n, orient)
        if current_violations == 0:
            return {'n': n, 'orientation': orient}
        if current_violations < best_violations:
            best_violations = current_violations
            best_orient = orient[:]
        
        # Hill climbing with restarts
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try flipping random edges
            num_flips_to_try = min(500, n * (n-1) // 2)
            for _ in range(num_flips_to_try):
                if time.time() > deadline:
                    break
                edge_idx = random.randint(0, n * (n-1) // 2 - 1)
                old_val = orient[edge_idx]
                orient[edge_idx] = 1 - old_val
                
                new_violations = count_violations(n, orient)
                if new_violations < current_violations:
                    current_violations = new_violations
                    improved = True
                    if new_violations == 0:
                        return {'n': n, 'orientation': orient}
                    if new_violations < best_violations:
                        best_violations = new_violations
                        best_orient = orient[:]
                else:
                    orient[edge_idx] = old_val
    
    # If we found a valid solution, return it
    if best_orient is not None and best_violations == 0:
        return {'n': n, 'orientation': best_orient}
    
    # Fallback: return the best we found (even if invalid, but this shouldn't happen)
    # Actually, we need to return a valid tournament. Let's use Paley 83 as fallback
    p = 83
    orient = get_paley_orientation(p)
    return {'n': p, 'orientation': orient}
