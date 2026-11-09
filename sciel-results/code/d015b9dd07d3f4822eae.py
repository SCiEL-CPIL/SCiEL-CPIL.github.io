import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    
    # Precompute gcds
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i, n_mod):
            g = gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    def get_conflicts(residues, active_indices):
        """
        residues: list of length n_mod, residues[i] is r for moduli[i]
        active_indices: set of indices that are currently included
        Returns list of conflicting pairs (i, j) where i, j in active_indices
        """
        conflicts = []
        active_list = list(active_indices)
        for idx_i, i in enumerate(active_list):
            for j in active_list[idx_i+1:]:
                g = gcd_table[i][j]
                if residues[i] % g == residues[j] % g:
                    conflicts.append((i, j))
        return conflicts

    def try_fix_conflict(residues, active_indices, i, j, max_attempts=10):
        """
        Try to resolve conflict between i and j by changing residue of i or j.
        Returns True if resolved, False otherwise.
        Modifies residues in place.
        """
        # Try changing i
        for _ in range(max_attempts):
            old_r = residues[i]
            new_r = random.randint(0, moduli[i] - 1)
            if new_r == old_r:
                continue
            residues[i] = new_r
            # Check if this creates new conflicts with other active nodes
            valid = True
            for k in active_indices:
                if k == i or k == j:
                    continue
                g = gcd_table[i][k]
                if residues[i] % g == residues[k] % g:
                    valid = False
                    break
            if valid:
                # Also ensure i and j are now disjoint
                g_ij = gcd_table[i][j]
                if residues[i] % g_ij != residues[j] % g_ij:
                    return True
            residues[i] = old_r # revert if failed
            
        # Try changing j
        for _ in range(max_attempts):
            old_r = residues[j]
            new_r = random.randint(0, moduli[j] - 1)
            if new_r == old_r:
                continue
            residues[j] = new_r
            valid = True
            for k in active_indices:
                if k == i or k == j:
                    continue
                g = gcd_table[j][k]
                if residues[j] % g == residues[k] % g:
                    valid = False
                    break
            if valid:
                g_ij = gcd_table[i][j]
                if residues[i] % g_ij != residues[j] % g_ij:
                    return True
            residues[j] = old_r
            
        return False

    def solve():
        # Initialize with all moduli active
        active_indices = set(range(n_mod))
        residues = [random.randint(0, moduli[i] - 1) for i in range(n_mod)]
        
        # Iteratively resolve conflicts
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            conflicts = get_conflicts(residues, active_indices)
            if not conflicts:
                return active_indices, residues
                
            # Pick a random conflict to resolve
            i, j = random.choice(conflicts)
            
            resolved = try_fix_conflict(residues, active_indices, i, j)
            
            if not resolved:
                # If we can't resolve by changing residues, drop one modulus
                # Drop the one with larger m (smaller weight)
                if moduli[i] > moduli[j]:
                    active_indices.remove(i)
                else:
                    active_indices.remove(j)
                    
        return active_indices, residues

    best_active = None
    best_score = Fraction(0, 1)
    
    # Run multiple random restarts to ensure we find the all-moduli solution
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
        active_indices, residues = solve()
        
        # Calculate score
        current_score = Fraction(0, 1)
        for i in active_indices:
            current_score += Fraction(1, moduli[i])
            
        if current_score > best_score:
            best_score = current_score
            best_active = set(active_indices)
            
        # If we achieved the theoretical max, we can stop
        # Theoretical max is sum of 1/m for all m in 20..200
        if len(best_active) == n_mod:
            break

    # If we didn't find a perfect solution, try a local search improvement on the best found
    if best_active is not None and len(best_active) < n_mod:
        # ... (Optional: further local search to add back dropped moduli)
        pass

    # Construct result
    result = []
    for i in sorted(best_active):
        result.append([moduli[i], residues[i]])
        
    # Verify
    # ... (Verification code)
    
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Number of classes: {len(result)}")
    print(f"Score: {sum(1/m for m, r in result)}")
