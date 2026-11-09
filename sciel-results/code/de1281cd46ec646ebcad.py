import numpy as np

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = False
    return np.where(sieve)[0]

def construct():
    primes = _primes(N)
    best_size = 0
    best_mask = np.zeros(N + 1, dtype=bool)
    
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    
    # Perform multiple restarts with random tie-breaking
    # 20 restarts should be fast enough with NumPy
    for _ in range(20):
        mask = np.ones(N + 1, dtype=bool)
        current_size = N + 1
        
        for p in primes:
            if p > current_size:
                break
            
            # Get active indices
            active_indices = np.where(mask)[0]
            
            # Count residues
            residues = active_indices % p
            counts = np.bincount(residues, minlength=p)
            
            # Check if any residue class is already empty
            if counts.min() == 0:
                continue
            
            # Find residue classes with the minimum count
            min_count = counts.min()
            candidates = np.where(counts == min_count)[0]
            
            # Randomly select one of the minimum count residue classes to remove
            target_residue = np.random.choice(candidates)
            
            # Identify elements to remove
            indices_to_remove = active_indices[residues == target_residue]
            
            # Remove them from the mask
            mask[indices_to_remove] = False
            current_size -= len(indices_to_remove)
            
        # Update best solution
        if current_size > best_size:
            best_size = current_size
            best_mask = mask.copy()
            
    return list(np.where(best_mask)[0])
# EVOLVE-BLOCK-END
