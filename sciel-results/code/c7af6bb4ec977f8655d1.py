"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    base = 51975
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer before 120s limit
    
    # Generate divisors of 51975 greater than 1
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    # Precompute residue lists for each divisor
    # For each divisor m, precompute the indices for each residue class
    residue_indices = {}
    for m in divisors:
        residue_indices[m] = {}
        for r in range(m):
            # Indices: r, r+m, r+2m, ...
            residue_indices[m][r] = np.arange(r, base, m)
    
    def score_classes(classes):
        """Count uncovered residues."""
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            idx = residue_indices[m][r]
            covered[idx] = 1
        return base - covered.sum()
    
    # Greedy initial solution
    covered = np.zeros(base, dtype=np.uint8)
    classes = []
    for m in divisors:
        best_r = -1
        best_count = -1
        for r in range(m):
            idx = residue_indices[m][r]
            count = np.sum(1 - covered[idx])
            if count > best_count:
                best_count = count
                best_r = r
        classes.append([m, best_r])
        idx = residue_indices[m][best_r]
        covered[idx] = 1
    
    best_score = score_classes(classes)
    best_classes = [c[:] for c in classes]
    
    # Local search: try swapping residues or removing/adding classes
    # We'll do a simple hill-climbing with restarts
    while time.time() < deadline:
        # Try to improve current best
        improved = False
        classes = [c[:] for c in best_classes]
        covered = np.zeros(base, dtype=np.uint8)
        for m, r in classes:
            idx = residue_indices[m][r]
            covered[idx] = 1
        current_score = score_classes(classes)
        
        # Try changing residue for each class
        for i, (m, r) in enumerate(classes):
            for new_r in range(m):
                if new_r == r:
                    continue
                # Compute delta
                old_idx = residue_indices[m][r]
                new_idx = residue_indices[m][new_r]
                
                # Remove old, add new
                # Count how many new residues are currently uncovered
                new_uncovered = np.sum(1 - covered[new_idx])
                # Count how many old residues were only covered by this class
                # This is complex, so let's just recompute score
                classes[i] = [m, new_r]
                new_score = score_classes(classes)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_classes = [c[:] for c in classes]
                else:
                    classes[i] = [m, r]
        
        # If no improvement, try a random restart with perturbation
        if not improved:
            # Perturb: randomly change some residues
            classes = [c[:] for c in best_classes]
            for i in range(len(classes)):
                if random.random() < 0.3:
                    m, r = classes[i]
                    classes[i] = [m, random.randint(0, m-1)]
            new_score = score_classes(classes)
            if new_score < best_score:
                best_score = new_score
                best_classes = [c[:] for c in classes]
    
    return best_classes
# EVOLVE-BLOCK-END
