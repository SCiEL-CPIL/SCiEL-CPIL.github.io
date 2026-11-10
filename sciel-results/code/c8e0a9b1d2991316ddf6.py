# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import random


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.
    
    Strategy: Use ±1 vectors. For two ±1 vectors, distance^2 = 4 * (Hamming distance).
    Norm^2 = 11 for all such vectors. We need 4 * Hamming_dist >= 11, i.e., Hamming_dist >= 3.
    So we need a binary code of length 11 with minimum Hamming distance >= 3.
    We use a greedy construction with multiple random orderings to find a large such code.
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    
    # Generate all possible ±1 vectors (2^11 = 2048 candidates)
    candidates = [tuple(c) for c in product([-1, 1], repeat=d)]
    
    def greedy_build(order):
        """Build a code using the given order of candidates."""
        selected = []
        
        for vec_tuple in order:
            # Check against all selected vectors
            compatible = True
            for sel_vec in selected:
                # Compute Hamming distance efficiently
                hamming_dist = 0
                for i in range(d):
                    if vec_tuple[i] != sel_vec[i]:
                        hamming_dist += 1
                        if hamming_dist >= 3:
                            break
                if hamming_dist < 3:
                    compatible = False
                    break
            
            if compatible:
                selected.append(vec_tuple)
        
        return selected
    
    # Use deterministic seed for reproducibility
    rng = random.Random(42)
    
    best_code = []
    
    # Try natural order first
    code = greedy_build(candidates)
    best_code = code
    
    # Try several random orderings
    for _ in range(50):
        shuffled = candidates.copy()
        rng.shuffle(shuffled)
        code = greedy_build(shuffled)
        if len(code) > len(best_code):
            best_code = code
    
    # Convert to numpy array
    points = np.array(best_code, dtype=np.int64)
    
    return points


# EVOLVE-BLOCK-END
