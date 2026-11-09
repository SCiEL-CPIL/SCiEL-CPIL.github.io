# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
import random

# Global state to cache weights for a specific dimension n
# We use a fixed, non-uniform weighting scheme to break symmetry.
# The idea: Vectors with 0s in 'high weight' positions are preferred.
# Why? 0 is the additive identity. In F_3, a 3-term AP (a, b, c) satisfies a+c=2b.
# If b_i = 0, then a_i + c_i = 0 => a_i=c_i=0.
# If b_i = 1, then a_i + c_i = 2 => (0,2) or (2,0).
# If b_i = 2, then a_i + c_i = 1 => (0,1) or (1,0).
# Vectors with many 0s are "constrained" to have neighbors also have 0s in those positions,
# which might actually be good for packing if we prioritize consistency?
# Or, conversely, vectors with 0s are "safe" because they don't force the "opposite" 
# (0 vs 2) dynamic as strongly as 1s do?
# Let's try prioritizing 0s at specific positions.

_cap_weights = {}
_start_time = None

def _get_weights(n):
    """
    Generate a set of weights for positions 0..n-1.
    We use a deterministic but non-uniform weighting.
    To make it interesting and break symmetry, we use a pseudo-random but fixed sequence
    based on the dimension n, or a simple geometric/arithmetic progression.
    
    Let's use a geometric decay or a prime-based pattern to avoid obvious periodicities.
    We want weights to be positive.
    Higher weight for position i means: if vector[i] == 0, add weight[i] to score.
    """
    if n in _cap_weights:
        return _cap_weights[n]
    
    # Generate weights
    # Strategy: Use a deterministic hash-like distribution of weights.
    # We want to favor 0s. 
    # Let's try: weight[i] = 1.0 + 0.5 * sin(i * 2.399) # Golden angle related?
    # Or simpler: weight[i] = (i % 3) + 1 ? No, that's periodic.
    
    # Let's use a "random" but fixed seed based on n to generate weights in [0.5, 1.5].
    # This ensures that for any n, the preference for 0s varies by position.
    rng = random.Random(n * 7919) # Fixed seed per n
    weights = [rng.uniform(0.5, 1.5) for _ in range(n)]
    
    _cap_weights[n] = weights
    return weights

def priority(vector, n):
    global _start_time
    
    # Deadline guard
    if _start_time is None:
        _start_time = time.time()
    
    # If we are running out of time, we can fall back to a very fast heuristic
    # or just return a simple score. But priority is called many times.
    # The calculation below is O(n).
    
    weights = _get_weights(n)
    
    score = 0.0
    
    # Iterate through vector
    # We prioritize 0s.
    # We also need to handle 1s and 2s.
    # Hypothesis: 0s are "good". 1s and 2s are "neutral" or "bad" depending on balance?
    # Let's stick to the directive: "assigning higher weights to positions where the vector contains the digit 0".
    # So: score += weights[i] if v == 0 else 0.
    
    # However, if we ONLY reward 0s, we will pick the all-zeros vector first, then all vectors with one non-zero, etc.
    # This effectively creates a priority order based on "number of zeros" weighted by position.
    # This is structurally different from "balanced digits" which penalizes low 0s if n is large (since n/3 is target).
    # For n=8, target 0s is ~2.66. 
    # Balanced heuristic penalizes 0 zeros (deviation 2.66) and 8 zeros (deviation 5.33).
    # My new heuristic rewards 8 zeros highly (sum of all weights) and 0 zeros lowly (0).
    # So this will strongly prefer the {0}^n vector and vectors close to it.
    # Is {0}^n a good center for a cap set?
    # The set {0, 2}^n is a cap set. It contains the all-zeros vector.
    # If we prioritize 0s, we will likely fill up the {0, 2}^n subspace first?
    # No, because {0, 2}^n contains vectors with many 2s (and thus few 0s).
    # Example: (2,2,2,2,2,2,2,2) has 0 zeros. Score 0.
    # Vector (0,0,0,0,0,0,0,0) has 8 zeros. Score sum(weights).
    # Vector (1,1,1,1,1,1,1,1) has 0 zeros. Score 0.
    
    # So this heuristic will prioritize vectors with MANY 0s.
    # This effectively biases the search towards the "corner" of the cube {0,1,2}^n defined by 0s.
    
    # Let's implement this.
    for i, v in enumerate(vector):
        if v == 0:
            score += weights[i]
        # We could also penalize 1s and 2s?
        # The directive says "leveraging the fact that 0 is the additive identity... vectors with more zeros are structurally less likely to form... triples".
        # It doesn't explicitly say to penalize others, but to "assign higher weights... where... 0".
        # So we just add weight for 0s.
        
    return score

# EVOLVE-BLOCK-END
