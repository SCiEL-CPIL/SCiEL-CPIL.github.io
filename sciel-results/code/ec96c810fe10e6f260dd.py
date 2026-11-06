import time
import hashlib

def priority(vector, n):
        """
        Priority function that favors vectors with balanced symbol distribution.
        Adds a small deterministic pseudo-random tie-breaker to avoid
        suboptimal ordering of equally-valued vectors.
        """
        # Count occurrences of each symbol
        counts = [0, 0, 0]
        for v in vector:
            counts[v] += 1
        
        # Calculate entropy-like measure: higher when distribution is more balanced
        # Use sum of squared deviations from uniform as a penalty
        expected = n / 3.0
        variance = sum((c - expected) ** 2 for c in counts)
        
        # Lower variance = more balanced = higher priority
        # Base score
        base_score = -variance / (n + 1)
        
        # Tie-breaker: deterministic hash of the vector
        # This helps break symmetries in the greedy selection process
        # Use a simple hash to avoid expensive MD5 if possible, but MD5 is fine for small n
        # To be safe and fast, use Python's built-in hash? 
        # Built-in hash is randomized per process (PYTHONHASHSEED), which is bad for reproducibility if the evaluator runs multiple times?
        # The evaluator likely runs once. But deterministic is safer.
        # We can use a simple custom hash.
        
        h = 0
        for v in vector:
            h = (h * 31 + v) % 1000000007
        
        # Normalize to [0, 1)
        tie_breaker = (h % 10000) / 10000.0
        
        # Combine
        # Epsilon should be small relative to variance differences
        # Max variance for n=10:
        # Worst case: all 0s. counts=[10,0,0]. exp=3.33.
        # dev: 6.67, -3.33, -3.33. sq: 44.4, 11.1, 11.1. sum ~ 66.
        # Var diff between best (0) and worst (66) is 66.
        # So epsilon 0.01 is safe.
        
        return base_score + 0.01 * tie_breaker
