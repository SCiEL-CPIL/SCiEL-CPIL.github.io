"""Cap set greedy priority: prioritize vectors with high coordinate diversity."""
# EVOLVE-BLOCK-START
import time

def priority(vector, n):
    # Prioritize vectors with high coordinate spread (variance)
    # Vectors with diverse coordinates form fewer 3-APs
    # Use negative sum of squared deviations from mean as a "diversity" score
    # Higher priority = more diverse = better for cap set
    
    # Convert to list for processing
    v = list(vector)
    
    # Compute mean
    mean = sum(v) / n if n > 0 else 0
    
    # Compute variance (sum of squared deviations)
    variance = sum((x - mean) ** 2 for x in v)
    
    # Higher variance = more diverse = higher priority
    return variance
# EVOLVE-BLOCK-END
