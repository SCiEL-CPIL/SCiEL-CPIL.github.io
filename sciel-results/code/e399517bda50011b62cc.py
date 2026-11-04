import numpy as np
import time

def priority(adjacency):
    # Deadline guard
    start_time = time.time()
    
    # Convert to float for matrix operations
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Safety check early
    if time.time() - start_time > 110:
        return -degrees
    
    # Weight of a vertex: how "valuable" is it to keep?
    # Low degree -> high value.
    # w(u) = 1 / (degree(u) + 1)
    weights = 1.0 / (degrees + 1.0)
    
    # Sum of neighbor values (weighted)
    neighbor_value_sum = A @ weights
    
    # Sum of degrees of neighbors (unweighted)
    sum_neighbor_deg = A @ degrees
    
    # Combine strategies:
    # Current program: -(degrees + neighbor_value_sum)
    # Inspiration 2: sum_neighbor_deg - degrees
    # 
    # Let's try a combination that emphasizes both:
    # We want to pick vertices that have low degree (low cost) 
    # but whose neighbors have high degree (high opportunity cost for others).
    # 
    # Score = sum_neighbor_deg - degrees - alpha * neighbor_value_sum
    # where alpha balances the two terms.
    # 
    # Actually, let's think about it differently:
    # The current program's score is: -(degrees + neighbor_value_sum)
    # which means we prefer low degree AND low neighbor value sum.
    # But neighbor_value_sum is high when neighbors have low degree (since w=1/(d+1)).
    # So low neighbor_value_sum means neighbors have HIGH degree.
    # That's consistent with inspiration 2!
    #
    # Let's try: score = sum_neighbor_deg - degrees - neighbor_value_sum
    # This combines the unweighted neighbor degree sum with the weighted one.
    
    scores = sum_neighbor_deg - degrees - neighbor_value_sum
    
    # Safety check
    if time.time() - start_time > 110:
        return -degrees
    
    return scores
