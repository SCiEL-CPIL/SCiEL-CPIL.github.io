"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Computes a priority score for each vertex for the greedy independent set decoder.
    
    Heuristic:
    1. Isolated vertices (degree 0) are ideal candidates (score 0, highest).
    2. For other vertices, we want to minimize the "cost" of picking them.
       The cost is the sum of the "values" of the neighbors we block.
       A neighbor v is more "valuable" if it has a low degree (easier to pick independently).
       We approximate Value(v) as 1 / (degree(v) + 1).
       So, Score(u) = - sum( Value(v) for v in neighbors(u) ).
    3. We also add a small penalty for the degree of u itself to break ties
       in favor of lower-degree vertices when neighbor structures are similar.
    """
    # Ensure adjacency is a numpy array of floats for matrix multiplication
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    # Compute degrees
    degrees = adjacency.sum(axis=1)
    
    # Compute the "value" of each vertex based on its degree.
    # Lower degree -> Higher value (more desirable to keep in the independent set).
    # Using 1/(d+1) to avoid division by zero and to scale down high-degree vertices.
    vertex_values = 1.0 / (degrees + 1.0)
    
    # Calculate the sum of neighbor values for each vertex u.
    # This represents the "cost" of picking u (the value of neighbors we lose).
    # adjacency is N x N. vertex_values is N x 1.
    # Result is N x 1.
    neighbor_value_sum = adjacency @ vertex_values
    
    # Base score is negative cost.
    scores = -neighbor_value_sum
    
    # Add a small penalty for high degree to prefer lower-degree vertices
    # if the neighbor value sums are close.
    # The factor 0.1 is a heuristic weight.
    scores -= 0.1 * degrees
    
    # Isolated vertices have degree 0.
    # neighbor_value_sum = 0.
    # scores = 0 - 0 = 0.
    # Other vertices have negative scores.
    # So isolated vertices are correctly ranked highest.
    
    return scores
# EVOLVE-BLOCK-END
