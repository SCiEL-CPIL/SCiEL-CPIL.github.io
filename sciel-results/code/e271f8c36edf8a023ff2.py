"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a score to each vertex for greedy Maximum Independent Set selection.
    
    The greedy decoder picks vertices in descending score order, skipping neighbours
    of already-picked vertices. We want to maximize the resulting independent set size.
    
    Strategy: Use a combination of graph-theoretic heuristics to score vertices.
    A good heuristic for greedy MIS is to prefer vertices that:
    1. Have low degree (less likely to block many others)
    2. Have neighbours with high degree (removing such a vertex blocks high-degree neighbours,
       which is good because those high-degree neighbours would block many others anyway)
    3. Are not "central" in the graph
    
    We'll compute several features and combine them into a single score.
    
    Key insight: For Erdos-Renyi graphs with n=200, the average degree depends on p.
    A robust approach is to use local structural properties.
    
    Let's use the following score:
    - Primary: negative degree (lower degree = higher priority)
    - Secondary: negative sum of neighbour degrees (if neighbours are high-degree, 
      picking this vertex is more valuable because it "covers" high-degree nodes)
    
    Actually, let me think more carefully. In greedy MIS:
    - When we pick a vertex v, we add v to the independent set and remove v and all its neighbours.
    - We want to pick vertices that are "cheap" in terms of how many potential future picks they eliminate.
    - A vertex with low degree eliminates fewer neighbours, so it's "cheaper".
    - But also, if a vertex's neighbours are high-degree, those neighbours would have eliminated many more vertices if they were picked. So picking a low-degree vertex whose neighbours are high-degree is doubly good.
    
    Let's define:
    score(v) = -degree(v) + alpha * (average degree of neighbours of v)
    
    Wait, that's not quite right. Let me reconsider.
    
    Actually, a well-known heuristic for greedy MIS is to pick the vertex with minimum degree. This is the "Min-Degree" heuristic. The seed program already does this with -degree.
    
    To improve, we can use a more sophisticated scoring. Let's try:
    score(v) = -degree(v) + beta * (sum of degrees of non-neighbours that are adjacent to v's neighbours)
    
    Hmm, that's getting complex. Let me think of a simpler improvement.
    
    Alternative: Use the concept of "independent set potential". For each vertex, estimate how many vertices could potentially be in an independent set if we pick this vertex.
    
    A simpler approach: Use a weighted combination of:
    1. -degree(v): lower degree is better
    2. -number_of_common_neighbours_with_high_degree: 
    3. Centrality measures: prefer peripheral vertices
    
    Let me try a different angle. The key tradeoff in greedy MIS is:
    - Picking a low-degree vertex is good because it blocks few others.
    - But if all low-degree vertices are clustered together, picking one might not be as good as picking a slightly higher-degree vertex in a different part of the graph.
    
    For Erdos-Renyi graphs, the structure is relatively homogeneous, so the min-degree heuristic should work reasonably well. To improve, we can add a tie-breaking or secondary criterion.
    
    Let's try: score(v) = -degree(v) + epsilon * (something that breaks ties favorably)
    
    Actually, let's use a more principled approach. Consider the following:
    - For each vertex v, compute the "cost" of picking v as the number of vertices that would be removed from consideration (v itself + all neighbours).
    - This is simply degree(v) + 1.
    - We want to minimize this cost, so score = -(degree(v) + 1) = -degree(v) - 1.
    - This is essentially the same as the seed program (up to a constant).
    
    To get a real improvement, we need to look beyond just degree. Let's consider:
    - The "coverage" of a vertex: how many high-degree vertices does it neighbour?
    - If v is adjacent to many high-degree vertices, picking v is more valuable because those high-degree vertices would have had a large "coverage" themselves.
    
    Let's define:
    score(v) = -degree(v) + lambda * (mean degree of neighbours of v)
    
    where lambda is a small positive constant. This rewards vertices that neighbour high-degree vertices.
    
    Let's implement this and see.
    
    Actually, I realize we should be more careful. Let's think about what makes a good greedy MIS heuristic:
    
    1. Min-degree: Pick the vertex with the smallest degree. This is a classic heuristic.
    2. Max-neighborhood: Pick the vertex whose neighbourhood is "most wasteful" if left unpicked.
    3. A combination.
    
    Let's try a score that combines:
    - Low degree (good)
    - Neighbours with high degree (good, because picking v "handles" those high-degree neighbours)
    
    score(v) = -degree(v) + 0.1 * (sum of degrees of neighbours of v) / max(1, degree(v))
    
    The second term is the average degree of v's neighbours. If v's neighbours are high-degree, this term is large, increasing v's score.
    
    Let's implement this.
    """
    start_time = time.time()
    
    # Convert to numpy array if needed
    adj = np.asarray(adjacency)
    n = adj.shape[0]
    
    # Compute degrees
    degrees = adj.sum(axis=1).astype(float)
    
    # Compute sum of neighbour degrees for each vertex
    # For vertex v, sum of degrees of its neighbours = (adj @ degrees)[v]
    neigh_deg_sums = adj @ degrees
    
    # Average degree of neighbours (avoid division by zero)
    neigh_deg_avgs = np.where(degrees > 0, neigh_deg_sums / np.maximum(degrees, 1), 0)
    
    # Score: negative degree plus a small bonus for having high-degree neighbours
    # The bonus term encourages picking vertices that are adjacent to high-degree vertices
    # because those high-degree vertices would otherwise block many potential picks
    score = -degrees + 0.1 * neigh_deg_avgs
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 120:
        # Return a simple fallback
        return -degrees
    
    return score.astype(float)
# EVOLVE-BLOCK-END
