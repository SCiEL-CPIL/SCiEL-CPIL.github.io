import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Direct score-vector optimization via iterative blocker-penalty reduction.
    - Instead of permuting a fixed ordering, we maintain a continuous score vector.
    - We identify "blockers": vertices with high scores whose selection prevents many 
      other high-potential vertices from being selected.
    - We iteratively reduce the score of the most damaging blockers and increase the 
      score of their blocked neighbors, effectively pushing the score distribution 
      to separate competing clusters.
    - This is a structural change: it modifies the relative distances between scores 
      to create "gaps" that allow the greedy decoder to skip problematic neighbors.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Precompute neighbor lists and degrees
    neighbors_list = [np.where(adj[i])[0].tolist() for i in range(n)]
    deg = adj.sum(axis=1).astype(int)
    
    def greedy_is_size(scores):
        """Compute the size of the independent set produced by the greedy decoder."""
        order = np.argsort(-scores, kind='stable')
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v] and not blocked[v]:
                selected[v] = True
                count += 1
                for u in neighbors_list[v]:
                    if not selected[u]:
                        blocked[u] = True
        return count, order, selected, blocked

    # Initial scores: based on inverse degree (low degree = high score)
    # Add small random noise to break ties
    rng = np.random.default_rng(42)
    scores = -deg.astype(float) + rng.random(n) * 0.1
    scores -= scores.min() # Shift to positive
    
    current_size, _, _, _ = greedy_is_size(scores)
    best_scores = scores.copy()
    best_size = current_size

    # Iterative refinement: Blocker Penalty Reduction
    # We simulate the greedy process and identify which vertices, when selected, 
    # block the most "valuable" (high score) unselected neighbors.
    
    # To speed up, we use a simplified metric: 
    # Penalty(v) = sum of scores of neighbors u of v that are blocked by v 
    #             and have higher score than v's "alternative" (or just high score).
    # We reduce score of v if Penalty(v) is high.
    
    max_iterations = 5000
    step_size = 1.0
    
    for _ in range(max_iterations):
        if time.time() - start_time > deadline:
            break
        
        # Evaluate current state
        current_size, order, selected, blocked = greedy_is_size(scores)
        
        if current_size > best_size:
            best_size = current_size
            best_scores = scores.copy()
        
        # Identify the "critical" step in the greedy process where a selection 
        # causes a large drop in potential future selections.
        # We approximate this by looking at the selected vertices and their blocked neighbors.
        
        # Calculate a "damage" score for each selected vertex:
        # How many high-score vertices did it block?
        damage = np.zeros(n)
        for v in order:
            if selected[v]:
                # Sum of scores of neighbors that are blocked by v
                # Note: A neighbor might be blocked by multiple selected vertices, 
                # but in the greedy process, the first one to block it is the "culprit".
                # However, for simplicity, we attribute the block to the selected vertex 
                # with the highest score among those that blocked it? 
                # Actually, in the greedy loop, `blocked[u]` becomes True when the 
                # FIRST selected neighbor is processed.
                # So we need to know which selected vertex blocked each blocked vertex.
                pass
        
        # More efficient approach: 
        # Re-simulate and track the "blocker" for each blocked vertex.
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        blocker = np.full(n, -1, dtype=int)
        count = 0
        for v in order:
            if not selected[v] and not blocked[v]:
                selected[v] = True
                count += 1
                for u in neighbors_list[v]:
                    if not selected[u] and not blocked[u]:
                        blocked[u] = True
                        blocker[u] = v
        
        # Now, for each selected vertex v, calculate the sum of scores of vertices it blocked.
        damage = np.zeros(n)
        for u in range(n):
            if blocked[u] and blocker[u] != -1:
                v = blocker[u]
                damage[v] += scores[u]
        
        # Find the vertex with the highest damage that is currently selected.
        # If we lower its score slightly, it might not be selected, or it might be 
        # selected later, allowing its neighbors to be selected instead.
        # We pick the selected vertex with the highest damage.
        selected_indices = np.where(selected)[0]
        if len(selected_indices) == 0:
            break
            
        damages_of_selected = damage[selected_indices]
        if damages_of_selected.max() <= 0:
            # No significant damage, try random perturbation or break
            break
        
        worst_v = selected_indices[np.argmax(damages_of_selected)]
        
        # Reduce the score of the worst blocker
        # We reduce it enough to potentially swap its order with one of its blocked neighbors
        # or to make it "less attractive" so the greedy decoder might pick a neighbor instead.
        
        # Heuristic: Reduce score by a small amount, and increase score of blocked neighbors slightly?
        # Or just reduce the blocker.
        
        # Let's reduce the blocker's score by a factor of the damage.
        # If damage is high, reduce more.
        reduction = step_size * (1.0 + damage[worst_v] / (scores.max() + 1e-9))
        scores[worst_v] -= reduction
        
        # To prevent scores from going to -infinity, we can normalize or just let them be.
        # The greedy decoder only cares about relative order.
        
        # Optional: Increase scores of the blocked neighbors to make them more likely to be picked
        # if the blocker is delayed.
        blocked_neighbors = [u for u in neighbors_list[worst_v] if blocked[u]]
        for u in blocked_neighbors:
            scores[u] += reduction * 0.1

        # Normalize scores to keep them in a reasonable range to avoid float issues
        # scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        
        # If no improvement in last 100 iterations, restart with perturbation
        # (Simplified: just continue, the damage metric should guide us)

    # Final evaluation
    final_size, _, _, _ = greedy_is_size(best_scores)
    if final_size < best_size:
        # Should not happen, but just in case
        pass

    # Return the best scores found
    return best_scores
