import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Iterative Simulated Greedy (Dynamic Degree-based)
    1. We simulate the greedy selection process using the Lowest Degree First (LDF) heuristic.
    2. In each iteration, we identify the vertex with the current minimum degree among remaining vertices.
    3. We assign a score to this vertex based on its "survival time" (iteration index) and its degree at removal.
       Specifically, vertices that survive longer (are picked later in the simulation) and have low degree at removal are considered "better" candidates for the final greedy decoder.
       Actually, for the final decoder (which picks highest score first), we want vertices that are "easy" to pick to have high scores.
       
       Let's refine the scoring logic:
       The final decoder picks vertices in descending score order.
       If we simulate LDF (picking min degree first), the vertices picked early in the simulation are "good" independent set candidates in the context of the LDF strategy.
       However, the final decoder is fixed to pick by our score.
       
       Alternative Insight:
       The "cost" of picking a vertex is roughly its degree.
       A vertex that remains in the graph for many iterations of LDF simulation is "robust".
       
       Let's try a different angle: "Independent Set Potential" via iterative peeling.
       We compute a score for each vertex $v$ as:
       Score(v) = - (Degree at removal in LDF simulation) + alpha * (Iteration index of removal)
       
       Why?
       - Low degree at removal: Good for MIS (fewer conflicts).
       - High iteration index: The vertex survived many rounds of peeling. This suggests it was not "blocked" by earlier, more central vertices. It is likely part of the "core" of the graph or a sparse region that persists.
       
       Wait, if the final decoder picks highest score first, and we want to mimic LDF (which picks low degree first), there is a tension.
       The final decoder is a *fixed* greedy: Pick highest score, remove neighbors.
       If we assign high scores to vertices that LDF would pick *late*, the final decoder will pick them *first*.
       Picking a "late" LDF vertex first might be bad if it has high degree in the original graph?
       
       Let's look at the current best: Static LDF + Hub Blocking.
       Static LDF: Score = -deg.
       This makes the final decoder pick the lowest degree vertices first. This is exactly the LDF heuristic.
       
       The improvement came from "Hub Blocking": adding a bonus for neighbors with high degree.
       
       My new strategy:
       Compute the "LDF Order" explicitly.
       Let $t_v$ be the iteration index when $v$ is removed in a full LDF simulation.
       Let $d_v^{rem}$ be the degree of $v$ at the moment of removal.
       
       If we set Score(v) = -d_v^{rem} + beta * t_v:
       The final decoder will pick vertices with low removal-degree and high removal-time first.
       
       Hypothesis: Vertices that are removed late in LDF are "peripheral" or in sparse clusters that weren't disrupted by hubs. They are "safe" picks.
       Vertices removed early are "hubs" or connected to hubs. They are "risky" (high conflict).
       
       So, high score -> Pick first.
       We want to pick "safe" vertices first.
       Safe = Low removal degree AND Late removal time?
       
       Let's test this logic.
       If a vertex is isolated, it is removed at some point with degree 0.
       If a vertex is a hub, it is removed early with high degree.
       
       If we prioritize late-removal, low-degree vertices, we are essentially prioritizing vertices that are "far" from the core hubs in the LDF peeling sense.
       
       Implementation:
       1. Simulate LDF peeling on the graph.
          - Keep a set of active vertices.
          - In each step, find min degree among active.
          - Record removal time and degree for that vertex.
          - Remove it and its neighbors from active.
       2. Construct score: Score[v] = -deg_at_removal[v] + beta * (n_iterations - removal_time[v])
          (Inverting time so that "late" = high value).
       
       Complexity: O(N * (N+E)) in worst case if naive. For N=200, this is 200 * (200+edges) ~ 200 * 2000 = 400,000 ops. Very fast.
       
       This is materially different from static calculation because it accounts for the *interaction* of removals (dynamic degree changes).
    """
    start_time = time.time()
    
    # Convert to numpy array
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    # Data structures for simulation
    # active: boolean array, True if vertex is still in the graph
    active = np.ones(n, dtype=bool)
    
    # deg: current degree of active vertices
    deg = adj.sum(axis=1).astype(float)
    
    # Records
    removal_degree = np.zeros(n)
    removal_time = np.zeros(n)
    
    # Iteration counter
    t = 0
    
    # We need an efficient way to find min degree among active vertices.
    # For N=200, a linear scan is O(N) per step, total O(N^2) = 40,000 ops.
    # Updating degrees is O(E) per step.
    # Total ~ N*E ~ 200 * 2000 = 400,000 ops. Safe.
    
    while np.any(active):
        # Find active vertex with minimum degree
        # Mask out inactive vertices with a large value
        current_deg = np.where(active, deg, np.inf)
        min_idx = np.argmin(current_deg)
        
        if not active[min_idx]:
            break # Should not happen if np.any(active) is true
            
        # Record
        removal_degree[min_idx] = deg[min_idx]
        removal_time[min_idx] = t
        
        # Remove min_idx and its neighbors
        # Neighbors are where adj[min_idx] is True
        neighbors = np.where(adj[min_idx])[0]
        
        # Update active status
        active[min_idx] = False
        active[neighbors] = False
        
        # Update degrees of remaining active vertices
        # For each neighbor of min_idx, if they are still active (they aren't, we just set them to False), 
        # wait, the neighbors are removed.
        # The degrees of the *other* active vertices need to be updated?
        # No, in LDF, when we remove u, we remove u and N(u).
        # The graph shrinks. The degrees of the *remaining* active vertices change 
        # because their neighbors (in N(u) and u) are gone.
        
        # Specifically, for any active vertex v, if v is connected to u or any neighbor of u, 
        # its degree decreases.
        
        # Efficient update:
        # The set of removed vertices in this step is R = {min_idx} U N(min_idx).
        # For each active vertex v, deg[v] -= number of neighbors v has in R.
        
        # Let's construct a boolean mask for R
        removed_mask = np.zeros(n, dtype=bool)
        removed_mask[min_idx] = True
        removed_mask[neighbors] = True
        
        # Calculate how many neighbors each active vertex has in R
        # This is a matrix-vector product: adj @ removed_mask
        # But we only care about active vertices.
        
        # adj is bool. removed_mask is bool.
        # count_removed_neighbors = adj @ removed_mask
        # This is O(N^2) per step. 200 * 200 = 40,000.
        # Total steps ~ N. Total O(N^3) = 8,000,000. Still very fast for N=200.
        
        counts = adj @ removed_mask
        deg -= counts
        
        # Ensure no negative degrees due to float precision or logic errors (should be exact integers)
        deg[deg < 0] = 0
        
        t += 1
        
        # Safety check
        if time.time() - start_time > 140:
            break
    
    # If we broke early due to time, we might have unassigned scores.
    # Fill remaining with a neutral score based on initial degree (fallback)
    unassigned = removal_time == 0 # This is risky as t=0 is valid. 
    # Better: track which ones were assigned.
    # Actually, the loop runs until active is empty. With N=200, it finishes in <1ms.
    # The time check is just a paranoia guard.
    
    # Construct Score
    # We want high score for:
    # 1. Low removal degree (easy to pick, few conflicts)
    # 2. High removal time (survived long, "robust", not a hub)
    
    # Normalize removal_time to [0, 1] roughly? Or just use it directly?
    # Max t is N. Max degree is N.
    # Let's balance them.
    
    # Score = -removal_degree + beta * removal_time
    # If beta is too high, we prioritize "peripheral" vertices over "central" ones?
    # LDF picks low degree first.
    # If we mimic LDF, we should pick vertices that LDF would pick *first*?
    # No, the final decoder picks *highest score* first.
    # If we want the final decoder to behave like LDF, we should give high scores to vertices that LDF picks *first* (low degree, early time).
    # Score = -removal_degree - gamma * removal_time?
    
    # Let's rethink.
    # LDF is a good heuristic for MIS.
    # LDF picks v with min deg.
    # If we assign Score(v) = -deg_original(v), the final decoder picks min deg first. This is LDF.
    # Current best adds: + beta * avg_neighbor_deg.
    # This boosts vertices next to hubs.
    
    # My dynamic score:
    # If I use Score = -removal_degree, I am using the degree *at the time of removal* in the LDF process.
    # For a vertex picked first, removal_degree == original_degree.
    # For a vertex picked later, removal_degree < original_degree (because neighbors were removed).
    # So -removal_degree is *higher* (less negative) for later vertices than -original_degree.
    # This effectively *penalizes* early picks (hubs) and *rewards* late picks (peripherals)?
    # Wait.
    # Hub: orig deg 100, rem deg 100. Score = -100.
    # Peripheral: orig deg 10, rem deg 2 (9 neighbors removed). Score = -2.
    # Final decoder picks Peripheral (Score -2) before Hub (Score -100).
    # This is OPPOSITE to LDF! LDF would pick Peripheral (deg 10) before Hub (deg 100)?
    # No, LDF picks min degree. 10 < 100. So LDF picks Peripheral first.
    # My score: Peripheral (-2) > Hub (-100). Final decoder picks Peripheral first.
    # So Score = -removal_degree *mimics* the relative ordering of LDF picks?
    # Not exactly.
    # In LDF, the Peripheral is picked at step k. The Hub is picked at step m.
    # If k < m, Peripheral is picked first.
    # In my score, Peripheral has higher score.
    # So the final decoder picks Peripheral first.
    # This aligns with LDF's *choice* of which vertex to pick next?
    # Actually, LDF picks the *global* min degree at each step.
    # My score is static (calculated once from the simulation).
    # The final decoder picks *global* max score at each step.
    # If my scores rank the vertices in the same order as the LDF removal order, then the final decoder will pick them in the same order as LDF.
    
    # Does -removal_degree rank them in LDF order?
    # LDF picks min current degree.
    # Vertex A picked at t=0 with deg 5. Score -5.
    # Vertex B picked at t=1 with deg 2 (was 10, lost 8 neighbors). Score -2.
    # LDF picked A then B.
    # Final decoder picks B (-2) then A (-5).
    # Order is REVERSED.
    
    # So to mimic LDF order, we should use Score = removal_degree?
    # Then B (2) < A (5). Final decoder picks A first.
    # This matches LDF order!
    
    # So, Score = removal_degree might be a good base.
    # But wait, we want to pick "good" vertices.
    # LDF is a good heuristic. So mimicking LDF order is a good strategy.
    # Therefore, Score = removal_degree (higher degree at removal = higher score = picked earlier) seems to reverse the "goodness" if we think "low degree is good".
    # But if the *order* is what matters, and LDF order is good, then we want the final decoder to pick in LDF order.
    # LDF order: A (deg 5) then B (deg 2).
    # To make Final Decoder pick A then B, Score(A) > Score(B).
    # Score(A) = 5, Score(B) = 2. 5 > 2. Correct.
    
    # So, Score = removal_degree mimics LDF selection order.
    
    # Can we improve on LDF?
    # LDF is greedy. It can be suboptimal.
    # The "Hub Blocking" term in the current best suggests that picking vertices *near* hubs is good.
    # In LDF, hubs are picked late (high degree). Vertices near hubs are picked earlier?
    # If a vertex is near a hub, its degree is high. It might be picked earlier than a vertex in a sparse cluster.
    
    # Let's try combining:
    # Score = removal_degree + alpha * (1 / (removal_time + 1))
    # Or simply:
    # Score = removal_degree
    # And see if adding a "hub proximity" bonus helps.
    # Hub proximity: sum of degrees of neighbors in original graph?
    # That's what the current program does.
    
    # Let's try:
    # Score = removal_degree + beta * (original_neighbor_degree_sum / (original_degree + 1))
    
    # This combines the dynamic "LDF order" signal (via removal_degree) with the static "hub blocking" signal.
    
    # Let's refine the "LDF Order" score.
    # If we just use removal_degree, we are essentially saying "Pick vertices that had high degree when they were removed by LDF".
    # This is equivalent to "Pick vertices that were 'busy' until late in the process".
    
    # Let's implement this hybrid.
    
    # Calculate static hub term
    orig_deg = adj.sum(axis=1)
    orig_neighbor_deg_sum = adj @ orig_deg
    safe_orig_deg = orig_deg + 1.0
    hub_term = orig_neighbor_deg_sum / safe_orig_deg
    
    # Dynamic term
    # We want to mimic LDF order.
    # LDF picks min current degree.
    # We established Score = removal_degree mimics LDF order.
    
    # Let's use:
    # Score = removal_degree + beta * hub_term
    
    # What is beta?
    # In current program, beta=0.5 worked well.
    # removal_degree scale: 0 to N.
    # hub_term scale: 0 to N.
    # Similar scales.
    
    # Let's try beta = 0.5.
    
    beta = 0.5
    score = removal_degree + beta * hub_term
    
    # One more consideration:
    # If a vertex is isolated, removal_degree = 0. hub_term = 0. Score = 0.
    # If a vertex is a hub, removal_degree = high. hub_term = high. Score = high.
    # Final decoder picks Hub first.
    # Is picking a Hub first good for MIS?
    # Usually NO. You want to pick low degree vertices to maximize count.
    # Picking a Hub blocks many vertices.
    # LDF picks Low Degree first.
    # My score: Hub (high) > Low Degree (low).
    # Final decoder picks Hub first.
    # This is OPPOSITE to LDF!
    
    # I made a mistake in the logic deduction.
    # LDF picks A (deg 5) then B (deg 2).
    # A is "higher" degree than B.
    # If I want Final Decoder to pick A then B, I need Score(A) > Score(B).
    # Score = removal_degree: 5 > 2. Correct.
    # So Final Decoder picks A (deg 5) first.
    # LDF picks A (deg 5) first?
    # LDF picks MIN degree.
    # If A has deg 5 and B has deg 2, LDF picks B (deg 2) first!
    # My previous example: "Vertex A picked at t=0 with deg 5. Vertex B picked at t=1 with deg 2".
    # This implies A was picked FIRST.
    # But LDF picks MIN degree. If B had deg 2, B should have been picked first, unless B wasn't active or had higher degree initially?
    # Ah, B's degree *decreased* to 2.
    # So initially, B might have had degree 10. A had degree 5.
    # LDF picks A (5 < 10).
    # Then A is removed. B's neighbors are removed. B's degree drops to 2.
    # Next step, B is picked.
    # So LDF Order: A then B.
    # My Score: A (rem deg 5), B (rem deg 2).
    # Score(A) = 5, Score(B) = 2.
    # Final Decoder picks A (5) then B (2).
    # This MATCHES LDF order.
    
    # So, Score = removal_degree DOES mimic LDF order.
    # And LDF is a good heuristic (picks low degree initially).
    # So this should be a good strategy.
    
    # The "Hub Blocking" term:
    # Current program: Score = -deg + 0.5 * hub_term.
    # This picks LOW degree first (negative sign).
    # My program: Score = rem_deg + 0.5 * hub_term.
    # This picks HIGH removal degree first.
    # Since removal_degree correlates with initial degree (roughly), this picks HIGH initial degree first?
    # No, removal_degree is the degree *at removal*.
    # For the first pick, rem_deg = orig_deg.
    # So for the first pick, it behaves like "Pick High Degree".
    # This is OPPOSITE to LDF (Pick Low Degree).
    
    # Wait.
    # If A (deg 5) is picked first by LDF, it means 5 was the MIN degree.
    # So all other vertices had degree >= 5.
    # So A is "low" relative to the graph.
    # My score for A is 5.
    # Other vertices have scores >= 5 (since their orig deg >= 5, and rem deg <= orig deg? No, rem deg can be lower).
    # If another vertex C had orig deg 10, and is picked later with rem deg 2.
    # Score(C) = 2.
    # Score(A) = 5.
    # Final Decoder picks A (5) then C (2).
    # This matches LDF.
    
    # So, does it pick "Low Degree" or "High Degree" first?
    # It picks the vertex with the HIGHEST removal degree first.
    # In the LDF process, the vertex picked first has the MINIMUM original degree.
    # Does it have the MAXIMUM removal degree?
    # Yes, because it is picked first, so its removal degree IS its original degree.
    # Any vertex picked later has its neighbors removed, so its removal degree is LESS than or equal to its original degree.
    # Since the first vertex had the MIN original degree, and others had HIGHER original degrees, the first vertex's removal degree (min orig) might be LOWER than the later vertices' original degrees, but we compare REMOVAL degrees.
    # Later vertices have reduced removal degrees.
    # So it is plausible that the first vertex (min orig) has a HIGHER removal degree than later vertices (reduced orig).
    # Thus, ranking by removal_degree descends roughly matches LDF order.
    
    # Conclusion: Score = removal_degree + beta * hub_term is a valid strategy to mimic LDF with hub awareness.
    
    return score

# Fallback for safety
def priority(adjacency):
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    active = np.ones(n, dtype=bool)
    deg = adj.sum(axis=1).astype(float)
    
    removal_degree = np.zeros(n)
    removal_time = np.zeros(n)
    
    t = 0
    while np.any(active):
        current_deg = np.where(active, deg, np.inf)
        min_idx = np.argmin(current_deg)
        
        if not active[min_idx]:
            break
            
        removal_degree[min_idx] = deg[min_idx]
        removal_time[min_idx] = t
        
        neighbors = np.where(adj[min_idx])[0]
        
        removed_mask = np.zeros(n, dtype=bool)
        removed_mask[min_idx] = True
        removed_mask[neighbors] = True
        
        counts = adj @ removed_mask
        deg -= counts
        deg[deg < 0] = 0
        
        active[min_idx] = False
        active[neighbors] = False
        
        t += 1
        
        if time.time() - start_time > 140:
            break
    
    orig_deg = adj.sum(axis=1)
    orig_neighbor_deg_sum = adj @ orig_deg
    safe_orig_deg = orig_deg + 1.0
    hub_term = orig_neighbor_deg_sum / safe_orig_deg
    
    beta = 0.5
    score = removal_degree + beta * hub_term
    
    return score
