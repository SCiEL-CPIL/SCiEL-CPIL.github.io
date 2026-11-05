"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy: Simulate a parallel discrete-time exclusion process.
    In each round, vertices with the fewest active (unselected, not-blocked) neighbors
    are selected. This captures dynamic interference. We run multiple randomized
    simulations to get a robust priority ordering based on the average "time-to-selection"
    or rank in the selection order.
    
    The greedy decoder picks vertices in descending score order. So we want vertices
    that are "good" to pick early to have high scores. In our simulation, vertices
    selected early in the exclusion process are likely good candidates.
    
    We use the inverse of the average selection rank (time step) as the score,
    so earlier selection => higher score.
    """
    n = adjacency.shape[0]
    # Convert to boolean array if needed, and ensure it's efficient
    adj = np.asarray(adjacency, dtype=bool)
    
    # If graph is empty or trivial
    if n == 0:
        return np.array([])
    
    # Precompute degrees for potential tie-breaking or initialization
    degrees = adj.sum(axis=1).astype(np.int32)
    
    # Number of simulations. 
    # Constraint: 120s wall clock.
    # n=200. Each simulation step involves finding min of active neighbors.
    # With numpy, we can vectorize the "active neighbors count" update somewhat,
    # but the sequential selection of the "best" vertex in a round makes it tricky.
    # However, the problem says "vertices with the fewest active neighbors are selected first in each round".
    # This implies a batch selection per round? Or one by one?
    # "parallel discrete-time exclusion process where vertices with the fewest active neighbors are selected first in each round"
    # Usually, in parallel MIS algorithms (like Luby's), you select a maximal independent set from a subset.
    # But here the description is specific: "selected first in each round".
    # Let's interpret: In each round, we look at all currently available vertices.
    # We calculate for each available vertex v, the number of available neighbors.
    # We select the vertex (or vertices?) with the minimum count.
    # If we select one vertex per round, it's a greedy algorithm with dynamic degree.
    # If we select all tied minima, it's parallel.
    # The prompt says "vertices ... are selected first", plural.
    # Let's assume we select ALL vertices that have the global minimum active-neighbor count in that round.
    # Then we remove them and their neighbors from the available set.
    
    # To get a priority score for the *fixed* greedy decoder:
    # The decoder picks one vertex at a time, descending score.
    # If we simulate the parallel process, we get a set of vertices selected in Round 1, Round 2, etc.
    # Vertices in Round 1 should have the highest priority.
    # Vertices in Round 2 next, etc.
    # To break ties within a round, we can use randomization or secondary metrics.
    
    # Let's run K simulations. In each simulation, we break ties randomly.
    # We record the "round" in which each vertex is selected.
    # Priority score = - (average round index). Lower round index => higher priority.
    # So score = - mean_round.
    
    # Optimization:
    # n=200 is small. K can be reasonably large, e.g., 50-100.
    # Each simulation:
    #   Available mask: bool array of size n.
    #   Loop until no available vertices:
    #     Calculate active neighbor count for each available vertex.
    #     min_count = min(active_neighbor_counts[available])
    #     candidates = available & (active_neighbor_counts == min_count)
    #     Select candidates. To make it deterministic for the *priority* function but robust,
    #     we actually want the priority to reflect the "typical" order.
    #     If we select all candidates at once, they all get the same round number.
    #     But the greedy decoder picks them one by one.
    #     If two vertices are in the same round, their relative order in the greedy decoder
    #     might matter if they are adjacent? No, if they are both selected in the same round
    #     of the parallel process, they are NOT adjacent (since selecting one blocks the other
    #     in the next step, but in the same step they are both "chosen" as a set? 
    #     Wait, if v and u are adjacent, and both have min count, can they both be selected in the same round?
    #     If v is selected, u is blocked. So they cannot both be in the final independent set.
    #     The parallel algorithm usually picks a maximal independent set from the candidates.
    #     If we just pick all min-count vertices, they might be adjacent.
    #     The prompt says: "skipping neighbours of picked vertices".
    #     The simulation is a model to generate scores.
    #     If the simulation picks adjacent vertices in the same "round", that's invalid for an independent set.
    #     So, within a round, we must pick an independent set from the candidates.
    #     Simplest: Pick the candidate with the lowest ID? Or random?
    #     Let's use a randomized greedy within the candidate set to resolve conflicts,
    #     assigning them the same "round" value but slightly different sub-scores?
    #     Or just assign the round number, and let the secondary tie-breaker in the final score handle it.
    
    # Revised Plan:
    # 1. Run K simulations.
    # 2. In each simulation:
    #    - Maintain `available` mask.
    #    - While any available:
    #       - Compute `deg_avail` = number of available neighbors for each available vertex.
    #       - `min_deg` = min(deg_avail[available])
    #       - `candidates` = indices where available AND deg_avail == min_deg.
    #       - We need to select an independent set from `candidates`.
    #       - To simulate "randomized tie-breaking", we can shuffle candidates and greedily pick non-adjacent ones.
    #       - Assign current `round` to all picked vertices.
    #       - Remove picked vertices and their neighbors from `available`.
    #       - Increment `round`.
    # 3. For each vertex, average its `round` value over K simulations.
    # 4. Return `-avg_round` as the score.
    
    # Complexity:
    # Each round reduces the available set.
    # Computing `deg_avail` is O(n^2) if done naively, or O(nnz) with sparse.
    # n=200, n^2 = 40,000.
    # Number of rounds is at most n (worst case 1 vertex per round), usually much less.
    # Let's say 50 rounds. 50 * 40,000 = 2,000,000 ops per simulation.
    # K=100 simulations => 200,000,000 ops.
    # Python loop overhead might be high.
    # We can vectorize the deg_avail calculation.
    # `deg_avail = available @ adj`? No, `adj` is symmetric.
    # `neighbor_counts = adj.dot(available.astype(float))` gives count of available neighbors for ALL vertices.
    # Then filter by `available`.
    
    K = 50  # Number of simulations
    start_time = time.time()
    TIME_LIMIT = 100.0  # Safety margin
    
    # Store sum of rounds for each vertex
    round_sums = np.zeros(n, dtype=np.float64)
    
    # Pre-allocate some arrays
    # adj is boolean.
    
    for sim in range(K):
        # Check time
        if time.time() - start_time > TIME_LIMIT:
            break
            
        available = np.ones(n, dtype=bool)
        current_round = 0
        
        while np.any(available):
            # Calculate number of available neighbors for each vertex
            # adj[i] is row i. adj[i] @ available gives count of available neighbors for i.
            # We can do this for all i at once: adj.dot(available)
            # adj is bool, available is bool. Result is int.
            neighbor_counts = adj.dot(available)
            
            # Only consider available vertices
            # Find min neighbor count among available vertices
            # Mask out unavailable vertices by setting their count to inf
            # Or just take min of neighbor_counts[available]
            avail_counts = neighbor_counts[available]
            min_count = np.min(avail_counts)
            
            # Candidates are available vertices with neighbor_count == min_count
            # We need to select an independent set from these candidates.
            # Let's get the indices of candidates
            # candidates_mask = available & (neighbor_counts == min_count)
            candidates_mask = np.zeros(n, dtype=bool)
            candidates_mask[available] = (neighbor_counts[available] == min_count)
            
            cand_indices = np.where(candidates_mask)[0]
            
            if len(cand_indices) == 0:
                break # Should not happen
            
            # Greedily select an independent set from cand_indices
            # Shuffle to introduce randomness in tie-breaking
            np.random.shuffle(cand_indices)
            
            selected_in_round = []
            # We need to check if a candidate is adjacent to any already selected in this round
            # Since cand_indices are all tied in min degree, they might be adjacent to each other.
            # We iterate through shuffled candidates.
            # Keep track of which vertices are "blocked" within this round's selection
            # Actually, if we pick v, we block its neighbors.
            # But we only care about blocking other *candidates* in this round?
            # No, if we pick v, v is in the independent set. Its neighbors cannot be in the independent set.
            # So if u is a neighbor of v, and u is also a candidate, u cannot be selected in this round.
            
            # Efficient check:
            # For each candidate c in shuffled order:
            #   if c is not blocked by previously selected candidates in this round:
            #      select c
            #      mark c's neighbors as blocked (for the purpose of this round's selection)
            
            # Since n is small, we can just keep a set or mask.
            blocked_in_round = np.zeros(n, dtype=bool)
            
            for c in cand_indices:
                if not blocked_in_round[c]:
                    selected_in_round.append(c)
                    # Block neighbors of c
                    # blocked_in_round[c] = True (c is selected, so it's "blocked" from being selected again, obviously)
                    # Also block neighbors
                    blocked_in_round[adj[c]] = True
                    # Note: adj[c] includes c itself if self-loops? No, MIS usually simple graphs.
                    # Assume no self-loops. adj[c] are neighbors.
            
            # Assign round to selected vertices
            round_sums[selected_in_round] += current_round
            
            # Update available set
            # Remove selected vertices and their neighbors
            # selected_mask = np.zeros(n, dtype=bool)
            # selected_mask[selected_in_round] = True
            # available[selected_mask] = False
            # available[adj.dot(selected_mask.astype(bool))] = False  # This marks neighbors as unavailable
            
            # More efficient:
            # Create a mask of vertices to remove: selected + neighbors of selected
            remove_mask = np.zeros(n, dtype=bool)
            # Use advanced indexing or boolean ops
            # Since selected_in_round is a list of indices:
            remove_mask[selected_in_round] = True
            # Add neighbors
            # We can do: remove_mask[adj[selected_in_round].any(axis=0)] = True
            # adj[selected_in_round] is a (k, n) bool matrix.
            # .any(axis=0) gives (n,) bool matrix of vertices that are neighbors of ANY selected vertex.
            if len(selected_in_round) > 0:
                remove_mask[adj[selected_in_round].any(axis=0)] = True
            
            available[remove_mask] = False
            current_round += 1
    
    # The score should be such that lower round sum => higher priority.
    # So priority = - round_sums
    # Add a small random noise or use degree as tie breaker?
    # The greedy decoder sorts by score. If scores are equal, order is arbitrary (usually stable sort by index).
    # To make it robust, we can add a tiny bit of noise based on degree or just return -round_sums.
    # Let's return -round_sums.
    
    # One issue: If a vertex is never selected in a simulation (shouldn't happen as we pick until all available are gone),
    # its sum remains 0, which is the highest priority. But in a connected graph, all vertices are eventually selected or blocked.
    # Wait, if a vertex is blocked, it is NOT selected.
    # The simulation assigns round numbers ONLY to selected vertices.
    # What about vertices that are blocked (neighbors of selected vertices)?
    # They are removed from `available` but never added to `round_sums`.
    # Their `round_sums` entry remains 0 (or whatever it was from previous sims, which is 0 initially).
    # This is a problem. Blocked vertices should have LOWER priority than selected vertices?
    # Or should they be assigned a round number corresponding to when they were blocked?
    # The goal is to find a good order for the *greedy decoder*.
    # The greedy decoder picks v, then skips neighbors.
    # If v is a "hub", picking it early is bad because it blocks many others.
    # If v is a "leaf", picking it early is good.
    # In our simulation, leaves tend to have low degree, so they are selected early (low round).
    # Hubs have high degree, so they are selected late (high round) or never (if blocked by neighbors).
    # If a hub is blocked, it was blocked by a neighbor selected in round R.
    # We should assign the hub a round number > R, or perhaps R + 1?
    # Actually, if a vertex is blocked in round R, it means a neighbor was selected in round R.
    # For the greedy decoder, if we pick the neighbor first, the hub is skipped.
    # So the hub's "effective" selection time is "after" the neighbor.
    # Let's assign blocked vertices the round number of the round they were blocked, plus a small epsilon?
    # Or just the round number.
    # If we assign the same round number to a selected vertex and a blocked neighbor,
    # the tie-breaking in the greedy decoder might pick the blocked one first, which is bad.
    # So we should assign blocked vertices a HIGHER round number (lower priority).
    # Let's track when a vertex is removed from `available` and whether it was selected or blocked.
    
    # Let's refine the simulation to assign scores to ALL vertices.
    
    # Restart with refined logic
    round_sums = np.zeros(n, dtype=np.float64)
    # We need to be careful with time.
    
    for sim in range(K):
        if time.time() - start_time > TIME_LIMIT:
            break
            
        available = np.ones(n, dtype=bool)
        current_round = 0
        # To handle blocked vertices, we can update round_sums when they are removed.
        
        while np.any(available):
            neighbor_counts = adj.dot(available)
            avail_counts = neighbor_counts[available]
            min_count = np.min(avail_counts)
            
            candidates_mask = np.zeros(n, dtype=bool)
            candidates_mask[available] = (neighbor_counts[available] == min_count)
            cand_indices = np.where(candidates_mask)[0]
            
            if len(cand_indices) == 0:
                break
            
            np.random.shuffle(cand_indices)
            
            selected_in_round = []
            blocked_in_round = np.zeros(n, dtype=bool)
            
            for c in cand_indices:
                if not blocked_in_round[c]:
                    selected_in_round.append(c)
                    blocked_in_round[adj[c]] = True
                    blocked_in_round[c] = True # Mark self as blocked to prevent re-selection
            
            # Assign round to selected
            if len(selected_in_round) > 0:
                round_sums[selected_in_round] += current_round
                
            # Determine which vertices are removed from available
            # Selected vertices are removed.
            # Neighbors of selected vertices are removed (blocked).
            remove_mask = np.zeros(n, dtype=bool)
            if len(selected_in_round) > 0:
                remove_mask[selected_in_round] = True
                # Neighbors
                nbrs = adj[selected_in_round].any(axis=0)
                remove_mask[nbrs] = True
            else:
                # This shouldn't happen if cand_indices is not empty
                break
            
            # Identify blocked vertices (removed but not selected)
            # blocked_vertices = remove_mask & ~selected_mask
            selected_mask = np.zeros(n, dtype=bool)
            if len(selected_in_round) > 0:
                selected_mask[selected_in_round] = True
            
            blocked_vertices_mask = remove_mask & ~selected_mask
            
            # Assign round to blocked vertices?
            # They are effectively "selected" at this round in terms of being processed,
            # but they are not in the independent set.
            # For priority, we want them to have lower priority than selected vertices.
            # So assign current_round + 0.5? Or just current_round?
            # If we assign current_round, they tie with selected.
            # Random tie-breaking in the decoder might pick a blocked vertex, which is suboptimal
            # because it doesn't add to the set, but it doesn't hurt if it's skipped?
            # No, the decoder *picks* vertices. If it picks a vertex that is a neighbor of a previously picked vertex,
            # the problem statement says "skipping neighbours of picked vertices".
            # So the decoder logic is:
            # for v in sorted(vertices, key=score, reverse=True):
            #    if not any(u in picked for u in neighbors(v)):
            #       picked.add(v)
            # So if the decoder picks a "blocked" vertex (in our simulation sense),
            # it will be skipped if its neighbor (the selected one) was picked earlier.
            # If the blocked vertex is picked *before* the selected neighbor, it will be picked,
            # and then the selected neighbor will be skipped.
            # This is bad if the selected neighbor was a "good" choice and the blocked one was "bad".
            # But in our simulation, the blocked one was blocked BY the selected one, implying the selected one
            # was chosen in the parallel process.
            # To ensure the decoder picks the "selected" ones first, we should give them strictly higher scores
            # (lower round values) than the blocked ones.
            # So: selected get current_round, blocked get current_round + 0.1 (or similar).
            
            if np.any(blocked_vertices_mask):
                # Get indices of blocked vertices
                blocked_indices = np.where(blocked_vertices_mask)[0]
                round_sums[blocked_indices] += current_round + 0.1
                
            available[remove_mask] = False
            current_round += 1
    
    # Normalize? No, just return negative sums.
    # Vertices with lower average round sum have higher priority.
    scores = -round_sums
    
    # Add a small deterministic tie-breaker: lower degree is slightly better?
    # Or just leave it. The randomization in K simulations should break ties naturally.
    # However, if two vertices have identical round sums, the decoder's sort is stable,
    # so they will be ordered by index.
    # To make it more robust, we can add a tiny bit of noise based on degree.
    # Lower degree => higher score (add small positive value).
    # scores += (max_degree - degrees) * 1e-6
    
    return scores

# EVOLVE-BLOCK-END
