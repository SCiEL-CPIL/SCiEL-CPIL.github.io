import time
import math

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # Distance squared on triangular lattice: (a,b) represents a + b*omega, omega=exp(i*pi/3)
    def dist_sq(p1, p2):
        da = p1[0] - p2[0]
        db = p1[1] - p2[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Hex distance from origin
    def hex_dist(p):
        a, b = p
        return (abs(a) + abs(b) + abs(a+b)) // 2

    # Generate candidate pool: Radius 4 hexagon (61 points)
    # This is a superset of the radius 3 hexagon (37 points).
    # The optimal 32 points are likely contained within radius 3, but radius 4 provides more flexibility for local search.
    all_points = []
    for a in range(-4, 5):
        for b in range(-4, 5):
            if hex_dist([a, b]) <= 4:
                all_points.append([a, b])
    
    n_candidates = len(all_points)
    
    # Map point to index for fast lookup
    point_to_idx = {tuple(p): i for i, p in enumerate(all_points)}
    
    # Precompute neighbors (distance squared == 1) for each candidate
    # neighbors[i] is a set of indices j such that dist_sq(all_points[i], all_points[j]) == 1
    neighbors = [set() for _ in range(n_candidates)]
    for i in range(n_candidates):
        for j in range(i+1, n_candidates):
            if dist_sq(all_points[i], all_points[j]) == 1:
                neighbors[i].add(j)
                neighbors[j].add(i)

    # Initial configuration: Greedy center-out
    # Sort candidates by hex distance, then angle for determinism
    def angle(p):
        a, b = p
        x = a + b/2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
        
    candidates_sorted = sorted(range(n_candidates), key=lambda i: (hex_dist(all_points[i]), angle(all_points[i])))
    
    # Select first 32 points
    initial_selected = set(candidates_sorted[:32])
    current_selected = list(initial_selected)
    current_edges = count_edges([all_points[i] for i in current_selected])
    
    # Local Search with Delta Calculation
    # Maintain degrees: deg[i] = number of neighbors of i in current_selected
    deg = [0] * n_candidates
    for i in current_selected:
        for n in neighbors[i]:
            if n in initial_selected:
                deg[i] += 1
    # Note: The above loop is slightly inefficient. Let's do it properly.
    deg = [0] * n_candidates
    selected_set = set(current_selected)
    for i in current_selected:
        for n in neighbors[i]:
            if n in selected_set:
                deg[i] += 1
                
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Iterate over all selected points and all unselected points
        # To speed up, we can randomize order or just iterate
        # Since n=61, 32*29 ~ 900 swaps per pass. Very fast.
        
        # We need to be careful with the delta calculation update.
        # We will find the best swap in one pass and apply it, or apply any improving swap immediately.
        # Applying immediately is standard hill climbing.
        
        for i in current_selected:
            if time.time() > deadline - 5.0:
                break
            for j in range(n_candidates):
                if j in selected_set:
                    continue
                
                # Calculate delta
                # deg[i] is current degree of i in selected_set
                # deg[j] is current degree of j in selected_set (j is not selected, so this is 0? No, deg[j] is calculated based on selected_set. Since j is not in selected_set, its "degree" in the context of the set is not maintained in the same way, but we can calculate it on the fly or maintain it for all nodes).
                # Actually, `deg` array is only meaningful for selected nodes?
                # No, `deg[k]` = number of neighbors of k in `selected_set`.
                # So `deg[j]` is the number of neighbors of j in `selected_set`.
                
                # Is i and j adjacent?
                is_adj = 1 if j in neighbors[i] else 0
                
                delta = (deg[j] - is_adj) - deg[i]
                
                if delta > 0:
                    # Perform swap
                    # Remove i
                    selected_set.remove(i)
                    # Add j
                    selected_set.add(j)
                    
                    # Update deg for all affected nodes
                    # Nodes affected: i, j, and neighbors of i and j
                    
                    # For i (removed):
                    # deg[i] becomes 0 (since i is no longer in set, but deg[i] is defined as neighbors in set. Since i is not in set, it's not used for future delta calculations for i? 
                    # Wait, if i is not selected, we don't care about deg[i] for swaps FROM i. 
                    # But if i is unselected, and we swap k with i later, we need deg[i].
                    # So we must maintain deg for ALL nodes.
                    
                    # Update deg for neighbors of i
                    for n in neighbors[i]:
                        if n in selected_set:
                            deg[n] -= 1 # i was a neighbor, now it's not
                    deg[i] = 0 # i is not in set, so its degree in the set is 0? 
                    # Actually, deg[i] is the number of neighbors of i in selected_set.
                    # Since i is removed, it is not in selected_set.
                    # But for future swaps where i is the "unselected" candidate, we need to know how many neighbors i has in the NEW selected_set.
                    # The new selected_set is S' = S \ {i} U {j}.
                    # Neighbors of i in S' are (neighbors of i in S) \ {i} (i not in S) \ {j if j in N(i)}?
                    # No, neighbors of i in S' are simply the neighbors of i that are in S'.
                    # S' = S - {i} + {j}.
                    # So N(i) \cap S' = (N(i) \cap S) - {i} (irrelevant) - {j if j in N(i)}? No.
                    # If j is in N(i), then j is in S'. So i and j are adjacent in S'.
                    # But deg[i] is the count of neighbors of i in S'.
                    # Previously, deg[i] counted neighbors in S.
                    # The only change for i's neighbors is that i is removed and j is added.
                    # If j is a neighbor of i, then i gains a neighbor (j) in S'? No, i is NOT in S'.
                    # deg[i] is defined for ANY node i as |N(i) \cap S|.
                    # So if i is not in S, deg[i] is still |N(i) \cap S|.
                    # When we remove i from S, S changes.
                    # When we add j to S, S changes.
                    
                    # Let's re-calculate deg for all nodes involved.
                    # Affected nodes: i, j, and N(i) U N(j).
                    
                    # This is getting complicated to update incrementally correctly.
                    # Given N=61, we can just recalculate deg for all nodes after each swap?
                    # 61 * (avg deg ~ 6) = 366 operations.
                    # 900 swaps * 366 = 300k operations per pass.
                    # This is very fast in Python (microseconds).
                    
                    # Let's just recalculate deg for all nodes to be safe and simple.
                    # It's O(N * D).
                    for k in range(n_candidates):
                        count = 0
                        for n in neighbors[k]:
                            if n in selected_set:
                                count += 1
                        deg[k] = count
                    
                    # Update current_selected list? 
                    # We use selected_set for membership. 
                    # We iterate over current_selected. We should update it.
                    # But current_selected is a list. 
                    # Let's just use selected_set for the loop? 
                    # The loop `for i in current_selected` iterates over the old list.
                    # We should probably break and restart the loop or update the list.
                    # Simplest: break and restart the while loop iteration.
                    
                    current_edges += delta
                    improved = True
                    # Update current_selected to reflect the change for the next iteration of the inner loop?
                    # Or just break out to the outer while loop?
                    # Breaking out of the inner loop and continuing the outer while loop is safer.
                    break
            if improved:
                break
        
        # If we broke due to improvement, we restart the pass.
        # If we finished the pass without improvement, improved remains False.
        
    # Final result
    result = [all_points[i] for i in current_selected]
    
    # Verify
    assert len(result) == 32
    assert len(set(tuple(p) for p in result)) == 32
    for p in result:
        assert abs(p[0]) <= 64 and abs(p[1]) <= 64
        
    return result
