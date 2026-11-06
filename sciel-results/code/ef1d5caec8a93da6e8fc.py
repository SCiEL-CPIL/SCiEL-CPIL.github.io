import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe
    
    n = 128
    # Generate all 4-APs
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for start in range(1, n - 3 * d + 1):
            # 0-indexed positions
            aps.append((start - 1, start + d - 1, start + 2 * d - 1, start + 3 * d - 1))
    
    num_aps = len(aps)
    
    # Map position to list of AP indices
    pos_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(i)
            
    best_solution = None
    best_k = 128
    
    # Try increasing number of colors
    for k in range(3, 11):
        if time.time() > deadline - 5:
            break
            
        found = False
        # Number of restarts depends on k
        # k=3 is hardest, try more? Or k=5 is easier, try less?
        # Let's try 20 restarts for each k.
        max_restarts = 30
        
        for r in range(max_restarts):
            if time.time() > deadline - 2:
                break
                
            # Random initialization
            assignment = [random.randint(0, k - 1) for _ in range(n)]
            
            # Calculate initial violations
            # ap_state[ap_idx] = number of distinct colors? Or just violation flag?
            # We need to update incrementally.
            # Let's maintain ap_color_counts[ap_idx][color]
            # Since k is small, we can use a list of size k for each AP.
            ap_color_counts = [[0] * k for _ in range(num_aps)]
            
            # Initialize counts
            for i, ap in enumerate(aps):
                for pos in ap:
                    ap_color_counts[i][assignment[pos]] += 1
                    
            # Calculate initial violations
            ap_violations = [0] * num_aps
            total_violations = 0
            for i in range(num_aps):
                distinct = sum(1 for c in ap_color_counts[i] if c > 0)
                if distinct < 3:
                    ap_violations[i] = 1
                    total_violations += 1
                else:
                    ap_violations[i] = 0
                    
            if total_violations == 0:
                best_solution = assignment
                best_k = k
                found = True
                break
                
            # Local Search
            # Hill climbing with random restarts
            for step in range(50000):
                if time.time() > deadline - 2:
                    break
                    
                if total_violations == 0:
                    best_solution = assignment
                    best_k = k
                    found = True
                    break
                    
                # Pick random position
                p = random.randint(0, n - 1)
                old_color = assignment[p]
                new_color = random.randint(0, k - 1)
                
                if new_color == old_color:
                    continue
                
                # Calculate delta
                delta_v = 0
                affected_aps = pos_to_aps[p]
                
                # We need to check each affected AP
                # For each AP, remove old_color, add new_color, check distinct count
                # This is O(k) per AP if we recount, or O(1) if we track distinct count.
                # Let's track distinct count per AP.
                
                # We don't have distinct count array. Let's add it.
                # ap_distinct[ap_idx]
                
                # To avoid re-initializing ap_distinct every restart, we can just recalculate it?
                # No, we need it for updates.
                # Let's maintain ap_distinct.
                
                # Wait, I didn't define ap_distinct in the loop above.
                # Let's add it.
                
                # Actually, recalculating distinct count for 4 elements is very fast.
                # set(assignment[pos] for pos in ap)
                # Let's just do that for simplicity and correctness, accepting the O(4) cost.
                
                # But we need to update total_violations.
                # So we need to know the OLD violation status and NEW violation status.
                
                # Store old status
                old_statuses = {}
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    # Count distinct
                    colors_in_ap = set(assignment[pos] for pos in ap)
                    old_statuses[ap_idx] = 1 if len(colors_in_ap) < 3 else 0
                    
                # Apply change
                assignment[p] = new_color
                
                # Calculate new status and delta
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    colors_in_ap = set(assignment[pos] for pos in ap)
                    new_status = 1 if len(colors_in_ap) < 3 else 0
                    delta_v += new_status - old_statuses[ap_idx]
                    
                total_violations += delta_v
                
                # Acceptance
                if delta_v < 0:
                    continue # Keep
                elif delta_v == 0:
                    if random.random() < 0.5:
                        continue # Keep
                    else:
                        # Revert
                        assignment[p] = old_color
                        total_violations -= delta_v
                else:
                    # Revert
                    assignment[p] = old_color
                    total_violations -= delta_v
                    
            if found:
                break
                
        if found:
            break
            
    if best_solution is not None:
        return best_solution
        
    # Fallback: Return a valid solution with many colors
    # Use 128 colors, each position unique
    return list(range(n))

