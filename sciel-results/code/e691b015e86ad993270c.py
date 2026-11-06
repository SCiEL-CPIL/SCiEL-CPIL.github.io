import random
import time

def construct():
    n = 16
    num_edges = n * (n - 1) // 2
    # Precompute edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    
    # Map edge index to (u, v)
    # Map (u, v) to edge index
    edge_index = {}
    for idx, (u, v) in enumerate(edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
        
    # Precompute K4s
    k4s = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    k4s.append((a, b, c, d))
                    
    num_k4s = len(k4s)
    
    # For each K4, list of edge indices
    k4_edges = []
    for k4 in k4s:
        a, b, c, d = k4
        e1 = edge_index[(a, b)]
        e2 = edge_index[(a, c)]
        e3 = edge_index[(a, d)]
        e4 = edge_index[(b, c)]
        e5 = edge_index[(b, d)]
        e6 = edge_index[(c, d)]
        k4_edges.append([e1, e2, e3, e4, e5, e6])
        
    # For each edge, list of K4 indices containing it
    edge_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_k4s[e_idx].append(k4_idx)
            
    # We try k = 13, 14, 15
    # Time limit 120s. Allocate ~30s per k.
    
    start_time = time.time()
    
    for k in [13, 14, 15]:
        # Initialize random coloring
        coloring = [random.randint(0, k - 1) for _ in range(num_edges)]
        
        # Initialize counts and masks
        # counts[k4_idx][color]
        counts = [[0] * k for _ in range(num_k4s)]
        masks = [0] * num_k4s
        bad_count = 0
        
        for k4_idx in range(num_k4s):
            for e_idx in k4_edges[k4_idx]:
                c = coloring[e_idx]
                counts[k4_idx][c] += 1
                masks[k4_idx] |= (1 << c)
            if bin(masks[k4_idx]).count('1') < 5:
                bad_count += 1
                
        if bad_count == 0:
            return coloring
            
        # Local Search
        # We use a simple hill climbing / random restarts
        # Or simulated annealing
        # Given the small size, random restarts with hill climbing might work.
        # Let's try a few restarts within the time budget for this k.
        
        max_time_per_k = 30
        restarts = 0
        while time.time() - start_time < 120 - (15 - k) * 5: # Roughly allocate time
            # Actually, let's just run for a fixed number of steps or time slice
            # Let's do 5 seconds per restart attempt? No, that's too slow.
            # Let's do 10000 steps per restart.
            
            current_bad = bad_count
            current_coloring = coloring[:]
            current_counts = [c[:] for c in counts]
            current_masks = masks[:]
            
            # Hill Climb
            for step in range(10000):
                if current_bad == 0:
                    break
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                old_c = current_coloring[e_idx]
                # Try a new color
                new_c = random.randint(0, k - 1)
                if new_c == old_c:
                    continue
                    
                # Calculate delta in bad_count
                delta_bad = 0
                # We need to update counts and masks for affected K4s
                # To do this efficiently, we can just simulate the change
                # and count how many K4s become bad/good.
                
                # Since we need to revert, we can store the changes
                # Or just compute the new bad count for affected K4s
                
                # Let's just compute the new state for affected K4s
                # and see if total bad count decreases.
                
                # To avoid complex revert logic, we can just apply, check, and revert if worse.
                # But applying requires updating counts/masks.
                
                # Let's try a different approach:
                # Compute the "cost" of changing edge e_idx from old_c to new_c.
                # Cost = (new_bad_count - old_bad_count)
                
                # For each K4 containing e_idx:
                #   old_mask = current_masks[k4_idx]
                #   old_counts = current_counts[k4_idx]
                
                #   new_counts = old_counts[:]
                #   new_counts[old_c] -= 1
                #   new_counts[new_c] += 1
                
                #   new_mask = old_mask
                #   if new_counts[old_c] == 0:
                #       new_mask &= ~(1 << old_c)
                #   new_mask |= (1 << new_c)
                
                #   old_bad = 1 if bin(old_mask).count('1') < 5 else 0
                #   new_bad = 1 if bin(new_mask).count('1') < 5 else 0
                #   delta_bad += new_bad - old_bad
                
                # If delta_bad < 0, accept.
                # If delta_bad == 0, accept with some probability (SA) or always (Hill).
                # If delta_bad > 0, reject (Hill) or accept with prob exp(-delta/T) (SA).
                
                # Let's use Simulated Annealing.
                T = 1.0
                cooling = 0.999
                
                # Calculate delta
                delta_bad = 0
                affected_k4s = edge_k4s[e_idx]
                
                for k4_idx in affected_k4s:
                    old_mask = current_masks[k4_idx]
                    old_counts = current_counts[k4_idx]
                    
                    new_counts = old_counts[:]
                    new_counts[old_c] -= 1
                    new_counts[new_c] += 1
                    
                    new_mask = old_mask
                    if new_counts[old_c] == 0:
                        new_mask &= ~(1 << old_c)
                    new_mask |= (1 << new_c)
                    
                    old_bad = 1 if bin(old_mask).count('1') < 5 else 0
                    new_bad = 1 if bin(new_mask).count('1') < 5 else 0
                    delta_bad += new_bad - old_bad
                    
                if delta_bad < 0:
                    # Accept
                    current_coloring[e_idx] = new_c
                    for k4_idx in affected_k4s:
                        current_counts[k4_idx][old_c] -= 1
                        if current_counts[k4_idx][old_c] == 0:
                            current_masks[k4_idx] &= ~(1 << old_c)
                        current_counts[k4_idx][new_c] += 1
                        current_masks[k4_idx] |= (1 << new_c)
                    current_bad += delta_bad
                else:
                    # Accept with probability exp(-delta_bad / T)
                    if random.random() < pow(2.71828, -delta_bad / T):
                        current_coloring[e_idx] = new_c
                        for k4_idx in affected_k4s:
                            current_counts[k4_idx][old_c] -= 1
                            if current_counts[k4_idx][old_c] == 0:
                                current_masks[k4_idx] &= ~(1 << old_c)
                            current_counts[k4_idx][new_c] += 1
                            current_masks[k4_idx] |= (1 << new_c)
                        current_bad += delta_bad
                        
                T *= cooling
                
            if current_bad == 0:
                return current_coloring
                
            # If we didn't find a solution, restart with new random coloring
            # Or keep the best found?
            # Let's restart.
            
            # Update time check
            if time.time() - start_time > 110:
                break
                
    # If no solution found with 13, 14, 15, fall back to 17 (seed)
    # Or return the best one found?
    # The task says "Minimize the number of colors used".
    # If we can't find 13, 14, 15, we should return the seed (17) to be safe.
    # But maybe we can return a valid 15-coloring if we found one?
    # The loop above returns immediately if found.
    # If it finishes without returning, it means no solution found in time.
    
    # Fallback: Seed program
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
