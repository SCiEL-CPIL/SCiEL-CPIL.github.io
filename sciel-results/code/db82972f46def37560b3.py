import random
import time
import math

# Global precomputation
N = 16
NUM_EDGES = 120
NUM_K4 = 1820

def _precompute():
    edge_idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_idx_map[(i, j)] = idx
            idx += 1
            
    k4_edges = []
    edge_to_k4s = [[] for _ in range(NUM_EDGES)]
    
    k4_idx = 0
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    edges = [
                        edge_idx_map[(a, b)],
                        edge_idx_map[(a, c)],
                        edge_idx_map[(a, d)],
                        edge_idx_map[(b, c)],
                        edge_idx_map[(b, d)],
                        edge_idx_map[(c, d)]
                    ]
                    k4_edges.append(edges)
                    for e in edges:
                        edge_to_k4s[e].append(k4_idx)
                    k4_idx += 1
                    
    return edge_idx_map, k4_edges, edge_to_k4s

EDGE_IDX_MAP, K4_EDGES, EDGE_TO_K4S = _precompute()

def construct():
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Try K=13 first
    for K in [13, 14, 15]:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Initialize
        colors = [random.randint(0, K - 1) for _ in range(NUM_EDGES)]
        
        # Build k4_counts
        # k4_counts[k] is a dict {color: count}
        k4_counts = [{} for _ in range(NUM_K4)]
        
        for k in range(NUM_K4):
            counts = {}
            for e in K4_EDGES[k]:
                c = colors[e]
                counts[c] = counts.get(c, 0) + 1
            k4_counts[k] = counts
            
        # Count violations
        violations = 0
        for k in range(NUM_K4):
            if len(k4_counts[k]) < 5:
                violations += 1
                
        if violations == 0:
            return colors
            
        # Simulated Annealing
        T = 10.0
        cooling = 0.999
        T_min = 0.01
        best_violations = violations
        best_colors = colors[:]
        
        while T > T_min and time.time() - start_time < time_limit - 5:
            # Pick random edge
            e = random.randint(0, NUM_EDGES - 1)
            old_c = colors[e]
            
            # Pick new color
            # To improve mixing, pick a random color different from old_c
            new_c = random.randint(0, K - 1)
            if new_c == old_c:
                continue
                
            # Calculate delta violations
            delta_v = 0
            # We need to know if the K4 was a violation before and after
            # We can simulate the change for the 91 K4s
            
            # To avoid modifying state if rejected, we calculate delta first
            # But modifying and reverting is faster if we don't store snapshots?
            # Storing snapshots of 91 dicts is expensive.
            # Let's just calculate the effect.
            
            # For each K4 containing e:
            # 1. old_count = k4_counts[k].get(old_c, 0)
            # 2. new_count = k4_counts[k].get(new_c, 0)
            # 3. old_distinct = len(k4_counts[k])
            # 4. Simulate:
            #    - Remove old_c: if old_count == 1, distinct decreases by 1.
            #    - Add new_c: if new_count == 0, distinct increases by 1.
            #    - Note: if old_c == new_c, no change (handled above).
            
            was_violation = 0
            is_violation = 0
            
            # We need to be careful.
            # Let's just update the structures, calculate delta, and revert if rejected.
            # Updating 91 dicts is fast.
            
            # Save old states? No, just revert.
            
            # Apply change
            for k in EDGE_TO_K4S[e]:
                counts = k4_counts[k]
                # Remove old_c
                oc = counts.get(old_c, 0)
                if oc == 1:
                    del counts[old_c]
                else:
                    counts[old_c] = oc - 1
                    
                # Add new_c
                nc = counts.get(new_c, 0)
                if nc == 0:
                    counts[new_c] = 1
                else:
                    counts[new_c] = nc + 1
                    
            # This is getting messy to revert.
            # Better approach: Calculate delta without modifying.
            
            # Let's restart the loop logic with pure calculation.
            
            delta_v = 0
            for k in EDGE_TO_K4S[e]:
                counts = k4_counts[k]
                old_distinct = len(counts)
                
                oc = counts.get(old_c, 0)
                nc = counts.get(new_c, 0)
                
                # Determine new distinct count
                new_distinct = old_distinct
                
                # Effect of removing old_c
                if oc == 1:
                    new_distinct -= 1
                # Effect of adding new_c
                if nc == 0:
                    new_distinct += 1
                    
                # Note: if old_c == new_c, we skipped.
                # If old_c != new_c:
                # If oc > 1, removing doesn't change distinct.
                # If nc > 0, adding doesn't change distinct.
                
                old_violation = 1 if old_distinct < 5 else 0
                new_violation = 1 if new_distinct < 5 else 0
                
                delta_v += new_violation - old_violation
                
            # Acceptance
            if delta_v <= 0 or random.random() < math.exp(-delta_v / T):
                # Apply change
                colors[e] = new_c
                for k in EDGE_TO_K4S[e]:
                    counts = k4_counts[k]
                    oc = counts.get(old_c, 0)
                    if oc == 1:
                        del counts[old_c]
                    else:
                        counts[old_c] = oc - 1
                        
                    nc = counts.get(new_c, 0)
                    if nc == 0:
                        counts[new_c] = 1
                    else:
                        counts[new_c] = nc + 1
                        
                violations += delta_v
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors[:]
                    if violations == 0:
                        return best_colors
                        
            T *= cooling
            
        # End SA for this K
        if best_violations == 0:
            return best_colors

    # If we didn't find a solution with K=13,14,15, return best found or fallback
    # Fallback to cyclic coloring (17 colors) to ensure validity
    colors = []
    for i in range(N):
        for j in range(i + 1, N):
            colors.append((i + j) % 17)
    return colors
