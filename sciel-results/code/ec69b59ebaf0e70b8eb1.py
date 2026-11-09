import time
import random
import numpy as np
import itertools

N = 20
E_TOTAL = N * (N - 1) // 2  # 190

# Precompute all edges
_all_edges = []
edge_index = {}
for i in range(E_TOTAL):
    u = i // (N - 1 - (i % (N - 1))) # This formula is tricky, let's just use loops
    # Simpler:
    pass

_all_edges = []
idx = 0
for u in range(N):
    for v in range(u + 1, N):
        _all_edges.append((u, v))
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
        idx += 1

# Precompute all 4-cycles
# For each set of 4 vertices, there are 3 distinct 4-cycles
_cycles = []
for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    # Cycle 1: a-b-c-d-a
    _cycles.append((edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)]))
    # Cycle 2: a-b-d-c-a
    _cycles.append((edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)]))
    # Cycle 3: a-c-b-d-a
    _cycles.append((edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)]))

C_TOTAL = len(_cycles)
_cycles = np.array(_cycles, dtype=np.int32)

# Map edge to cycles
_edge_to_cycles = [[] for _ in range(E_TOTAL)]
for cid, cycle in enumerate(_cycles):
    for e in cycle:
        _edge_to_cycles[e].append(cid)

# Precompute for each cycle, the indices of the other 3 edges for each position
# cycle_other_edges[cid][pos] = (e1, e2, e3) where e1,e2,e3 are the other edges
# This helps in delta calculation
_cycle_other = []
for cid in range(C_TOTAL):
    e0, e1, e2, e3 = _cycles[cid]
    _cycle_other.append([
        (e1, e2, e3),
        (e0, e2, e3),
        (e0, e1, e3),
        (e0, e1, e2)
    ])

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # State
    active = np.ones(E_TOTAL, dtype=bool)
    colors = np.random.randint(0, N, size=E_TOTAL, dtype=np.int32)
    
    # Precompute active count per cycle? 
    # Or just check on the fly.
    # Let's maintain a 'bad' array for cycles
    bad = np.zeros(C_TOTAL, dtype=bool)
    
    def is_rainbow(c0, c1, c2, c3):
        return (c0 != c1) and (c0 != c2) and (c0 != c3) and (c1 != c2) and (c1 != c3) and (c2 != c3)
    
    # Initialize bad array
    for cid in range(C_TOTAL):
        e0, e1, e2, e3 = _cycles[cid]
        if active[e0] and active[e1] and active[e2] and active[e3]:
            if not is_rainbow(colors[e0], colors[e1], colors[e2], colors[e3]):
                bad[cid] = True
    
    current_bad = np.sum(bad)
    current_edges = np.sum(active)
    M = 10.0
    current_obj = current_edges - M * current_bad
    
    best_obj = current_obj
    best_active = active.copy()
    best_colors = colors.copy()
    
    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    # Helper to update bad status for a cycle given its 4 colors and active status
    # Returns 1 if bad, 0 if not
    def cycle_is_bad(cid, act, col):
        e0, e1, e2, e3 = _cycles[cid]
        if not (act[e0] and act[e1] and act[e2] and act[e3]):
            return 0
        c0, c1, c2, c3 = col[e0], col[e1], col[e2], col[e3]
        if (c0 == c1) or (c0 == c2) or (c0 == c3) or (c1 == c2) or (c1 == c3) or (c2 == c3):
            return 1
        return 0

    # Main Loop
    while time.time() < deadline:
        # Pick move type
        if random.random() < 0.5:
            # Recolor
            # Pick random active edge
            active_indices = np.where(active)[0]
            if len(active_indices) == 0:
                break
            e_idx = active_indices[random.randint(0, len(active_indices) - 1)]
            old_c = colors[e_idx]
            new_c = random.randint(0, N - 1)
            while new_c == old_c:
                new_c = random.randint(0, N - 1)
            
            # Calculate delta
            delta_bad = 0
            for cid in _edge_to_cycles[e_idx]:
                # Check if cycle is currently active
                e0, e1, e2, e3 = _cycles[cid]
                # Since e_idx is active, check others
                # Find position of e_idx in cycle
                # We stored _cycle_other, but we need to know which position e_idx is
                # Let's just check active status of all 4
                if not (active[e0] and active[e1] and active[e2] and active[e3]):
                    continue
                
                # Get colors
                c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
                
                # Determine which color changes
                if e0 == e_idx:
                    old_val = c0
                    new_vals = (new_c, c1, c2, c3)
                elif e1 == e_idx:
                    old_val = c1
                    new_vals = (c0, new_c, c2, c3)
                elif e2 == e_idx:
                    old_val = c2
                    new_vals = (c0, c1, new_c, c3)
                else:
                    old_val = c3
                    new_vals = (c0, c1, c2, new_c)
                
                # Old badness
                old_bad = 0
                if (old_val == new_vals[1]) or (old_val == new_vals[2]) or (old_val == new_vals[3]) or \
                   (new_vals[1] == new_vals[2]) or (new_vals[1] == new_vals[3]) or (new_vals[2] == new_vals[3]):
                    # Note: new_vals[1..3] are unchanged, so their mutual equality is constant
                    # We only need to check if old_val conflicts with any of them
                    # And if the others conflict with each other (which is constant for this move)
                    pass # This logic is getting messy. Let's just recompute is_rainbow for old and new.
                
                # Simpler:
                old_rainbow = is_rainbow(c0, c1, c2, c3)
                
                if e0 == e_idx:
                    new_rainbow = is_rainbow(new_c, c1, c2, c3)
                elif e1 == e_idx:
                    new_rainbow = is_rainbow(c0, new_c, c2, c3)
                elif e2 == e_idx:
                    new_rainbow = is_rainbow(c0, c1, new_c, c3)
                else:
                    new_rainbow = is_rainbow(c0, c1, c2, new_c)
                
                if not old_rainbow:
                    delta_bad -= 1 # Was bad, now checking if it becomes good
                if not new_rainbow:
                    delta_bad += 1 # Becomes bad (or stays bad)
                
                # Correction:
                # delta = new_bad_count - old_bad_count
                # old_bad = 1 if not old_rainbow else 0
                # new_bad = 1 if not new_rainbow else 0
                # delta += (1 - new_rainbow) - (1 - old_rainbow) = old_rainbow - new_rainbow
                # So:
                # if old_rainbow and not new_rainbow: delta += 1
                # if not old_rainbow and new_rainbow: delta -= 1
                # else: delta += 0
                
                # Let's restart the delta logic cleanly
                # I will rewrite the loop body in the final code to be correct.
            
            # Placeholder for actual implementation
            pass

    # This draft is getting too long and complex in thought. 
    # I will write the final code directly with careful delta calculation.

    # Final Code Structure:
    # 1. Precompute
    # 2. Init
    # 3. Loop:
    #    - Random Move
    #    - Delta Calc
    #    - Accept/Reject
    #    - Update Best

    # Let's refine the Delta Calc for Recolor:
    # For each cycle containing e:
    #   If cycle is active:
    #     old_bad = 1 if not rainbow(old_colors) else 0
    #     new_bad = 1 if not rainbow(new_colors) else 0
    #     delta_bad += new_bad - old_bad
    
    # Delta Calc for Toggle (Add):
    # For each cycle containing e:
    #   If other 3 are active:
    #     new_bad = 1 if not rainbow(colors) else 0
    #     delta_bad += new_bad  (old was 0 because e was inactive)
    #     # Also need to mark cycle as active? 
    #     # We don't maintain 'active_cycle' array, we just check 'active' array.
    #     # So we don't need to update state for 'active_cycle', just 'bad' array?
    #     # Actually, we maintain 'bad' array to get current_bad quickly?
    #     # No, we calculate delta, and if accepted, we update 'bad' array.
    #     # If rejected, we don't.
    #     # So we need to update 'bad' array for the cycles affected.
    
    # This requires knowing if the cycle was active before.
    # If adding e:
    #   If other 3 active:
    #      Cycle becomes active.
    #      old_bad = 0 (since e was inactive, cycle was inactive -> not bad)
    #      new_bad = 1 if not rainbow else 0
    #      delta_bad += new_bad
    #   Else:
    #      delta_bad += 0
    
    # If removing e:
    #   If other 3 active:
    #      Cycle becomes inactive.
    #      old_bad = 1 if bad[cid] else 0 (Note: bad[cid] reflects current state where e is active)
    #      new_bad = 0
    #      delta_bad -= old_bad
    #   Else:
    #      delta_bad += 0 (Cycle was inactive, remains inactive)
    
    # This works.

    # I will implement this.

    # Re-initializing state for the final function
    active = np.ones(E_TOTAL, dtype=bool)
    colors = np.random.randint(0, N, size=E_TOTAL, dtype=np.int32)
    bad = np.zeros(C_TOTAL, dtype=bool)
    
    for cid in range(C_TOTAL):
        e0, e1, e2, e3 = _cycles[cid]
        if active[e0] and active[e1] and active[e2] and active[e3]:
            if not is_rainbow(colors[e0], colors[e1], colors[e2], colors[e3]):
                bad[cid] = True
                
    current_bad = np.sum(bad)
    current_edges = np.sum(active)
    M = 10.0
    current_obj = current_edges - M * current_bad
    best_obj = current_obj
    best_active = active.copy()
    best_colors = colors.copy()
    
    T = 1.0
    cooling = 0.9999
    min_T = 0.01
    
    while time.time() < deadline:
        if random.random() < 0.5:
            # Recolor
            active_indices = np.where(active)[0]
            if len(active_indices) == 0: continue
            e_idx = active_indices[random.randint(0, len(active_indices) - 1)]
            old_c = colors[e_idx]
            new_c = random.randint(0, N - 1)
            while new_c == old_c:
                new_c = random.randint(0, N - 1)
            
            delta_bad = 0
            affected_cycles = []
            for cid in _edge_to_cycles[e_idx]:
                e0, e1, e2, e3 = _cycles[cid]
                if not (active[e0] and active[e1] and active[e2] and active[e3]):
                    continue
                
                c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
                old_rainbow = is_rainbow(c0, c1, c2, c3)
                
                if e0 == e_idx:
                    new_rainbow = is_rainbow(new_c, c1, c2, c3)
                elif e1 == e_idx:
                    new_rainbow = is_rainbow(c0, new_c, c2, c3)
                elif e2 == e_idx:
                    new_rainbow = is_rainbow(c0, c1, new_c, c3)
                else:
                    new_rainbow = is_rainbow(c0, c1, c2, new_c)
                
                if old_rainbow and not new_rainbow:
                    delta_bad += 1
                elif not old_rainbow and new_rainbow:
                    delta_bad -= 1
                
                affected_cycles.append(cid)
            
            delta_edges = 0
            delta_obj = delta_edges - M * delta_bad
            
            if delta_obj < 0 or (T > min_T and random.random() < np.exp(-delta_obj / T)):
                colors[e_idx] = new_c
                for cid in affected_cycles:
                    e0, e1, e2, e3 = _cycles[cid]
                    if active[e0] and active[e1] and active[e2] and active[e3]:
                        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
                        bad[cid] = not is_rainbow(c0, c1, c2, c3)
                current_bad += delta_bad
                current_obj += delta_obj
                if current_obj > best_obj:
                    best_obj = current_obj
                    best_active = active.copy()
                    best_colors = colors.copy()
        else:
            # Toggle
            e_idx = random.randint(0, E_TOTAL - 1)
            if active[e_idx]:
                # Remove
                delta_bad = 0
                affected_cycles = []
                for cid in _edge_to_cycles[e_idx]:
                    e0, e1, e2, e3 = _cycles[cid]
                    # Check if other 3 are active
                    # e_idx is one of them.
                    if e0 == e_idx:
                        others_active = active[e1] and active[e2] and active[e3]
                    elif e1 == e_idx:
                        others_active = active[e0] and active[e2] and active[e3]
                    elif e2 == e_idx:
                        others_active = active[e0] and active[e1] and active[e3]
                    else:
                        others_active = active[e0] and active[e1] and active[e2]
                    
                    if others_active:
                        # Cycle was active.
                        if bad[cid]:
                            delta_bad -= 1
                        affected_cycles.append(cid)
                
                delta_edges = -1
                delta_obj = delta_edges - M * delta_bad
                
                if delta_obj < 0 or (T > min_T and random.random() < np.exp(-delta_obj / T)):
                    active[e_idx] = False
                    for cid in affected_cycles:
                        bad[cid] = False # Cycle becomes inactive, so not bad
                    current_bad += delta_bad
                    current_edges += delta_edges
                    current_obj += delta_obj
                    if current_obj > best_obj:
                        best_obj = current_obj
                        best_active = active.copy()
                        best_colors = colors.copy()
            else:
                # Add
                delta_bad = 0
                affected_cycles = []
                for cid in _edge_to_cycles[e_idx]:
                    e0, e1, e2, e3 = _cycles[cid]
                    if e0 == e_idx:
                        others_active = active[e1] and active[e2] and active[e3]
                    elif e1 == e_idx:
                        others_active = active[e0] and active[e2] and active[e3]
                    elif e2 == e_idx:
                        others_active = active[e0] and active[e1] and active[e3]
                    else:
                        others_active = active[e0] and active[e1] and active[e2]
                    
                    if others_active:
                        # Cycle becomes active.
                        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
                        if not is_rainbow(c0, c1, c2, c3):
                            delta_bad += 1
                        affected_cycles.append(cid)
                
                delta_edges = 1
                delta_obj = delta_edges - M * delta_bad
                
                if delta_obj < 0 or (T > min_T and random.random() < np.exp(-delta_obj / T)):
                    active[e_idx] = True
                    for cid in affected_cycles:
                        e0, e1, e2, e3 = _cycles[cid]
                        c0, c1, c2, c3 = colors[e0], colors[e1], colors[e2], colors[e3]
                        bad[cid] = not is_rainbow(c0, c1, c2, c3)
                    current_bad += delta_bad
                    current_edges += delta_edges
                    current_obj += delta_obj
                    if current_obj > best_obj:
                        best_obj = current_obj
                        best_active = active.copy()
                        best_colors = colors.copy()
        
        T *= cooling
        if T < min_T:
            T = min_T

    # Return best result
    result = []
    for i in range(E_TOTAL):
        if best_active[i]:
            u, v = _all_edges[i]
            result.append([u, v, int(best_colors[i])])
    return result

# Note: The is_rainbow function is defined inside construct in the draft above? 
# No, I defined it outside in the thought, but inside in the code block structure.
# I will place it outside for clarity and speed (avoiding closure lookup).

def _is_rainbow(c0, c1, c2, c3):
    return (c0 != c1) and (c0 != c2) and (c0 != c3) and (c1 != c2) and (c1 != c3) and (c2 != c3)

# Update construct to use _is_rainbow

# Final check on logic:
# If I add an edge, and it creates a bad cycle, delta_bad increases.
# Objective decreases if M > 1.
# So it's unlikely to be accepted unless T is high.
# This encourages a valid subgraph.
# Recoloring a bad cycle to make it good: delta_bad decreases. Objective increases. Always accepted.
# This should drive the system to a valid state.
# Then, adding edges that don't create bad cycles: delta_bad 0, delta_edges 1. Objective increases. Accepted.
# Adding edges that create bad cycles: Objective decreases. Rejected (mostly).
# So we end up with a maximal valid subgraph.

# One optimization: If bad_count is 0, we can try to add edges more aggressively?
# The SA handles this via T.

# I will output the code now.

