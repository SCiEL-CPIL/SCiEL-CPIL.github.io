import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute K4 structures
    n = 16
    # Generate all pairs (i, j) with i < j
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
    
    num_edges = len(edges) # 120
    
    # Generate all K4s (sets of 4 vertices)
    k4s = []
    k4_edges_list = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    verts = (a, b, c, d)
                    # Get edge indices for this K4
                    e1 = edge_index[(a, b)]
                    e2 = edge_index[(a, c)]
                    e3 = edge_index[(a, d)]
                    e4 = edge_index[(b, c)]
                    e5 = edge_index[(b, d)]
                    e6 = edge_index[(c, d)]
                    k4s.append(verts)
                    k4_edges_list.append([e1, e2, e3, e4, e5, e6])
    
    num_k4s = len(k4s) # 1820
    
    # Map edge index to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges_list):
        for e in e_list:
            edge_to_k4s[e].append(k4_idx)
    
    def count_bad(colours):
        """Count K4s with < 5 distinct colours."""
        bad_count = 0
        for k4_idx in range(num_k4s):
            e_list = k4_edges_list[k4_idx]
            c_set = set(colours[e] for e in e_list)
            if len(c_set) < 5:
                bad_count += 1
        return bad_count

    def get_bad_k4s(colours):
        """Return list of indices of bad K4s."""
        bads = []
        for k4_idx in range(num_k4s):
            e_list = k4_edges_list[k4_idx]
            c_set = set(colours[e] for e in e_list)
            if len(c_set) < 5:
                bads.append(k4_idx)
        return bads

    def try_solve(k, time_limit):
        """Try to find a valid k-colouring."""
        end_time = time.time() + time_limit
        best_colours = None
        best_bad = float('inf')
        
        # Multiple restarts
        while time.time() < end_time:
            # Random init
            colours = [random.randint(0, k - 1) for _ in range(num_edges)]
            bad_k4s = get_bad_k4s(colours)
            current_bad = len(bad_k4s)
            
            if current_bad == 0:
                return colours
            
            # Simulated Annealing
            T = 1.0
            no_improve_count = 0
            max_no_improve = 1000
            
            while time.time() < end_time and current_bad > 0:
                if current_bad < best_bad:
                    best_bad = current_bad
                    best_colours = colours.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count > max_no_improve:
                        break # Restart this annealing cycle
                
                # Pick a random bad K4
                if not bad_k4s:
                    break
                k4_idx = random.choice(bad_k4s)
                e_list = k4_edges_list[k4_idx]
                e_idx = random.choice(e_list)
                
                old_c = colours[e_idx]
                new_c = random.randint(0, k - 1)
                if new_c == old_c:
                    continue
                
                # Evaluate change
                # We need to update bad status of all K4s containing e_idx
                affected_k4s = edge_to_k4s[e_idx]
                
                # Calculate delta
                delta = 0
                # We need to know the colour sets of affected K4s to update efficiently
                # For simplicity in this inner loop, we can just recalculate the bad count
                # but that is O(num_k4s * 6) which is ~10k ops. 
                # 10k ops * 1000 steps = 10M ops. Might be too slow in Python if not optimized.
                # Let's try to be smarter.
                
                # Store current colour sets for all K4s?
                # That uses memory but speeds up updates.
                # Let's maintain `k4_colour_counts` or just re-evaluate affected ones.
                
                # Optimization: Maintain `k4_colours` as a list of sets.
                # Let's do that.
                
                pass # Placeholder for logic below
                
            # If we broke out, try next restart
        return best_colours if best_bad == 0 else None

    # Refine try_solve with maintained state for speed
    def try_solve_fast(k, time_limit):
        end_time = time.time() + time_limit
        
        # State:
        # colours: list of 120 ints
        # k4_sets: list of 1820 sets of colours
        # bad_k4s: set of k4 indices with len(set) < 5
        
        while time.time() < end_time:
            colours = [random.randint(0, k - 1) for _ in range(num_edges)]
            k4_sets = [set(colours[e] for e in e_list) for e_list in k4_edges_list]
            bad_k4s = set(i for i, s in enumerate(k4_sets) if len(s) < 5)
            
            if not bad_k4s:
                return colours
            
            T = 1.0
            cool_rate = 0.999
            min_T = 0.01
            
            while time.time() < end_time and bad_k4s:
                if T < min_T:
                    break
                
                # Pick a random bad K4
                k4_idx = random.choice(list(bad_k4s))
                e_list = k4_edges_list[k4_idx]
                e_idx = random.choice(e_list)
                
                old_c = colours[e_idx]
                new_c = random.randint(0, k - 1)
                if new_c == old_c:
                    continue
                
                # Update affected K4s
                affected = edge_to_k4s[e_idx]
                delta = 0
                # We need to check if the change improves or worsens the bad status
                # For each affected K4:
                # 1. Remove old_c, add new_c to its set
                # 2. Check if it becomes bad or good
                
                # To avoid modifying sets and then reverting if rejected, 
                # we can calculate the new size.
                
                # Let's just modify, check delta, and revert if rejected.
                
                for ak in affected:
                    s = k4_sets[ak]
                    # Check if old_c is the only one of that colour? 
                    # No, sets don't track counts. 
                    # If we just use sets, removing an element that might still be present 
                    # in other edges of the K4 is tricky without counts.
                    
                    # We need counts.
                    # Let's switch to `k4_colour_counts` (dict or array).
                    # Since k is small, an array of size k is good.
                    
                    pass
            return None # Failed this restart

    # Let's implement the count-based version properly
    def solve_with_counts(k, time_limit):
        end_time = time.time() + time_limit
        # k4_counts: list of 1820 arrays of size k
        # This is 1820 * 15 ~ 27k integers. Very small.
        
        while time.time() < end_time:
            colours = [random.randint(0, k - 1) for _ in range(num_edges)]
            k4_counts = [ [0]*k for _ in range(num_k4s) ]
            
            # Init counts
            for k4_idx, e_list in enumerate(k4_edges_list):
                for e in e_list:
                    k4_counts[k4_idx][colours[e]] += 1
            
            # Init bad status
            bad_k4s = set()
            for k4_idx in range(num_k4s):
                distinct = sum(1 for c in k4_counts[k4_idx] if c > 0)
                if distinct < 5:
                    bad_k4s.add(k4_idx)
            
            if not bad_k4s:
                return colours
            
            T = 1.0
            cool_rate = 0.9995
            min_T = 0.01
            no_improve = 0
            
            while time.time() < end_time and bad_k4s:
                if T < min_T:
                    break
                
                # Pick bad K4
                k4_idx = random.choice(list(bad_k4s))
                e_list = k4_edges_list[k4_idx]
                e_idx = random.choice(e_list)
                
                old_c = colours[e_idx]
                new_c = random.randint(0, k - 1)
                if new_c == old_c:
                    continue
                
                # Evaluate
                delta = 0
                affected = edge_to_k4s[e_idx]
                
                for ak in affected:
                    cnts = k4_counts[ak]
                    # Before change
                    was_bad = (sum(1 for c in cnts if c > 0) < 5)
                    
                    # Apply change
                    cnts[old_c] -= 1
                    cnts[new_c] += 1
                    
                    # After change
                    is_bad = (sum(1 for c in cnts if c > 0) < 5)
                    
                    if was_bad and not is_bad:
                        delta -= 1
                    elif was_bad and is_bad:
                        delta += 0 # No change in bad count, but colour distribution changed
                    elif not was_bad and is_bad:
                        delta += 1
                    else:
                        delta += 0
                    
                    # Revert for now, will apply if accepted
                    cnts[old_c] += 1
                    cnts[new_c] -= 1
                
                # Acceptance
                if delta < 0:
                    accept = True
                else:
                    if delta == 0:
                        accept = True # Always accept neutral moves to explore
                    else:
                        if random.random() < math.exp(-delta / T):
                            accept = True
                        else:
                            accept = False
                
                if accept:
                    # Apply change
                    colours[e_idx] = new_c
                    for ak in affected:
                        cnts = k4_counts[ak]
                        was_bad = (sum(1 for c in cnts if c > 0) < 5)
                        
                        cnts[old_c] -= 1
                        cnts[new_c] += 1
                        
                        is_bad = (sum(1 for c in cnts if c > 0) < 5)
                        
                        if was_bad and not is_bad:
                            bad_k4s.discard(ak)
                        elif not was_bad and is_bad:
                            bad_k4s.add(ak)
                    
                    if delta < 0:
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
                T *= cool_rate
                
                if no_improve > 2000:
                    break # Restart this annealing run
            
            # If we exited inner loop without solving, outer loop restarts with new random init

    # Strategy:
    # Try k=13 for 30s
    # Try k=14 for 30s
    # Try k=15 for 30s
    # If all fail, return seed
    
    for k in [13, 14, 15]:
        if time.time() > deadline - 10:
            break
        time_limit = min(30, deadline - time.time() - 5)
        if time_limit <= 0:
            break
        
        res = solve_with_counts(k, time_limit)
        if res is not None:
            return res
    
    # Fallback to seed
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
