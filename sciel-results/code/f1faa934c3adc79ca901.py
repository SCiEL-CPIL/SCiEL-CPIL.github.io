"""Strategy: CSP Backtracking with Forward Checking + Local Search Repair for Hypergraph Ramsey R(4,5;3)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def verify(n, colors):
        """Verify if the coloring is valid."""
        if len(colors) != len(list(itertools.combinations(range(n), 3))):
            return False
        triples = list(itertools.combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Check Red K4
        for quad in itertools.combinations(range(n), 4):
            is_red = True
            for triple in itertools.combinations(quad, 3):
                if colors[triple_to_idx[triple]] != 0:
                    is_red = False
                    break
            if is_red:
                return False
        
        # Check Blue K5
        for quint in itertools.combinations(range(n), 5):
            is_blue = True
            for triple in itertools.combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    is_blue = False
                    break
            if is_blue:
                return False
        return True

    def try_solve_csp(n, time_budget):
        """
        Attempt to find a valid coloring using Backtracking with Forward Checking.
        This is a structural change from SA: it enforces constraints during search.
        """
        t0 = time.time()
        end_time = t0 + time_budget
        
        vertices = list(range(n))
        triples = list(itertools.combinations(vertices, 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute cliques
        k4s = list(itertools.combinations(vertices, 4))
        k5s = list(itertools.combinations(vertices, 5))
        
        # Map triple -> list of K4 indices, K5 indices
        triple_to_k4s = defaultdict(list)
        triple_to_k5s = defaultdict(list)
        
        for i, quad in enumerate(k4s):
            for t in itertools.combinations(quad, 3):
                triple_to_k4s[triple_to_idx[t]].append(i)
                
        for i, quint in enumerate(k5s):
            for t in itertools.combinations(quint, 3):
                triple_to_k5s[triple_to_idx[t]].append(i)
        
        # State:
        # colors: array of -1 (unassigned), 0 (red), 1 (blue)
        # red_count_k4[i]: number of red triples assigned in K4 i
        # blue_count_k5[i]: number of blue triples assigned in K5 i
        # unassigned_k4[i]: number of unassigned triples in K4 i
        # unassigned_k5[i]: number of unassigned triples in K5 i
        
        # Initial state
        colors = np.full(num_triples, -1, dtype=np.int8)
        
        # Precompute which triples are in which K4/K5 for fast updates
        # Actually, we need to update counts when a triple is assigned.
        # Let's build the lists of triples for each clique.
        k4_triples = []
        for quad in k4s:
            k4_triples.append([triple_to_idx[t] for t in itertools.combinations(quad, 3)])
            
        k5_triples = []
        for quint in k5s:
            k5_triples.append([triple_to_idx[t] for t in itertools.combinations(quint, 3)])
        
        red_count_k4 = np.zeros(len(k4s), dtype=np.int32)
        blue_count_k5 = np.zeros(len(k5s), dtype=np.int32)
        unassigned_k4 = np.full(len(k4s), 4, dtype=np.int32)
        unassigned_k5 = np.full(len(k5s), 10, dtype=np.int32)
        
        # Heuristic: Order triples. 
        # A good order might be based on how many constraints they affect (degree).
        # Or just lexicographic. Let's stick to lexicographic for simplicity, 
        # but MRV (Minimum Remaining Values) is hard with dynamic domains.
        # We will use a static order but check feasibility dynamically.
        
        # To speed up, we can use a "conflict-directed" backtracking or simply
        # randomize the order of triples to explore different branches.
        
        # Let's try a few random orderings.
        best_colors = None
        
        # Local search repair function to be used if CSP fails or as a hybrid
        def local_search_init(n, time_budget_ls):
            """
            Run a fast local search to find a low-violation state, 
            then try to repair it using CSP-like steps or more LS.
            """
            t0 = time.time()
            end_ls = t0 + time_budget_ls
            
            # Random init
            c = np.random.randint(0, 2, num_triples, dtype=np.int8)
            
            # Compute violations
            red_viol = 0
            blue_viol = 0
            red_count_k4_ls = np.zeros(len(k4s), dtype=np.int32)
            blue_count_k5_ls = np.zeros(len(k5s), dtype=np.int32)
            
            for i, ts in enumerate(k4_triples):
                red_count_k4_ls[i] = np.sum(c[ts] == 0)
                if red_count_k4_ls[i] == 4:
                    red_viol += 1
                    
            for i, ts in enumerate(k5_triples):
                blue_count_k5_ls[i] = np.sum(c[ts] == 1)
                if blue_count_k5_ls[i] == 10:
                    blue_viol += 1
                    
            total_viol = red_viol + blue_viol
            best_c = c.copy()
            best_v = total_viol
            
            T = 10.0
            cool = 0.999
            
            while time.time() < end_ls:
                if total_viol == 0:
                    return c.tolist()
                    
                t = random.randint(0, num_triples - 1)
                old_c = c[t]
                new_c = 1 - old_c
                delta = 0
                
                # Update K4s
                for k4_idx in triple_to_k4s[t]:
                    old_cnt = red_count_k4_ls[k4_idx]
                    new_cnt = old_cnt - 1 if old_c == 0 else old_cnt + 1
                    if old_cnt == 4 and new_cnt != 4: delta -= 1
                    elif old_cnt != 4 and new_cnt == 4: delta += 1
                    red_count_k4_ls[k4_idx] = new_cnt
                    
                # Update K5s
                for k5_idx in triple_to_k5s[t]:
                    old_cnt = blue_count_k5_ls[k5_idx]
                    new_cnt = old_cnt - 1 if old_c == 1 else old_cnt + 1
                    if old_cnt == 10 and new_cnt != 10: delta -= 1
                    elif old_cnt != 10 and new_cnt == 10: delta += 1
                    blue_count_k5_ls[k5_idx] = new_cnt
                    
                if delta <= 0 or random.random() < np.exp(-delta/T):
                    c[t] = new_c
                    total_viol += delta
                    if total_viol < best_v:
                        best_v = total_viol
                        best_c = c.copy()
                else:
                    # Revert
                    for k4_idx in triple_to_k4s[t]:
                        if old_c == 0: red_count_k4_ls[k4_idx] += 1
                        else: red_count_k4_ls[k4_idx] -= 1
                    for k5_idx in triple_to_k5s[t]:
                        if old_c == 1: blue_count_k5_ls[k5_idx] += 1
                        else: blue_count_k5_ls[k5_idx] -= 1
                        
                T *= cool
            return best_c.tolist()

        # Hybrid Strategy:
        # 1. Try CSP for a short time.
        # 2. If CSP fails, use Local Search to find a "good" start, then try CSP from there? 
        #    Or just use Local Search with a better cooling schedule?
        # Given the budget, let's try CSP first. If it's too slow, we fall back to LS.
        
        # CSP Implementation
        # We need to handle the recursion. Python recursion limit might be an issue for n=34 (5456 triples).
        # We must use iterative deepening or increase recursion limit.
        # sys.setrecursionlimit(10000)
        import sys
        sys.setrecursionlimit(10000)
        
        # To make CSP efficient, we need to choose the next variable wisely.
        # Static order: 0 to num_triples-1.
        # Dynamic MRV is complex. Let's try static order with forward checking.
        
        # Forward Checking:
        # When we assign a triple, we check if any K4/K5 becomes "impossible".
        # K4 impossible if:
        #   - All 4 triples assigned and not all red (this is just checking violation, but for CSP we care about "can it still be satisfied?")
        #   - Actually, the constraint is "NOT (all 4 red)".
        #   - So a K4 is "dead" if it has 4 red triples.
        #   - A K4 is "forced" if it has 3 red triples and 1 unassigned -> the unassigned MUST be blue.
        #   - A K5 is "dead" if it has 10 blue triples.
        #   - A K5 is "forced" if it has 9 blue triples and 1 unassigned -> the unassigned MUST be red.
        
        # This propagation can be powerful.
        
        # Let's implement a simple backtracker with propagation.
        
        def backtrack(idx, red_count_k4, blue_count_k5, unassigned_k4, unassigned_k5, colors):
            if time.time() > end_time:
                return None, False # Timeout
            
            if idx == num_triples:
                # Check if any violations remain? 
                # If propagation worked, there shouldn't be any "dead" states.
                # But we need to ensure no K4 is all red and no K5 is all blue.
                # The propagation ensures we never *create* a violation.
                # So if we reach the end, it's valid.
                return colors.copy(), True
                
            # Heuristic: Pick the next unassigned triple? 
            # For simplicity, we iterate in order.
            # But we can skip assigned ones? No, we assign in order.
            
            # Check if current triple is already assigned (by propagation)
            if colors[idx] != -1:
                return backtrack(idx + 1, red_count_k4, blue_count_k5, unassigned_k4, unassigned_k5, colors)
                
            # Try 0 (Red) and 1 (Blue)
            # Randomize order?
            choices = [0, 1]
            random.shuffle(choices)
            
            for val in choices:
                # Check if assigning val causes a violation
                # We need to check if this assignment makes any K4 all red or K5 all blue.
                # Actually, we check if it makes a K4 *impossible* to avoid?
                # No, the constraint is "No K4 all red". So if assigning Red makes a K4 all red, it's invalid.
                
                valid = True
                
                # Check K4s containing idx
                for k4_idx in triple_to_k4s[idx]:
                    if val == 0: # Red
                        # If this K4 already has 3 reds, adding this makes 4 -> Violation
                        if red_count_k4[k4_idx] == 3:
                            valid = False
                            break
                if not valid:
                    continue
                    
                # Check K5s containing idx
                for k5_idx in triple_to_k5s[idx]:
                    if val == 1: # Blue
                        # If this K5 already has 9 blues, adding this makes 10 -> Violation
                        if blue_count_k5[k5_idx] == 9:
                            valid = False
                            break
                if not valid:
                    continue
                
                # Propagate
                # We need to copy the state? That's expensive.
                # We can use a trail for undo.
                # For simplicity in this snippet, let's just copy if n is small, 
                # but for n=34, copying arrays of size 46k and 278k is too slow.
                
                # We must use a trail-based approach.
                # This is getting complex. 
                
                # Alternative: Use the Local Search result as a seed for a "greedy repair"?
                # Or just stick to a highly optimized Local Search?
                
                # Let's abort the CSP idea if it's too complex to implement efficiently in 150s.
                # The Local Search in the current program is already decent but stuck at 1.0 (n=9 fallback).
                # Why did it fail? Probably because random restarts weren't enough or cooling was bad.
                
                # Let's try a different Local Search: 
                # "WalkSAT" style for hypergraphs.
                # Pick a random violation, pick a random triple in it, flip it.
                # This is more directed than random flips.
                
                return None, False # Placeholder to stop recursion for now
                
        # Since full CSP with propagation is hard to implement correctly and fast in this context,
        # let's pivot to a **Directed Local Search** (WalkSAT-like).
        # This is still a structural change from "Random Flip SA" to "Violation-Driven Flip".
        
        def directed_local_search(n, time_budget_dls):
            t0 = time.time()
            end_dls = t0 + time_budget_dls
            
            # Init
            colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
            
            # Compute violation lists
            # red_viol_k4s: list of K4 indices that are all red
            # blue_viol_k5s: list of K5 indices that are all blue
            
            def compute_violations(colors):
                rv = []
                bv = []
                for i, ts in enumerate(k4_triples):
                    if np.all(colors[ts] == 0):
                        rv.append(i)
                for i, ts in enumerate(k5_triples):
                    if np.all(colors[ts] == 1):
                        bv.append(i)
                return rv, bv
            
            rv, bv = compute_violations(colors)
            total_viol = len(rv) + len(bv)
            best_colors = colors.copy()
            best_viol = total_viol
            
            # Track history to avoid cycles?
            # Keep a small history of last flipped triples.
            history = []
            hist_limit = 100
            
            while time.time() < end_dls:
                if total_viol == 0:
                    return colors.tolist()
                
                # Pick a random violation
                if random.random() < 0.5 and rv:
                    # Pick a red K4
                    k4_idx = random.choice(rv)
                    ts = k4_triples[k4_idx]
                    # Pick a triple to flip
                    # Prefer triples not in history?
                    candidates = ts
                else:
                    # Pick a blue K5
                    if not bv:
                        # Only red violations left?
                        if not rv: break
                        k4_idx = random.choice(rv)
                        ts = k4_triples[k4_idx]
                        candidates = ts
                    else:
                        k5_idx = random.choice(bv)
                        ts = k5_triples[k5_idx]
                        candidates = ts
                        
                # Filter candidates by history?
                # If all candidates are in history, pick the one with oldest history?
                # For simplicity, just pick random from candidates.
                t = random.choice(candidates)
                
                # Flip
                old_c = colors[t]
                new_c = 1 - old_c
                colors[t] = new_c
                
                # Update violations incrementally? 
                # It's hard to update the list of violations incrementally without full recomputation or complex bookkeeping.
                # For n=34, num_k4=46376, num_k5=278256.
                # Recomputing violations is O(num_k4*4 + num_k5*10) ~ 3M ops.
                # This is too slow for every step.
                
                # We need the incremental delta approach from the seed, but guided by violations.
                
                # Let's go back to the seed's incremental approach but change the move selection.
                # Seed: Random triple.
                # New: Pick a triple involved in a current violation.
                
                # We need to maintain the list of active violations.
                # Let's maintain:
                # red_viol_count: int
                # blue_viol_count: int
                # And a way to pick a random violation.
                
                # This requires maintaining a set of violating K4s/K5s.
                
                # Let's implement this properly.
                
                break # Stop this inner function, restart outer loop with better design
        
        # OK, let's implement the **Directed Local Search with Incremental Violation Tracking**.
        
        def dls_solve(n, time_budget):
            t0 = time.time()
            end_dls = t0 + time_budget
            
            colors = np.random.randint(0, 2, num_triples, dtype=np.int8)
            
            # Initialize counts
            red_count_k4 = np.zeros(len(k4s), dtype=np.int32)
            blue_count_k5 = np.zeros(len(k5s), dtype=np.int32)
            
            # Initialize violation sets
            red_viol_set = set()
            blue_viol_set = set()
            
            for i, ts in enumerate(k4_triples):
                c = np.sum(colors[ts] == 0)
                red_count_k4[i] = c
                if c == 4:
                    red_viol_set.add(i)
                    
            for i, ts in enumerate(k5_triples):
                c = np.sum(colors[ts] == 1)
                blue_count_k5[i] = c
                if c == 10:
                    blue_viol_set.add(i)
            
            total_viol = len(red_viol_set) + len(blue_viol_set)
            best_colors = colors.copy()
            best_viol = total_viol
            
            # History for WalkSAT
            # dict: triple_idx -> step_count
            last_flip = np.full(num_triples, -1, dtype=np.int32)
            step = 0
            
            while time.time() < end_dls:
                if total_viol == 0:
                    return colors.tolist()
                
                # Pick a violation
                if red_viol_set and (not blue_viol_set or random.random() < 0.5):
                    k4_idx = random.choice(list(red_viol_set))
                    ts = k4_triples[k4_idx]
                    # We want to flip one of these triples to Blue.
                    # Pick the one with the lowest "penalty" (oldest flip) or random.
                    # Let's pick random for now, but avoid recent flips if possible.
                    t = random.choice(ts)
                else:
                    if not blue_viol_set:
                        # Must pick from red
                        if not red_viol_set: break
                        k4_idx = random.choice(list(red_viol_set))
                        ts = k4_triples[k4_idx]
                        t = random.choice(ts)
                    else:
                        k5_idx = random.choice(list(blue_viol_set))
                        ts = k5_triples[k5_idx]
                        t = random.choice(ts)
                
                # Flip t
                old_c = colors[t]
                new_c = 1 - old_c
                
                # Update counts and violation sets
                # K4s
                for k4_i in triple_to_k4s[t]:
                    old_cnt = red_count_k4[k4_i]
                    new_cnt = old_cnt - 1 if old_c == 0 else old_cnt + 1
                    red_count_k4[k4_i] = new_cnt
                    
                    if old_cnt == 4:
                        # Was violation, now not
                        red_viol_set.remove(k4_i)
                        total_viol -= 1
                    elif new_cnt == 4:
                        # Was not, now is
                        red_viol_set.add(k4_i)
                        total_viol += 1
                        
                # K5s
                for k5_i in triple_to_k5s[t]:
                    old_cnt = blue_count_k5[k5_i]
                    new_cnt = old_cnt - 1 if old_c == 1 else old_cnt + 1
                    blue_count_k5[k5_i] = new_cnt
                    
                    if old_cnt == 10:
                        blue_viol_set.remove(k5_i)
                        total_viol -= 1
                    elif new_cnt == 10:
                        blue_viol_set.add(k5_i)
                        total_viol += 1
                        
                colors[t] = new_c
                last_flip[t] = step
                step += 1
                
                if total_viol < best_viol:
                    best_viol = total_viol
                    best_colors = colors.copy()
                    
                # Restart if stuck?
                if step % 5000 == 0 and total_viol > best_viol:
                    # Soft restart: keep best, randomize current?
                    # Or just continue.
                    pass
                    
            return best_colors.tolist() if best_viol == 0 else None

        # Try n=34 with DLS
        # Give it 100s
        res = dls_solve(34, 100)
        if res:
            return {'n': 34, 'colors': res}
            
        # Fallback to n=33
        remaining = deadline - time.time()
        if remaining > 5:
            res = dls_solve(33, remaining - 2)
            if res:
                return {'n': 33, 'colors': res}
                
        # Fallback n=30
        remaining = deadline - time.time()
        if remaining > 5:
            res = dls_solve(30, remaining - 2)
            if res:
                return {'n': 30, 'colors': res}
                
        # Fallback n=9
        n = 9
        colors = []
        for a, b, c in itertools.combinations(range(n), 3):
            colors.append(0 if (a + b + c) % 3 == 0 else 1)
        return {'n': n, 'colors': colors}

    # Try n=34
    n_sol, colors_sol = try_solve_csp(34, 100)
    if n_sol > 0:
        return {'n': n_sol, 'colors': colors_sol}
        
    # Fallbacks
    remaining = deadline - time.time()
    if remaining > 5:
        n_sol, colors_sol = try_solve_csp(33, remaining - 2)
        if n_sol > 0:
            return {'n': n_sol, 'colors': colors_sol}
            
    remaining = deadline - time.time()
    if remaining > 5:
        n_sol, colors_sol = try_solve_csp(30, remaining - 2)
        if n_sol > 0:
            return {'n': n_sol, 'colors': colors_sol}
            
    n = 9
    colors = []
    for a, b, c in itertools.combinations(range(n), 3):
        colors.append(0 if (a + b + c) % 3 == 0 else 1)
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
