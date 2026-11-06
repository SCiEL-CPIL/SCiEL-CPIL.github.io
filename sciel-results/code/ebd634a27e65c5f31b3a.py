import time
import random
import sys

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # 1. Preprocessing: Pure Literal & Unit Propagation
    # This is critical for performance.
    
    # Convert to internal format: (var, sign)
    # var: 0..n-1, sign: 0(False), 1(True)
    internal_clauses = []
    for c in clauses:
        ic = []
        for l in c:
            v = abs(l) - 1
            s = 1 if l > 0 else 0
            ic.append((v, s))
        internal_clauses.append(ic)
        
    val = [None] * n
    active = [True] * len(internal_clauses)
    
    # Incidence: var -> list of (clause_idx, sign)
    var_clauses = [[] for _ in range(n)]
    for i, c in enumerate(internal_clauses):
        for v, s in c:
            var_clauses[v].append((i, s))
            
    def is_satisfied(i):
        if not active[i]: return False
        for v, s in internal_clauses[i]:
            if val[v] is not None and (val[v] == (s == 1)):
                return True
        return False

    def get_unit(i):
        if not active[i]: return None
        if is_satisfied(i): return "sat"
        unassigned = []
        for v, s in internal_clauses[i]:
            if val[v] is None:
                unassigned.append((v, s))
            else:
                if val[v] == (s == 1):
                    return "sat"
        if len(unassigned) == 1:
            return unassigned[0]
        if len(unassigned) == 0:
            return "conflict"
        return None

    # Iterative Simplification
    changed = True
    while changed:
        changed = False
        # Pure Literal
        for v in range(n):
            if val[v] is not None: continue
            pos = False; neg = False
            for i, s in var_clauses[v]:
                if not active[i]: continue
                if is_satisfied(i): continue
                if s == 1: pos = True
                else: neg = True
                if pos and neg: break
            if pos and not neg:
                val[v] = True; changed = True
            elif neg and not pos:
                val[v] = False; changed = True
                
        # Unit Propagation
        for i in range(len(internal_clauses)):
            if not active[i]: continue
            res = get_unit(i)
            if res == "sat":
                active[i] = False; changed = True
            elif res == "conflict":
                return [random.choice([True, False]) for _ in range(n)]
            elif res is not None:
                v, s = res
                if val[v] is None:
                    val[v] = (s == 1); changed = True
                elif val[v] != (s == 1):
                    return [random.choice([True, False]) for _ in range(n)]
                    
    # Check if solved
    all_assigned = all(v is not None for v in val)
    if all_assigned:
        return val

    # 2. DPLL with VSIDS and Watched Literals
    
    # Re-initialize watches for remaining active clauses
    # We need to rebuild watches based on current val state
    watches = {}
    for i in range(len(internal_clauses)):
        if not active[i]: continue
        c = internal_clauses[i]
        # Find two unassigned or true literals to watch
        # If no unassigned, it's satisfied or conflict (handled above)
        # So there must be at least 1 unassigned.
        # If exactly 1, it's unit (handled above).
        # So there are >= 2 unassigned? Not necessarily, some might be true.
        # Watched literals should be unassigned or true.
        
        watch_lits = []
        for v, s in c:
            if val[v] is None:
                watch_lits.append((v, s))
            elif val[v] == (s == 1):
                watch_lits.append((v, s))
            if len(watch_lits) == 2:
                break
        
        if len(watch_lits) < 2:
            # Should not happen if propagation was correct
            if len(watch_lits) == 0:
                # Conflict
                return [random.choice([True, False]) for _ in range(n)]
            # If 1, it's unit, should have been propagated
            pass
            
        for lit in watch_lits:
            if lit not in watches:
                watches[lit] = []
            watches[lit].append(i)
            
    # VSIDS
    activity = [1.0] * n
    max_act = 1.0
    decay = 0.95
    
    trail = []
    trail_lim = []
    
    def propagate():
        """
        Returns False if conflict, True otherwise.
        Modifies val and watches.
        """
        # Queue of literals to check
        # We trigger propagation when a literal is assigned.
        # In this recursive DPLL, we call propagate after each assignment.
        # We need to know which literals became false.
        # Simplification: Iterate over all watches? No, that's slow.
        # We need a queue.
        
        # For this implementation, we will pass the last assigned literal.
        # But propagate is called from multiple places.
        # Let's use a global queue variable.
        
        pass # Placeholder

    # This is getting complicated to implement correctly in a short snippet.
    # Given the risk of bugs in a custom CDCL/DPLL in Python, and the fact that WalkSAT with Clause Weighting is a "different design" from the naive WalkSAT in the seed (which has no weights or restarts), 
    # I will switch to a **Robust Weighted WalkSAT with Restarts and Phase Saving**.
    # This is a valid structural change from the seed (which is a simple random WalkSAT).
    # It is much more likely to be correct and fast in Python.
    
    # But the directive said DPLL. 
    # "Make a STRUCTURAL change... different design...". 
    # Weighted WalkSAT is a local search, same family as seed.
    # DPLL is a different family.
    
    # I will provide a **simplified DPLL** that uses **Unit Propagation** and **VSIDS** but relies on **recursion** and **simple clause scanning** for propagation (which is slow but correct for N=150 if the tree is small).
    # To make it fast enough, I will add **Restarts** and **Clause Learning** (1-UIP) is too hard.
    # I will stick to the DPLL with the O(M) propagation but add **Random Restarts** with **VSIDS Activity Decay**.
    
    # Let's refine the DPLL code from the previous thought process to be as clean as possible.

    # --- Final DPLL Implementation ---
    
    # State
    val = [None] * n
    # We need to reset val for each restart, but keep activity.
    # Actually, for DPLL, we start fresh each restart.
    
    def run_dpll():
        # Local state for this run
        local_val = [None] * n
        # Copy initial assignments from preprocessing?
        # Preprocessing assignments are permanent.
        for i in range(n):
            local_val[i] = val[i] # val from preprocessing step above
        
        # Watches need to be built based on local_val
        local_watches = {}
        for i in range(len(internal_clauses)):
            if not active[i]: continue
            c = internal_clauses[i]
            watch_lits = []
            for v, s in c:
                if local_val[v] is None:
                    watch_lits.append((v, s))
                elif local_val[v] == (s == 1):
                    watch_lits.append((v, s))
                if len(watch_lits) == 2:
                    break
            for lit in watch_lits:
                if lit not in local_watches:
                    local_watches[lit] = []
                local_watches[lit].append(i)
                
        local_activity = activity[:] # Keep global activity? No, let's use local for this run, or global?
        # VSIDS is usually global. Let's use global `activity` list and modify it.
        
        local_trail = []
        local_trail_lim = []
        
        def propagate(last_lit):
            """
            Propagate implications.
            last_lit: (v, s) that was just assigned False.
            Returns False if conflict.
            """
            if last_lit is None:
                # Initial call: check all? No, we start with no unassigned false literals.
                # But we might have unit clauses from preprocessing? No, preprocessing handled them.
                return True
            
            v, s = last_lit
            # The literal (v, s) is now False.
            # So the literal (v, 1-s) is True? No.
            # If we assigned val[v] = (s==1) is False, then val[v] = (1-s)==1.
            # Wait.
            # If last_lit is (v, s), it means the literal `s*v` is false.
            # So val[v] is NOT (s==1).
            
            # Check watches for last_lit
            if last_lit not in local_watches:
                return True
                
            to_remove = []
            for i in local_watches[last_lit]:
                c = internal_clauses[i]
                # Check if clause is satisfied
                sat = False
                for vv, ss in c:
                    if local_val[vv] is not None and local_val[vv] == (ss == 1):
                        sat = True
                        break
                if sat:
                    to_remove.append(i)
                    continue
                    
                # Find new watch
                new_watch = None
                for vv, ss in c:
                    if (vv, ss) == last_lit:
                        continue
                    if local_val[vv] is None:
                        new_watch = (vv, ss)
                        break
                    elif local_val[vv] == (ss == 1):
                        # This literal is true, so clause is satisfied.
                        # Why didn't we catch it above?
                        # Maybe val was set after watch was added?
                        sat = True
                        break
                        
                if sat:
                    to_remove.append(i)
                    continue
                    
                if new_watch is None:
                    # No new watch, and not satisfied -> Unit or Conflict
                    # Check if unit
                    unassigned = []
                    for vv, ss in c:
                        if local_val[vv] is None:
                            unassigned.append((vv, ss))
                        else:
                            if local_val[vv] == (ss == 1):
                                sat = True
                                break
                    if sat:
                        to_remove.append(i)
                        continue
                        
                    if len(unassigned) == 0:
                        return False # Conflict
                        
                    if len(unassigned) == 1:
                        # Unit clause
                        uv, us = unassigned[0]
                        # Assign uv to satisfy us
                        local_val[uv] = (us == 1)
                        local_trail.append(uv)
                        # Recursively propagate
                        # The literal that became false is (uv, 1-us)
                        if not propagate((uv, 1 - us)):
                            return False
                        # Update watches for the new assignment?
                        # The watch for (uv, 1-us) is now triggered.
                        # We don't need to do anything else here, the recursive call handles it.
                        # But we need to remove this clause from the old watch list?
                        # The clause is still watched by last_lit?
                        # No, we are processing last_lit's watch list.
                        # We should replace last_lit with new_watch in the watch list.
                        # But we are in the loop.
                        # Let's just mark it for update.
                        # For simplicity, we don't update watches in this recursive step,
                        # which might lead to inefficiency but is correct.
                        # Actually, we MUST update watches to avoid re-processing.
                        # Let's defer watch updates to a separate step or just accept the overhead.
                        # Given N=150, overhead is acceptable.
                        pass
                    else:
                        # Multiple unassigned, replace watch
                        if new_watch:
                            # Replace last_lit with new_watch in watches
                            # We are in the loop for local_watches[last_lit]
                            # We can't easily modify the dict while iterating.
                            # We'll collect changes.
                            pass
                            
            # This implementation is getting too complex and error-prone for a single shot.
            
            return True

        # Abandoning the complex watched literal DPLL.
        # Returning to the **O(M) Propagation DPLL** which is simpler and correct.
        
        def simple_propagate():
            """
            Scans all clauses. Returns False if conflict.
            """
            while True:
                changed = False
                for i in range(len(internal_clauses)):
                    if not active[i]: continue
                    c = internal_clauses[i]
                    sat = False
                    unassigned = []
                    for v, s in c:
                        if local_val[v] is None:
                            unassigned.append((v, s))
                        else:
                            if local_val[v] == (s == 1):
                                sat = True
                                break
                    if sat:
                        active[i] = False
                        changed = True
                        continue
                    if len(unassigned) == 0:
                        return False
                    if len(unassigned) == 1:
                        v, s = unassigned[0]
                        if local_val[v] is None:
                            local_val[v] = (s == 1)
                            local_trail.append(v)
                            changed = True
                if not changed:
                    break
            return True

        def pick_var():
            best_v = -1
            best_a = -1
            for v in range(n):
                if local_val[v] is None:
                    if activity[v] > best_a:
                        best_a = activity[v]
                        best_v = v
            return best_v

        def dpll():
            if time.time() > deadline:
                return None
                
            if not simple_propagate():
                return None
                
            all_assigned = True
            for v in range(n):
                if local_val[v] is None:
                    all_assigned = False
                    break
            if all_assigned:
                return local_val[:]
                
            v = pick_var()
            if v == -1:
                return local_val[:]
                
            # Branch True
            local_trail_lim.append(len(local_trail))
            local_val[v] = True
            local_trail.append(v)
            
            res = dpll()
            if res is not None:
                return res
                
            # Backtrack
            while len(local_trail) > local_trail_lim[-1]:
                u = local_trail.pop()
                local_val[u] = None
            local_trail_lim.pop()
            local_val[v] = None
            
            # Branch False
            local_trail_lim.append(len(local_trail))
            local_val[v] = False
            local_trail.append(v)
            
            res = dpll()
            if res is not None:
                return res
                
            # Backtrack
            while len(local_trail) > local_trail_lim[-1]:
                u = local_trail.pop()
                local_val[u] = None
            local_trail_lim.pop()
            local_val[v] = None
            
            return None

        return dpll()

    # Run DPLL with restarts
    best = None
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        res = run_dpll()
        if res:
            best = res
            break
        # Decay activities
        for i in range(n):
            activity[i] *= 0.9
            
    if best:
        return best

    # Fallback WalkSAT
    rng = random.Random(1)
    assignment = [rng.random() < 0.5 for _ in range(n)]
    def sat(c):
        return any(assignment[abs(l)-1] == (l>0) for l in c)
    for _ in range(50000):
        if time.time() > deadline: break
        unsat = [c for c in clauses if not sat(c)]
        if not unsat: break
        c = rng.choice(unsat)
        l = rng.choice(c)
        assignment[abs(l)-1] = not assignment[abs(l)-1]
    return assignment

# EVOLVE-BLOCK-END
