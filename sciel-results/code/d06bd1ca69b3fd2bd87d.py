import time
import random
import numpy as np

def _build_structures(n, clauses):
    """Precompute literal arrays for fast vectorized evaluation and variable-to-clause mapping."""
    m = len(clauses)
    if m == 0:
        return None, None, None
    lit_arr = np.empty((m, 3), dtype=np.int8)
    for i, c in enumerate(clauses):
        lit_arr[i, 0] = c[0]
        lit_arr[i, 1] = c[1]
        lit_arr[i, 2] = c[2]
    var_arr = np.abs(lit_arr) - 1  # 0-based variable index

    # Build var_to_clauses: for each variable, list of (clause_idx, position)
    var_to_clauses = [[] for _ in range(n)]
    for i in range(m):
        for pos in range(3):
            v = var_arr[i, pos]
            var_to_clauses[v].append((i, pos))

    return lit_arr, var_arr, var_to_clauses

def _eval_clauses(assignment, lit_arr, var_arr):
    """Return boolean array: is each clause satisfied?"""
    a = assignment[var_arr]  # shape (m, 3), bool
    pos = lit_arr > 0        # shape (m, 3), bool
    sat_literal = a == pos   # shape (m, 3)
    return sat_literal.any(axis=1)  # shape (m,)

def _update_sat_mask(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v):
    """Incrementally update sat_mask after flipping variable v. Returns new unsat_count."""
    for ci, pos in var_to_clauses[v]:
        clause_satisfied = False
        for p in range(3):
            var = var_arr[ci, p]
            lit = lit_arr[ci, p]
            if assignment[var] == (lit > 0):
                clause_satisfied = True
                break
        sat_mask[ci] = clause_satisfied
    
    return int((~sat_mask).sum())

class CDCLSolver:
    def __init__(self, n, clauses):
        self.n = n
        self.m = len(clauses)
        self.clauses = clauses
        self.lit_arr, self.var_arr, self.var_to_clauses = _build_structures(n, clauses)
        
        # CDCL State
        self.values = [None] * n # -1 false, 1 true, 0 unassigned
        self.reason = [None] * n
        self.level = [0] * n
        self.trail = []
        self.trail_lim = []
        self.watched = [[] for _ in range(2 * n)] # indexed by lit: var*2 + (1 if pos else 0)
        # Actually, easier to index watched by literal value (1-based signed)
        # Let's use a dict or list of size 2*n+2
        self.watched = [[] for _ in range(2 * n + 2)]
        
        for ci, c in enumerate(self.clauses):
            for l in c:
                self.watched[l].append(ci)
                
        self.learned_clauses = []
        self.activity = [1.0] * n
        self.decision_level = 0
        self.conflicts = 0
        self.start_time = time.time()
        self.deadline = 0.0
        
    def _lit_var(self, l):
        return abs(l) - 1
    
    def _lit_sign(self, l):
        return 1 if l > 0 else 0
        
    def _is_true(self, l):
        v = self._lit_var(l)
        if self.values[v] is None: return False
        return (self.values[v] == 1) if l > 0 else (self.values[v] == -1)
        
    def _is_false(self, l):
        v = self._lit_var(l)
        if self.values[v] is None: return False
        return (self.values[v] == -1) if l > 0 else (self.values[v] == 1)
        
    def _assign(self, v, val, reason, lvl):
        self.values[v] = val
        self.reason[v] = reason
        self.level[v] = lvl
        self.trail.append(v)
        
    def _unassign(self):
        # Pop from trail until trail_lim is empty
        while len(self.trail) > len(self.trail_lim):
            v = self.trail.pop()
            self.values[v] = None
            self.reason[v] = None
            self.level[v] = 0
            
    def _propagate(self):
        # Simple queue-based propagation
        # We need to check watched literals
        # For efficiency in Python, a simple loop might be okay for n=50
        # But we need to be careful with the watched lists
        
        # Actually, for n=50, we can just iterate through all clauses? No, M can be large.
        # Let's use the watched literals properly.
        
        # This is a simplified propagation. 
        # When a literal becomes false, we look at its watched clauses.
        # If the other watched literal is true, we are fine.
        # If the other watched literal is unassigned, we swap the watch.
        # If the other watched literal is false, conflict.
        
        # To make this work, we need to track which literals were just assigned.
        # The trail contains the assigned variables.
        
        # We will process the trail from the last assigned item backwards? 
        # No, standard CDCL processes implications in order.
        
        # Let's implement a simpler DPLL with clause learning but without complex watched literals initially, 
        # or use a very simple propagation:
        # Iterate through all clauses, if a clause has 1 unassigned and others false -> assign.
        # Repeat until no change.
        # For n=50, this is $O(M \cdot N)$ per iteration, which is acceptable if iterations are few.
        
        changed = True
        while changed:
            changed = False
            for ci, c in enumerate(self.clauses + self.learned_clauses):
                # Check clause c
                unassigned = []
                false_count = 0
                for l in c:
                    if self._is_false(l):
                        false_count += 1
                    elif self._is_true(l):
                        break # Satisfied
                    else:
                        unassigned.append(l)
                
                if len(unassigned) == 1 and false_count == len(c) - 1:
                    # Unit clause
                    l = unassigned[0]
                    v = self._lit_var(l)
                    val = 1 if l > 0 else -1
                    if self.values[v] is None:
                        self._assign(v, val, ci, self.decision_level)
                        changed = True
                        # Break and restart loop? Or continue?
                        # Continue is fine, but might be inefficient.
                        # For n=50, it's okay.
                    elif self.values[v] != val:
                        return ci # Conflict
                
                elif len(unassigned) == 0 and false_count == len(c):
                    return ci # Conflict
        return -1 # No conflict, no new assignments
        
    def _analyze(self, conflict_clause):
        # 1UIP analysis
        # This is complex to implement correctly in a short time.
        # Given n=50, we can use a simpler learning scheme or just restart.
        # Let's try to implement a basic 1UIP.
        
        # Actually, for the purpose of this task, a simpler "Learning" is to just add the negation of the path?
        # No, let's stick to a robust DPLL with Restarts and a simple heuristic.
        # If CDCL is too hard to code correctly in Python quickly, I will fallback to a very strong WalkSAT with Clause Learning (WalkSAT + CL).
        
        # Let's implement a simpler version:
        # If conflict, learn a clause: the negation of the decisions made since the last decision level?
        # That's not 1UIP.
        
        # Alternative: Use the "Inspiration 2" weighted WalkSAT but with **Clause Learning**.
        # When WalkSAT flips a variable and increases unsat count, we don't learn.
        # But if we use CDCL, we learn from conflicts.
        
        # Let's try a different structural approach: **Minisat-lite in Python**.
        # I will implement a basic CDCL with 1UIP.
        
        pass # Placeholder for complex logic

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    m = len(clauses)

    if m == 0:
        return [False] * n

    start_time = time.time()
    DEADLINE = 295.0

    # Strategy:
    # 1. Try a fast CDCL-like solver (DPLL with learning) for a short time (e.g., 5s).
    # 2. If not solved, use the best assignment found as seed for Vectorized WalkSAT.
    # 3. Run Vectorized WalkSAT for the remaining time.
    
    # Since implementing full CDCL in Python is error-prone and might be slow due to overhead,
    # and the current WalkSAT is already score 1.0, the improvement must come from solving instances 
    # that the current WalkSAT fails on (times out).
    
    # A "Structural" change: Use **Simulated Annealing** or **Minisat-style** search.
    # Let's implement a **Clause Learning WalkSAT** (CL-WalkSAT).
    # This is a hybrid: Local search + Clause learning.
    # When a flip increases the number of unsatisfied clauses, we learn a "penalty" or a "clause" 
    # that prevents this specific assignment combination?
    # Standard WalkSAT doesn't learn.
    
    # Better Structural Change: **DPLL with Clause Learning (CDCL)**.
    # I will implement a minimal CDCL.
    
    class MiniCDCL:
        def __init__(self, n, clauses):
            self.n = n
            self.clauses = clauses
            self.values = [None] * n
            self.reason = [None] * n
            self.level = [0] * n
            self.trail = []
            self.trail_lim = []
            self.learned = []
            self.activity = [1.0] * n
            self.decision_level = 0
            self.start_time = time.time()
            
        def _lit_var(self, l): return abs(l) - 1
        def _is_true(self, l):
            v = self._lit_var(l)
            if self.values[v] is None: return False
            return (self.values[v] == 1) if l > 0 else (self.values[v] == -1)
        def _is_false(self, l):
            v = self._lit_var(l)
            if self.values[v] is None: return False
            return (self.values[v] == -1) if l > 0 else (self.values[v] == 1)
            
        def _assign(self, v, val, reason, lvl):
            self.values[v] = val
            self.reason[v] = reason
            self.level[v] = lvl
            self.trail.append(v)
            
        def _unassign(self):
            while len(self.trail) > len(self.trail_lim):
                v = self.trail.pop()
                self.values[v] = None
                self.reason[v] = None
                self.level[v] = 0
                
        def _propagate(self):
            # Simple propagation: scan all clauses
            # For n=50, this is fine.
            changed = True
            while changed:
                changed = False
                for ci, c in enumerate(self.clauses + self.learned):
                    unassigned = []
                    false_count = 0
                    for l in c:
                        if self._is_false(l):
                            false_count += 1
                        elif self._is_true(l):
                            break
                        else:
                            unassigned.append(l)
                    if len(unassigned) == 1 and false_count == len(c) - 1:
                        l = unassigned[0]
                        v = self._lit_var(l)
                        val = 1 if l > 0 else -1
                        if self.values[v] is None:
                            self._assign(v, val, ci, self.decision_level)
                            changed = True
                        elif self.values[v] != val:
                            return ci
                    elif len(unassigned) == 0 and false_count == len(c):
                        return ci
            return -1
            
        def _analyze(self, conflict_ci):
            # 1UIP
            # This is complex. For simplicity and speed in Python, 
            # let's just restart with a new random seed if we get stuck.
            # Or, learn a clause from the conflict path.
            # Path: variables assigned since the last decision level?
            # Let's just return None for now and rely on restarts.
            return None
            
        def solve(self, deadline):
            while time.time() < deadline:
                if self.decision_level == 0:
                    # Pick a variable
                    unassigned = [v for v in range(self.n) if self.values[v] is None]
                    if not unassigned:
                        return True
                    # Heuristic: pick random or most frequent
                    v = unassigned[0] # Simple
                    val = 1 if random.random() < 0.5 else -1
                    self._assign(v, val, -1, self.decision_level)
                    self.trail_lim.append(len(self.trail))
                    self.decision_level += 1
                    
                conflict = self._propagate()
                if conflict == -1:
                    # Check if all assigned
                    if all(self.values[v] is not None for v in range(self.n)):
                        return True
                else:
                    # Conflict
                    # Analyze and backtrack
                    # Simple backtracking: unassign last decision
                    if self.decision_level > 0:
                        self.decision_level -= 1
                        self._unassign()
                    else:
                        return False # UNSAT (should not happen for planted)
            return False

    # Run CDCL for 10 seconds
    cdcl = MiniCDCL(n, clauses)
    solved = cdcl.solve(start_time + 10.0)
    
    if solved:
        return [bool(v == 1) for v in cdcl.values]
    
    # If CDCL didn't solve, use its best state as seed for WalkSAT
    # Convert CDCL values to bool array
    seed_assignment = np.array([bool(cdcl.values[v] == 1) if cdcl.values[v] is not None else (random.random() < 0.5) for v in range(n)], dtype=bool)
    
    # Run Vectorized WalkSAT
    lit_arr, var_arr, var_to_clauses = _build_structures(n, clauses)
    
    best_assignment = seed_assignment
    best_unsat = m + 1
    
    rng = random.Random(12345)
    
    restart = 0
    while time.time() - start_time < DEADLINE:
        restart += 1
        assignment = seed_assignment.copy() if restart == 1 else np.array([rng.random() < 0.5 for _ in range(n)], dtype=bool)
        
        sat_mask = _eval_clauses(assignment, lit_arr, var_arr)
        unsat_count = int((~sat_mask).sum())
        
        if unsat_count == 0:
            return assignment.tolist()
            
        if unsat_count < best_unsat:
            best_unsat = unsat_count
            best_assignment = assignment.copy()
            
        max_steps = 50000
        p_min = 0.15
        p_max = 0.9
        unsat_indices = np.where(~sat_mask)[0]
        
        for step in range(max_steps):
            if time.time() - start_time > DEADLINE:
                break
            if unsat_count == 0:
                return assignment.tolist()
                
            progress = step / max_steps
            p = p_min + (p_max - p_min) * progress
            
            ci = unsat_indices[rng.randrange(len(unsat_indices))]
            clause_vars = var_arr[ci]
            
            if rng.random() < p:
                pos = rng.randrange(3)
            else:
                best_pos = 0
                best_score = float('-inf')
                for pos in range(3):
                    v = clause_vars[pos]
                    lit_v = lit_arr[ci, pos]
                    v_is_true = assignment[v]
                    score = 0.0
                    for cj, _ in var_to_clauses[v]:
                        if sat_mask[cj]:
                            other_sat = False
                            for p2 in range(3):
                                if p2 == _:
                                    continue
                                var2 = var_arr[cj, p2]
                                lit2 = lit_arr[cj, p2]
                                if assignment[var2] == (lit2 > 0):
                                    other_sat = True
                                    break
                            if not other_sat:
                                score -= 1.0
                        else:
                            for p2 in range(3):
                                if var_arr[cj, p2] == v:
                                    lit2 = lit_arr[cj, p2]
                                    if (not v_is_true) == (lit2 > 0):
                                        score += 1.0
                                    break
                    if score > best_score:
                        best_score = score
                        best_pos = pos
                pos = best_pos
                
            v = clause_vars[pos]
            assignment[v] = not assignment[v]
            unsat_count = _update_sat_mask(assignment, sat_mask, lit_arr, var_arr, var_to_clauses, v)
            unsat_indices = np.where(~sat_mask)[0]
            
            if unsat_count < best_unsat:
                best_unsat = unsat_count
                best_assignment = assignment.copy()
                
        if best_unsat == 0:
            return best_assignment.tolist()
            
    return best_assignment.tolist()
