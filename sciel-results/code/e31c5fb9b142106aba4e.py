import time
import random

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    
    # Deadline guard: 330s limit, use 320s to be safe
    start_time = time.time()
    deadline = start_time + 320.0
    
    rng = random.Random(0)
    
    # Pre-process clauses for fast access
    # clause_lits[i] is the list of literals in clause i
    # var_clauses[v] is the list of clause indices containing variable v (0-indexed var)
    # var_sign[v] is the sign of the literal in that clause? No, just need to check value.
    
    m = len(clauses)
    clause_lits = []
    var_clauses = [[] for _ in range(n)]
    
    for i, clause in enumerate(clauses):
        lits = []
        for l in clause:
            v = abs(l) - 1
            var_clauses[v].append(i)
            lits.append(l)
        clause_lits.append(lits)
    
    # State
    # assignment[v] is True if v is True, False if v is False
    # clause_unsat_count[i] is the number of literals in clause i that are currently False (unsatisfied)
    # unsat_set: set of clause indices where clause_unsat_count[i] > 0
    
    def init_state():
        assignment = [rng.random() < 0.5 for _ in range(n)]
        clause_unsat_count = [0] * m
        unsat_set = set()
        
        for i, lits in enumerate(clause_lits):
            count = 0
            for l in lits:
                v = abs(l) - 1
                is_true = assignment[v]
                lit_is_true = (l > 0 and is_true) or (l < 0 and not is_true)
                if not lit_is_true:
                    count += 1
            clause_unsat_count[i] = count
            if count > 0:
                unsat_set.add(i)
        return assignment, clause_unsat_count, unsat_set

    def flip_var(v, assignment, clause_unsat_count, unsat_set):
        """
        Flip variable v. Update clause_unsat_count and unsat_set.
        Returns the number of clauses that changed status (for potential heuristic use, though not strictly needed here).
        """
        # Determine the new value
        new_val = not assignment[v]
        assignment[v] = new_val
        
        # Update all clauses containing v
        for i in var_clauses[v]:
            lits = clause_lits[i]
            # Find the literal in this clause corresponding to v
            # We know v is in this clause. We need to know the sign.
            # We can iterate, but it's better to store sign.
            # Let's re-structure: store (var, sign) in clause_lits? Or just look it up.
            # For speed, let's assume we can find it quickly. 
            # Actually, let's pre-compute a map: clause_var_sign[i] = dict {var: sign}
            
            pass # Placeholder for logic below

    # To make flip_var O(D), we need to know the sign of v in each clause quickly.
    # Let's pre-compute: clause_var_info[i] is a dict {var: sign}
    clause_var_info = []
    for i, clause in enumerate(clauses):
        info = {}
        for l in clause:
            v = abs(l) - 1
            info[v] = 1 if l > 0 else -1
        clause_var_info.append(info)
    
    def flip_var(v, assignment, clause_unsat_count, unsat_set):
        new_val = not assignment[v]
        assignment[v] = new_val
        
        for i in var_clauses[v]:
            sign = clause_var_info[i][v]
            # If sign is 1, literal is v. If sign is -1, literal is not v.
            # lit_is_true = (sign == 1 and new_val) or (sign == -1 and not new_val)
            # lit_is_true = (sign == 1) == new_val
            
            old_val = not new_val
            old_lit_true = (sign == 1) == old_val
            new_lit_true = (sign == 1) == new_val
            
            if old_lit_true and not new_lit_true:
                # Variable was satisfying, now unsatisfying
                clause_unsat_count[i] += 1
                if clause_unsat_count[i] == 1:
                    unsat_set.add(i)
            elif not old_lit_true and new_lit_true:
                # Variable was unsatisfying, now satisfying
                clause_unsat_count[i] -= 1
                if clause_unsat_count[i] == 0:
                    unsat_set.remove(i)
    
    # Main Search Loop
    best_assignment = None
    best_unsat_count = m + 1
    
    max_steps_per_restart = 5000
    max_restarts = 10000 # High enough to cover time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        assignment, clause_unsat_count, unsat_set = init_state()
        
        if not unsat_set:
            return assignment
        
        if len(unsat_set) < best_unsat_count:
            best_unsat_count = len(unsat_set)
            best_assignment = assignment[:]
        
        steps = 0
        stuck_counter = 0
        last_unsat_size = len(unsat_set)
        
        while unsat_set:
            if time.time() > deadline:
                if len(unsat_set) < best_unsat_count:
                    best_unsat_count = len(unsat_set)
                    best_assignment = assignment[:]
                return best_assignment if best_assignment else assignment
            
            # Pick a random unsatisfied clause
            # Converting set to list every time is O(U). 
            # Alternative: maintain a list of unsat clauses and a pointer, or just use list(unsat_set) if U is small.
            # For 50 vars, U is usually small.
            unsat_list = list(unsat_set)
            if not unsat_list: break # Safety
            
            clause_idx = rng.choice(unsat_list)
            
            # Heuristic: Pick a variable to flip
            # With prob p, pick random. Else, pick best (minimizes new unsat count).
            p = 0.35 # Standard WalkSAT parameter
            
            lits = clause_lits[clause_idx]
            vars_in_clause = [abs(l) - 1 for l in lits]
            
            if rng.random() < p:
                v_to_flip = rng.choice(vars_in_clause)
            else:
                # Find variable that causes least damage
                best_v = None
                min_damage = float('inf')
                for v in vars_in_clause:
                    # Calculate damage: how many clauses containing v will become unsatisfied if we flip v?
                    # We need to check each clause containing v.
                    damage = 0
                    for ci in var_clauses[v]:
                        sign = clause_var_info[ci][v]
                        # If we flip v, does this clause become unsatisfied?
                        # It becomes unsatisfied if:
                        # 1. It is currently satisfied (count 0) -> impossible since we only look at vars in unsat clause? 
                        #    No, v is in an unsat clause, but other clauses containing v might be sat.
                        #    If a clause is currently sat (count 0), and v was satisfying it, flipping v makes it unsat.
                        # 2. It is currently unsat, but flipping v doesn't help (it remains unsat). This is not "damage" in the sense of creating NEW unsat clauses, but it doesn't reduce count.
                        #    Standard WalkSAT "best" flip minimizes the number of clauses that become unsatisfied (i.e., were sat and become unsat).
                        
                        # Check if clause ci is currently satisfied
                        if clause_unsat_count[ci] == 0:
                            # It is satisfied.
                            # Check if v is currently satisfying it.
                            # If v is satisfying it, flipping it will make it unsatisfied.
                            # lit_true = (sign == 1) == assignment[v]
                            if (sign == 1) == assignment[v]:
                                damage += 1
                    if damage < min_damage:
                        min_damage = damage
                        best_v = v
                v_to_flip = best_v
            
            flip_var(v_to_flip, assignment, clause_unsat_count, unsat_set)
            steps += 1
            
            # Restart if stuck
            if len(unsat_set) < last_unsat_size:
                stuck_counter = 0
                last_unsat_size = len(unsat_set)
            else:
                stuck_counter += 1
                if stuck_counter > 500:
                    break
        
        if not unsat_set:
            return assignment

    return best_assignment if best_assignment else [False] * n
