"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

class DPLL_Solver:
    def __init__(self, n, clauses):
        self.n = n
        self.clauses = clauses
        self.m = len(clauses)
        self.var_to_clauses = [[] for _ in range(n)]
        for i, c in enumerate(clauses):
            for lit in c:
                var_idx = abs(lit) - 1
                self.var_to_clauses[var_idx].append(i)
                
        self.assignment = [None] * n
        self.active_clauses = set(range(self.m))
        
    def is_satisfied(self, ci):
        c = self.clauses[ci]
        for lit in c:
            var_idx = abs(lit) - 1
            val = self.assignment[var_idx]
            if val is None:
                continue
            if (lit > 0 and val) or (lit < 0 and not val):
                return True
        return False

    def unit_propagation(self):
        changed = True
        while changed:
            changed = False
            units = []
            for ci in list(self.active_clauses):
                c = self.clauses[ci]
                unassigned_lits = []
                satisfied = False
                for lit in c:
                    var_idx = abs(lit) - 1
                    val = self.assignment[var_idx]
                    if val is None:
                        unassigned_lits.append(lit)
                    else:
                        if (lit > 0 and val) or (lit < 0 and not val):
                            satisfied = True
                            break
                if satisfied:
                    self.active_clauses.remove(ci)
                elif len(unassigned_lits) == 0:
                    return False
                elif len(unassigned_lits) == 1:
                    units.append((ci, unassigned_lits[0]))
                    
            for ci, lit in units:
                var_idx = abs(lit) - 1
                val = lit > 0
                if self.assignment[var_idx] is not None:
                    if self.assignment[var_idx] != val:
                        return False
                    continue
                self.assignment[var_idx] = val
                changed = True
                for cj in self.var_to_clauses[var_idx]:
                    if cj in self.active_clauses:
                        if self.is_satisfied(cj):
                            self.active_clauses.remove(cj)
        return True

    def pure_literal_elimination(self):
        changed = True
        while changed:
            changed = False
            for var_idx in range(self.n):
                if self.assignment[var_idx] is not None:
                    continue
                pos_count = 0
                neg_count = 0
                for ci in self.var_to_clauses[var_idx]:
                    if ci not in self.active_clauses:
                        continue
                    c = self.clauses[ci]
                    satisfied_by_other = False
                    for lit in c:
                        if abs(lit) - 1 == var_idx:
                            continue
                        other_var = abs(lit) - 1
                        if self.assignment[other_var] is not None:
                            if (lit > 0 and self.assignment[other_var]) or (lit < 0 and not self.assignment[other_var]):
                                satisfied_by_other = True
                                break
                    if satisfied_by_other:
                        continue
                    if var_idx + 1 in c:
                        pos_count += 1
                    else:
                        neg_count += 1
                if pos_count > 0 and neg_count == 0:
                    self.assignment[var_idx] = True
                    changed = True
                elif neg_count > 0 and pos_count == 0:
                    self.assignment[var_idx] = False
                    changed = True
                if changed:
                    for cj in self.var_to_clauses[var_idx]:
                        if cj in self.active_clauses:
                            if self.is_satisfied(cj):
                                self.active_clauses.remove(cj)
                    break
        return changed

    def dpll(self):
        if not self.active_clauses:
            return True
            
        for ci in self.active_clauses:
            c = self.clauses[ci]
            unassigned = 0
            for lit in c:
                var_idx = abs(lit) - 1
                if self.assignment[var_idx] is None:
                    unassigned += 1
            if unassigned == 0:
                return False
                
        if not self.unit_propagation():
            return False
            
        if not self.active_clauses:
            return True
            
        self.pure_literal_elimination()
        if not self.active_clauses:
            return True
            
        # Choose variable
        freq = [0] * self.n
        for ci in self.active_clauses:
            c = self.clauses[ci]
            for lit in c:
                var_idx = abs(lit) - 1
                if self.assignment[var_idx] is None:
                    freq[var_idx] += 1
                    
        best_var = -1
        max_freq = -1
        for i in range(self.n):
            if self.assignment[i] is None and freq[i] > max_freq:
                max_freq = freq[i]
                best_var = i
                
        if best_var == -1:
            return True
            
        # Save state
        old_assignment = self.assignment[:]
        old_active = set(self.active_clauses)
        
        # Try True
        self.assignment[best_var] = True
        if self.unit_propagation():
            if self.dpll():
                return True
                
        # Restore
        self.assignment = old_assignment
        self.active_clauses = old_active
        
        # Try False
        self.assignment[best_var] = False
        if self.unit_propagation():
            if self.dpll():
                return True
                
        # Restore
        self.assignment = old_assignment
        self.active_clauses = old_active
        
        return False

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    
    # Pre-process: remove tautologies and duplicates
    processed_clauses = []
    seen = set()
    for c in clauses:
        taut = False
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                if c[i] == -c[j]:
                    taut = True
                    break
            if taut:
                break
        if taut:
            continue
        norm = tuple(sorted(c))
        if norm not in seen:
            seen.add(norm)
            processed_clauses.append(list(c))
            
    solver = DPLL_Solver(n, processed_clauses)
    if solver.dpll():
        for i in range(n):
            if solver.assignment[i] is None:
                solver.assignment[i] = True
        return solver.assignment
    else:
        return [random.random() < 0.5 for _ in range(n)]
# EVOLVE-BLOCK-END
