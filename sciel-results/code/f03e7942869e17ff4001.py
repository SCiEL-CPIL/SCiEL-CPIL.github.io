import random
import time

def solve(instance):
    n = instance["n"]
    clauses = instance["clauses"]
    num_clauses = len(clauses)
    
    cnf = []
    for c in clauses:
        cnf.append([l for l in c])
        
    deadline = time.time() + 290
    rng = random.Random(42)
    
    var_clauses = [[] for _ in range(n)]
    for ci, c in enumerate(cnf):
        for l in c:
            var = abs(l) - 1
            var_clauses[var].append((ci, l))
            
    def is_sat(lit, asgn):
        var = abs(lit) - 1
        val = asgn[var]
        if val == -1:
            return False
        return (val == 1) if lit > 0 else (val == 0)
        
    def is_fals(lit, asgn):
        var = abs(lit) - 1
        val = asgn[var]
        if val == -1:
            return False
        return (val == 0) if lit > 0 else (val == 1)
        
    def propagate(asgn):
        queue = []
        for ci, c in enumerate(cnf):
            unassigned = []
            for l in c:
                if asgn[abs(l)-1] == -1:
                    unassigned.append(l)
            if len(unassigned) == 0:
                sat = any(is_sat(l, asgn) for l in c)
                if not sat:
                    return False
            elif len(unassigned) == 1:
                queue.append((ci, unassigned[0]))
                
        while queue:
            ci, unit_lit = queue.pop()
            var = abs(unit_lit) - 1
            if asgn[var] != -1:
                if is_fals(unit_lit, asgn):
                    return False
                continue
            
            asgn[var] = 1 if unit_lit > 0 else 0
            
            for (ci2, lit2) in var_clauses[var]:
                if ci2 == ci: continue
                unassigned = []
                sat = False
                for l in cnf[ci2]:
                    if is_sat(l, asgn):
                        sat = True
                        break
                    if asgn[abs(l)-1] == -1:
                        unassigned.append(l)
                if sat:
                    continue
                if len(unassigned) == 0:
                    return False
                if len(unassigned) == 1:
                    queue.append((ci2, unassigned[0]))
                    
        return True

    def dpll(asgn, depth, time_limit):
        if time.time() > time_limit:
            return "TIMEOUT"
        
        asgn = asgn[:]
        
        if not propagate(asgn):
            return False
            
        if -1 not in asgn:
            return True
            
        var_polarity = [0] * n
        for ci, c in enumerate(cnf):
            for l in c:
                if asgn[abs(l)-1] == -1:
                    var = abs(l) - 1
                    sign = 1 if l > 0 else -1
                    if var_polarity[var] == 0:
                        var_polarity[var] = sign
                    elif var_polarity[var] != sign:
                        var_polarity[var] = 2
        
        for var in range(n):
            if asgn[var] == -1 and var_polarity[var] in [1, -1]:
                asgn[var] = 1 if var_polarity[var] == 1 else 0
                if not propagate(asgn):
                    asgn[var] = -1
                    continue
                res = dpll(asgn, depth+1, time_limit)
                if res == "TIMEOUT":
                    return "TIMEOUT"
                if res:
                    return True
                asgn[var] = -1
        
        best_clause = None
        min_len = 100
        for ci, c in enumerate(cnf):
            unassigned = [l for l in c if asgn[abs(l)-1] == -1]
            if len(unassigned) > 0 and len(unassigned) < min_len:
                min_len = len(unassigned)
                best_clause = c
                if min_len == 1: break
                
        if best_clause is None:
            return True
            
        var = abs(best_clause[0]) - 1
        
        asgn[var] = 1
        res = dpll(asgn, depth+1, time_limit)
        if res == "TIMEOUT":
            return "TIMEOUT"
        if res:
            return True
            
        asgn[var] = 0
        res = dpll(asgn, depth+1, time_limit)
        if res == "TIMEOUT":
            return "TIMEOUT"
        if res:
            return True
            
        return False

    dpll_limit = time.time() + 10
    res = dpll([-1]*n, 0, dpll_limit)
    
    if res == True:
        return [b == 1 for b in asgn]
        
    assignment = [rng.random() < 0.5 for _ in range(n)]
    
    def count_sat(asgn):
        cnt = 0
        for c in cnf:
            if any(is_sat(l, asgn) for l in c):
                cnt += 1
        return cnt
        
    def walksat(asgn, max_iters, time_limit):
        cur_count = count_sat(asgn)
        if cur_count == num_clauses:
            return asgn, cur_count
        
        no_improve = 0
        for _ in range(max_iters):
            if time.time() > time_limit:
                break
                
            unsat = []
            for i, c in enumerate(cnf):
                if not any(is_sat(l, asgn) for l in c):
                    unsat.append(i)
                    
            if not unsat:
                return asgn, num_clauses
                
            ci = rng.choice(unsat)
            c = cnf[ci]
            
            scores = []
            for l in c:
                var = abs(l) - 1
                gain = 0
                for (ci2, _) in var_clauses[var]:
                    c2 = cnf[ci2]
                    if any(is_sat(l2, asgn) for l2 in c2):
                        continue
                    if any(l2 == l for l2 in c2):
                        gain += 1
                    else:
                        gain -= 1
                scores.append(gain)
                
            max_s = max(scores)
            candidates = [i for i, s in enumerate(scores) if s == max_s]
            idx = rng.choice(candidates)
            l = c[idx]
            var = abs(l) - 1
            asgn[var] = (l > 0)
            
            new_count = count_sat(asgn)
            if new_count > cur_count:
                cur_count = new_count
                no_improve = 0
                if cur_count == num_clauses:
                    return asgn, cur_count
            else:
                no_improve += 1
                
            if no_improve > 1000:
                break
                
        return asgn, cur_count
        
    best_asgn = assignment[:]
    best_count = count_sat(best_asgn)
    
    while time.time() < deadline - 5:
        assignment = [rng.random() < 0.5 for _ in range(n)]
        assignment, cur_count = walksat(assignment, 10000, deadline - 5)
        if cur_count > best_count:
            best_count = cur_count
            best_asgn = assignment[:]
        if best_count == num_clauses:
            break
            
    return [b for b in best_asgn]
