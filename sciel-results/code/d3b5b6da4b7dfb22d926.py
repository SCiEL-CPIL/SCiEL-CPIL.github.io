import time
import random

def solve(instance):
    n, clauses = instance["n"], instance["clauses"]
    deadline = time.time() + 285
    
    m = len(clauses)
    cnf = []
    for c in clauses:
        cnf.append(tuple(2*(abs(l)-1) + (0 if l>0 else 1) for l in c))
    
    all_clauses = list(cnf)
    all_watches = []
    for c in all_clauses:
        if len(c) >= 2:
            all_watches.append([c[0], c[1]])
        elif len(c) == 1:
            all_watches.append([c[0], c[0]])
        else:
            all_watches.append([-1, -1])
    
    lit_watchers = [[] for _ in range(2*n)]
    for ci in range(len(all_clauses)):
        for pos in range(2):
            lit = all_watches[ci][pos]
            if lit >= 0:
                lit_watchers[lit].append((ci, pos))
    
    assignment = [-1] * n
    var_level = [-1] * n
    trail = []
    trail_lim = [0]
    dl = 0
    
    activity = [1.0] * n
    var_inc = 1.0
    decay_factor = 0.95
    decay_count = 0
    DECAY_EVERY = 100
    
    def is_true(lit):
        v = lit >> 1
        a = assignment[v]
        if a == -1:
            return False
        return (a == 1) if (lit & 1) == 0 else (a == 0)
    
    def is_false(lit):
        v = lit >> 1
        a = assignment[v]
        if a == -1:
            return False
        return (a == 0) if (lit & 1) == 0 else (a == 1)
    
    def propagate():
        nonlocal all_watches, lit_watchers
        while len(trail) > trail_lim[dl]:
            v = trail.pop()
            a = assignment[v]
            falsified = 2*v + 1 if a == 1 else 2*v
            
            watchers = lit_watchers[falsified]
            i = 0
            while i < len(watchers):
                ci, pos = watchers[i]
                watch = all_watches[ci]
                clause = all_clauses[ci]
                
                other = watch[1 - pos]
                if is_true(other):
                    i += 1
                    continue
                
                new_lit = -1
                for lit in clause:
                    if lit == falsified:
                        continue
                    if is_false(lit):
                        continue
                    new_lit = lit
                    break
                
                if new_lit == -1:
                    return False, ci
                
                watch[pos] = new_lit
                last = len(watchers) - 1
                watchers[i] = watchers[last]
                watchers.pop()
                lit_watchers[new_lit].append((ci, pos))
            
        return True, None
    
    def add_clause(clause):
        nonlocal all_clauses, all_watches, lit_watchers
        all_clauses.append(clause)
        ci = len(all_clauses) - 1
        if len(clause) >= 2:
            all_watches.append([clause[0], clause[1]])
        elif len(clause) == 1:
            all_watches.append([clause[0], clause[0]])
        else:
            all_watches.append([-1, -1])
        for pos in range(2):
            lit = all_watches[ci][pos]
            if lit >= 0:
                lit_watchers[lit].append((ci, pos))
        return ci
    
    def unassign_to(level):
        nonlocal dl
        while len(trail) > trail_lim[level]:
            v = trail.pop()
            assignment[v] = -1
            var_level[v] = -1
        dl = level
    
    def bump(v):
        nonlocal activity, var_inc
        activity[v] += var_inc
    
    def decay_activities():
        nonlocal activity, var_inc, decay_count
        if decay_count >= DECAY_EVERY:
            for i in range(n):
                activity[i] *= decay_factor
            var_inc /= decay_factor
            decay_count = 0
    
    def pick_branch_var():
        nonlocal decay_count
        decay_count += 1
        decay_activities()
        best = -1
        best_act = -1.0
        for v in range(n):
            if assignment[v] == -1 and activity[v] > best_act:
                best_act = activity[v]
                best = v
        return best
    
    rng = random.Random(123)
    
    while time.time() < deadline:
        ok, conf_ci = propagate()
        if not ok:
            if dl == 0:
                break
            
            conf = all_clauses[conf_ci]
            levels = set()
            for lit in conf:
                v = lit >> 1
                if var_level[v] > 0:
                    levels.add(var_level[v])
            
            if not levels:
                break
            
            levels = sorted(levels, reverse=True)
            bj = levels[1] if len(levels) >= 2 else max(0, levels[0] - 1)
            
            learnt = [lit for lit in conf if var_level[lit >> 1] >= bj]
            if len(learnt) > 1 and not any(is_true(lit) for lit in learnt):
                add_clause(learnt)
            
            unassign_to(bj)
            for lit in conf:
                bump(lit >> 1)
            continue
        
        unassigned = [v for v in range(n) if assignment[v] == -1]
        if not unassigned:
            for c in all_clauses:
                if not any(is_true(lit) for lit in c):
                    break
            else:
                return [assignment[v] == 1 for v in range(n)]
            break
        
        v = pick_branch_var()
        if v == -1:
            break
        
        val = 1 if rng.random() < 0.5 else 0
        dl += 1
        trail_lim.append(len(trail))
        assignment[v] = val
        var_level[v] = dl
        trail.append(v)
    
    return [assignment[v] == 1 if assignment[v] != -1 else False for v in range(n)]
