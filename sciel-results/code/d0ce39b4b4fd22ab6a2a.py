"""Max-SAT formulation with WalkSAT local search for square-pair-sum avoidance."""
import math
import random
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    n = 256
    # Precompute forbidden pairs: (a, b) with a <= b and a+b is a perfect square
    squares = set()
    for k in range(2, 23):  # 2^2=4 to 22^2=484, max sum is 256+256=512
        squares.add(k * k)
    # Actually max sum is 512, so squares up to 512: 4,9,...,484
    squares = set(k * k for k in range(2, 23) if k * k <= 512)
    
    # Build conflict graph: for each i, which j (j >= i) conflict with i
    conflicts = [[] for _ in range(n + 1)]  # 1-indexed
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            if (a + b) in squares:
                conflicts[a].append(b)
                if b != a:
                    conflicts[b].append(a)
    
    # Diagonal conflicts: a where 2a is a perfect square
    diag_conflict = set()
    for a in range(1, n + 1):
        if (2 * a) in squares:
            diag_conflict.add(a)
    
    # For efficient evaluation, precompute for each variable its conflicting partners
    # adj[i] = list of j such that (i,j) is a forbidden pair (i != j)
    # self_conflict[i] = True if i alone violates (2i is square)
    adj = [[] for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            if (a + b) in squares:
                if a != b:
                    adj[a].append(b)
                    adj[b].append(a)
    
    self_conflict = [False] * (n + 1)
    for a in range(1, n + 1):
        if (2 * a) in squares:
            self_conflict[a] = True
    
    # State: selected[i] = True if i is in the set
    # Score components:
    #   violated = number of forbidden pairs (a,b) where both a and b are selected
    #   cardinality = number of selected elements
    # We want to maximize cardinality with violated = 0
    # Penalty score = violated * P - cardinality, minimize this
    
    P = 10000  # large penalty for violations
    
    def evaluate(selected):
        """Return (violated_count, cardinality)."""
        violated = 0
        card = 0
        for i in range(1, n + 1):
            if selected[i]:
                card += 1
                if self_conflict[i]:
                    violated += 1
                for j in adj[i]:
                    if j > i and selected[j]:
                        violated += 1
        return violated, card
    
    def penalty_score(selected):
        v, c = evaluate(selected)
        return v * P - c
    
    # WalkSAT-style local search
    best_selected = None
    best_score = float('inf')
    best_card = 0
    
    def try_assignment(selected):
        nonlocal best_selected, best_score, best_card
        v, c = evaluate(selected)
        if v == 0:
            s = -c  # penalty = 0*P - c = -c, lower is better
        else:
            s = v * P - c
        if s < best_score:
            best_score = s
            best_selected = selected[:]
            best_card = c
        return s
    
    # Initial assignment: start with all non-self-conflicting elements selected
    # then repair
    selected = [False] * (n + 1)
    for i in range(1, n + 1):
        if not self_conflict[i]:
            selected[i] = True
    
    # Initial greedy removal to satisfy constraints
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        for i in range(1, n + 1):
            if not selected[i]:
                continue
            if self_conflict[i]:
                selected[i] = False
                changed = True
                continue
            for j in adj[i]:
                if j > i and selected[j]:
                    # Remove the one with higher degree to keep more
                    if len(adj[i]) >= len(adj[j]):
                        selected[i] = False
                    else:
                        selected[j] = False
                    changed = True
                    break
    
    try_assignment(selected)
    
    # Local search: WalkSAT with adaptive weights
    # For each step, pick a variable to flip based on:
    #   - If there are violations, pick a variable involved in a violated clause
    #   - Otherwise, pick a random variable to try to increase cardinality
    
    # Precompute for faster delta evaluation
    # When we flip variable i:
    #   If we add i (selected[i]: False -> True):
    #     - If self_conflict[i]: +1 violation
    #     - For each j in adj[i] where selected[j]: +1 violation
    #     - Cardinality: +1
    #   If we remove i (selected[i]: True -> False):
    #     - If self_conflict[i]: -1 violation
    #     - For each j in adj[i] where selected[j]: -1 violation
    #     - Cardinality: -1
    
    def flip_delta(i, selected):
        """Return (delta_violated, delta_card) if we flip i."""
        if selected[i]:
            # Removing i
            dv = 0
            if self_conflict[i]:
                dv -= 1
            for j in adj[i]:
                if selected[j]:
                    dv -= 1
            dc = -1
        else:
            # Adding i
            dv = 0
            if self_conflict[i]:
                dv += 1
            for j in adj[i]:
                if selected[j]:
                    dv += 1
            dc = +1
        return dv, dc
    
    # Main local search loop
    rng = random.Random(42)
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 2:
        v, c = evaluate(selected)
        s = v * P - c
        if s < best_score:
            best_score = s
            best_selected = selected[:]
            best_card = c
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > max_no_improve:
            # Restart with perturbation
            # Take best solution and randomly flip some
            selected = best_selected[:]
            for i in range(1, n + 1):
                if rng.random() < 0.05:
                    selected[i] = not selected[i]
            no_improve = 0
            continue
        
        if v == 0:
            # No violations, try to add elements
            # Find a non-selected element that can be added without creating violations
            candidates = []
            for i in range(1, n + 1):
                if selected[i]:
                    continue
                # Check if adding i would create violations
                if self_conflict[i]:
                    continue
                ok = True
                for j in adj[i]:
                    if selected[j]:
                        ok = False
                        break
                if ok:
                    candidates.append(i)
            
            if candidates:
                i = rng.choice(candidates)
                selected[i] = True
            else:
                # Stuck: try removing an element to allow others
                # Find an element whose removal allows adding more
                best_gain = 0
                best_remove = None
                for i in range(1, n + 1):
                    if not selected[i]:
                        continue
                    # Count how many non-selected neighbors i has
                    gain = 0
                    for j in adj[i]:
                        if not selected[j] and not self_conflict[j]:
                            # Check if j can be added if i is removed
                            # j would still conflict with other selected neighbors
                            can_add = True
                            for k in adj[j]:
                                if k != i and selected[k]:
                                    can_add = False
                                    break
                            if can_add:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_remove = i
                
                if best_remove is not None and best_gain > 0:
                    selected[best_remove] = False
                else:
                    # Random flip to escape
                    i = rng.randint(1, n)
                    selected[i] = not selected[i]
        else:
            # Has violations, try to fix
            # Find violated pairs and pick a variable to flip
            # Collect all variables involved in violations
            violated_vars = set()
            for i in range(1, n + 1):
                if selected[i]:
                    if self_conflict[i]:
                        violated_vars.add(i)
                    for j in adj[i]:
                        if j > i and selected[j]:
                            violated_vars.add(i)
                            violated_vars.add(j)
            
            if not violated_vars:
                # Shouldn't happen if v > 0, but safety
                i = rng.randint(1, n)
                selected[i] = not selected[i]
                continue
            
            # For each candidate flip, compute delta
            # We want to minimize penalty: minimize (v*P - c)
            # Flipping i changes v by dv and c by dc
            # New penalty = (v+dv)*P - (c+dc)
            # Delta penalty = dv*P - dc
            # We want the most negative delta penalty (biggest improvement)
            
            best_i = None
            best_delta_pen = 0  # we want to minimize, so find most negative
            
            # Sample from violated vars and some random ones
            candidates = list(violated_vars)
            # Add some random non-violated for diversity
            for _ in range(min(10, n)):
                candidates.append(rng.randint(1, n))
            
            for i in candidates:
                dv, dc = flip_delta(i, selected)
                delta_pen = dv * P - dc
                if delta_pen < best_delta_pen:
                    best_delta_pen = delta_pen
                    best_i = i
            
            if best_i is not None:
                selected[best_i] = not selected[best_i]
            else:
                # No improvement possible, random flip
                i = rng.choice(list(violated_vars))
                selected[i] = not selected[i]
    
    # Final: return the best valid solution found
    if best_selected is not None:
        result = [i for i in range(1, n + 1) if best_selected[i]]
        if result:
            return result
    
    # Fallback: greedy construction
    result = []
    for i in range(1, n + 1):
        if self_conflict[i]:
            continue
        ok = True
        for j in result:
            if (i + j) in squares:
                ok = False
                break
        if ok:
            result.append(i)
    
    return result if result else [1]
# EVOLVE-BLOCK-END
