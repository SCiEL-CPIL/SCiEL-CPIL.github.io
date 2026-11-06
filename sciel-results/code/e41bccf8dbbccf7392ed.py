import random
import time

def construct():
    mandatory = {1, 2, 5, 11}
    max_val = 284
    
    def get_conflicts(S):
        """
        Returns the number of conflicting sum pairs.
        A conflict is a sum value that appears more than once.
        If a sum appears k times, it contributes (k-1) to the conflict count?
        Or is it the number of pairs of (a,b) that have the same sum?
        For validity, we need all sums to be distinct.
        So we want 0 conflicts.
        Cost function: 
        If we count the number of "bad" pairs of pairs...
        Simpler: 
        Let sums be a list of all a+b.
        Sort sums.
        Count adjacent duplicates.
        """
        L = sorted(S)
        n = len(L)
        sums = []
        for i in range(n):
            for j in range(i, n):
                sums.append(L[i] + L[j])
        sums.sort()
        conflicts = 0
        for i in range(1, len(sums)):
            if sums[i] == sums[i-1]:
                conflicts += 1
        return conflicts

    def greedy_init():
        S = set(mandatory)
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        for x in candidates:
            if x in S:
                continue
            # Check if adding x creates a conflict
            new_S = S | {x}
            if get_conflicts(new_S) == 0:
                S.add(x)
        return S

    def simulated_annealing():
        S = greedy_init()
        current_cost = get_conflicts(S)
        # We want to maximize size, so we should penalize conflicts heavily
        # and reward size.
        # Let's define a composite score.
        
        def score(S):
            c = get_conflicts(S)
            # If c > 0, it's invalid.
            # We want to minimize c, then maximize len(S).
            # Score = c * 1000 - len(S)
            return c * 1000 - len(S)
        
        current_score = score(S)
        best_S = set(S)
        best_score = current_score
        
        T = 50.0
        T_min = 0.1
        cooling = 0.995
        
        start_time = time.time()
        
        while T > T_min and time.time() - start_time < 110:
            # Generate neighbor
            r = random.random()
            if r < 0.3:
                # Add
                if len(S) < max_val:
                    candidates = [x for x in range(1, max_val + 1) if x not in S]
                    if candidates:
                        x = random.choice(candidates)
                        S_new = S | {x}
                    else:
                        continue
            elif r < 0.6:
                # Remove
                removable = [x for x in S if x not in mandatory]
                if removable:
                    x = random.choice(removable)
                    S_new = S - {x}
                else:
                    continue
            else:
                # Swap
                removable = [x for x in S if x not in mandatory]
                if not removable:
                    continue
                x_out = random.choice(removable)
                candidates = [x for x in range(1, max_val + 1) if x not in S]
                if not candidates:
                    continue
                x_in = random.choice(candidates)
                S_new = (S - {x_out}) | {x_in}
            
            new_score = score(S_new)
            delta = new_score - current_score
            
            if delta < 0:
                # Better
                S = S_new
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_S = set(S)
            else:
                # Worse
                if random.random() < pow(2.71828, -delta / T):
                    S = S_new
                    current_score = new_score
            
            T *= cooling
            
        return best_S

    # Run multiple restarts
    best_overall = set(mandatory)
    start_global = time.time()
    
    while time.time() - start_global < 115:
        S = simulated_annealing()
        if len(S) > len(best_overall):
            best_overall = set(S)
            
    return sorted(best_overall)
