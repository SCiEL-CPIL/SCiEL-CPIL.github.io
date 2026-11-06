import time
import sys

sys.setrecursionlimit(20000)

N = 256
weights = [0.0] * (N + 1)
for i in range(1, N + 1):
    weights[i] = 1.0 / i

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_score = 0.0
    best_set = set()
    
    # Greedy Incumbent
    def greedy():
        chosen = set()
        banned = set()
        candidates = set(range(1, N + 1))
        score = 0.0
        while candidates:
            x = max(candidates, key=lambda k: weights[k])
            chosen.add(x)
            score += weights[x]
            candidates.remove(x)
            
            new_bans = set()
            for z in chosen:
                y1 = 2 * x - z
                if 1 <= y1 <= N: new_bans.add(y1)
                y2 = 2 * z - x
                if 1 <= y2 <= N: new_bans.add(y2)
                if (x + z) % 2 == 0:
                    y3 = (x + z) // 2
                    if 1 <= y3 <= N: new_bans.add(y3)
            
            banned.update(new_bans)
            candidates -= banned
        return chosen, score

    initial_set, initial_score = greedy()
    best_set = initial_set
    best_score = initial_score
    
    def solve(chosen, banned, candidates, current_score, cand_sum):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_score > best_score:
                best_score = current_score
                best_set = set(chosen)
            return
        
        # Upper Bound
        if current_score + cand_sum <= best_score:
            return
        
        # Pick x
        x = max(candidates, key=lambda k: weights[k])
        
        # Branch 1: Include x
        # Calculate new bans
        new_bans = set()
        for z in chosen:
            y1 = 2 * x - z
            if 1 <= y1 <= N: new_bans.add(y1)
            y2 = 2 * z - x
            if 1 <= y2 <= N: new_bans.add(y2)
            if (x + z) % 2 == 0:
                y3 = (x + z) // 2
                if 1 <= y3 <= N: new_bans.add(y3)
        
        # Check validity: if any ban is in chosen, x is invalid
        valid = True
        for y in new_bans:
            if y in chosen:
                valid = False
                break
        
        if valid:
            new_chosen = chosen | {x}
            new_banned = banned | new_bans
            new_candidates = candidates - {x} - new_banned
            
            # Calculate new cand_sum
            # Subtract weight of x
            # Subtract weights of any new_bans that were in candidates
            dec = weights[x]
            for y in new_bans:
                if y in candidates:
                    dec += weights[y]
            
            new_cand_sum = cand_sum - dec
            
            solve(new_chosen, new_banned, new_candidates, current_score + weights[x], new_cand_sum)
        
        # Branch 2: Exclude x
        new_candidates_excl = candidates - {x}
        new_cand_sum_excl = cand_sum - weights[x]
        
        solve(chosen, banned, new_candidates_excl, current_score, new_cand_sum_excl)

    # Start
    initial_candidates = set(range(1, N + 1))
    initial_cand_sum = sum(weights[1:])
    
    solve(set(), set(), initial_candidates, 0.0, initial_cand_sum)
    
    return list(best_set)
