import time
import random

N = 12

def compute_threshold(auto):
    # auto is list of 2 lists of 12 ints
    # mask: 12-bit int
    start = (1 << N) - 1
    if start & (start - 1) == 0:
        return 0
    
    visited = {start}
    queue = [(start, 0)]
    
    sigma0 = auto[0]
    sigma1 = auto[1]
    
    while queue:
        mask, depth = queue.pop(0)
        for sigma in (sigma0, sigma1):
            new_mask = 0
            # Unroll loop for speed?
            for i in range(N):
                if mask & (1 << i):
                    new_mask |= (1 << sigma[i])
            
            if new_mask & (new_mask - 1) == 0:
                return depth + 1
            if new_mask not in visited:
                visited.add(new_mask)
                queue.append((new_mask, depth + 1))
    return -1 # Should not happen if synchronizing

def construct():
    # Pawlik automaton
    sigma0 = [(i + 1) % N for i in range(N)]
    sigma1 = [0 if i == 0 else i - 1 for i in range(N)]
    best_auto = [sigma0.copy(), sigma1.copy()]
    best_score = compute_threshold(best_auto)
    
    deadline = time.time() + 55 # Safety margin
    
    stagnant = 0
    random.seed(42) # For reproducibility in local search? Or just random
    
    while time.time() < deadline:
        # Mutate
        cand = [best_auto[0].copy(), best_auto[1].copy()]
        letter = random.randint(0, 1)
        state = random.randint(0, N - 1)
        new_dest = random.randint(0, N - 1)
        cand[letter][state] = new_dest
        
        score = compute_threshold(cand)
        
        if score > best_score:
            best_auto = cand
            best_score = score
            stagnant = 0
        else:
            stagnant += 1
            if stagnant > 200:
                # Restart from Pawlik or best?
                # Let's try a random kick
                # Or just reset to Pawlik to explore other basins?
                # Actually, just continuing might be better if we use simulated annealing.
                # For simplicity, let's just keep going.
                # If we really get stuck, we can restart.
                if stagnant > 1000:
                    best_auto = [[(i + 1) % N for i in range(N)], [0 if i == 0 else i - 1 for i in range(N)]]
                    best_score = compute_threshold(best_auto)
                    stagnant = 0
    
    return best_auto
