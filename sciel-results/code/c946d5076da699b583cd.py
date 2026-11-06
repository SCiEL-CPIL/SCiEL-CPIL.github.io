import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    N = 256

    # Precompute all 3-APs and 4-APs
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map elements to APs containing them
    elem_to_3aps = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(idx)
            
    elem_to_4aps = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(idx)

    # Greedy Initialization
    in_set = [False] * (N + 1)
    for x in range(1, N + 1):
        creates_4ap = False
        # Check 4-APs containing x
        # x as 1st: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            if in_set[x+d] and in_set[x+2*d] and in_set[x+3*d]:
                creates_4ap = True
                break
        if not creates_4ap:
            # x as 2nd: x-d, x, x+d, x+2d
            max_d = min(x - 1, (N - x) // 2)
            for d in range(1, max_d + 1):
                if in_set[x-d] and in_set[x+d] and in_set[x+2*d]:
                    creates_4ap = True
                    break
        if not creates_4ap:
            # x as 3rd: x-2d, x-d, x, x+d
            max_d = min((x - 1) // 2, N - x)
            for d in range(1, max_d + 1):
                if in_set[x-2*d] and in_set[x-d] and in_set[x+d]:
                    creates_4ap = True
                    break
        if not creates_4ap:
            # x as 4th: x-3d, x-2d, x-d, x
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if in_set[x-3*d] and in_set[x-2*d] and in_set[x-d]:
                    creates_4ap = True
                    break
        if not creates_4ap:
            in_set[x] = True

    # Calculate initial scores
    def calc_scores(in_set):
        s3 = 0
        s4 = 0
        # 3-APs
        for a, b, c in three_aps:
            if in_set[a] and in_set[b] and in_set[c]:
                s3 += 1
        # 4-APs
        for a, b, c, d in four_aps:
            if in_set[a] and in_set[b] and in_set[c] and in_set[d]:
                s4 += 1
        return s3, s4

    cur_s3, cur_s4 = calc_scores(in_set)
    P = 10.0
    cur_S = cur_s3 - P * cur_s4
    
    best_valid_set = in_set.copy()
    best_valid_score = cur_s3
    
    T = 5.0
    alpha = 0.999
    
    # WalkSAT-style weights for 4-APs
    weights = [0.0] * len(four_aps)
    
    while time.time() < deadline:
        # Occasionally reset weights
        if random.random() < 0.005:
            weights = [0.0] * len(four_aps)
            
        x = random.randint(1, N)
        
        if in_set[x]:
            # Remove x
            delta_s3 = 0
            delta_s4 = 0
            
            # 3-APs lost
            for idx in elem_to_3aps[x]:
                a, b, c = three_aps[idx]
                if in_set[a] and in_set[b] and in_set[c]:
                    delta_s3 -= 1
            
            # 4-APs lost
            for idx in elem_to_4aps[x]:
                a, b, c, d = four_aps[idx]
                if in_set[a] and in_set[b] and in_set[c] and in_set[d]:
                    delta_s4 -= 1
                    # Update weight: this AP is now satisfied (removed), so decrease weight?
                    # Or increase weight because we just removed a conflict?
                    # In WalkSAT, if a clause becomes true, we might not change weight.
                    # If a clause was false and we flipped a literal to make it true, we increase weight of that clause?
                    # Actually, standard WalkSAT:
                    # 1. Pick a random unsatisfied clause.
                    # 2. Pick a literal in it with prob p, else random literal.
                    # 3. Flip.
                    # 4. Update weights: for each clause, if satisfied, w -= p; if unsatisfied, w += (1-p).
                    
            in_set[x] = False
        else:
            # Add x
            delta_s3 = 0
            delta_s4 = 0
            
            # 3-APs gained
            for idx in elem_to_3aps[x]:
                a, b, c = three_aps[idx]
                if in_set[a] and in_set[b] and in_set[c]:
                    delta_s3 += 1
            
            # 4-APs gained
            for idx in elem_to_4aps[x]:
                a, b, c, d = four_aps[idx]
                if in_set[a] and in_set[b] and in_set[c] and in_set[d]:
                    delta_s4 += 1
                    
            in_set[x] = True
            
        # Update weights based on new state
        # This is expensive if done for all APs.
        # We can approximate: only update weights for APs involving x?
        # But weights affect future choices.
        # Let's stick to a simpler weight update:
        # If an AP is satisfied (all in set), it's "good" for 3-APs but "bad" if 4-AP.
        # The penalty P handles the 4-AP part.
        # Let's use the WalkSAT idea for 4-APs specifically.
        # If a 4-AP is currently present (violated), we want to avoid creating more.
        # So, for violated 4-APs, increase weight.
        # For satisfied 3-APs, we want to keep them.
        
        # Simplified Weight Update:
        # For each 4-AP involving x:
        for idx in elem_to_4aps[x]:
            a, b, c, d = four_aps[idx]
            is_violated = in_set[a] and in_set[b] and in_set[c] and in_set[d]
            if is_violated:
                weights[idx] += 0.1
            else:
                weights[idx] -= 0.05
                if weights[idx] < 0: weights[idx] = 0

        new_s3 = cur_s3 + delta_s3
        new_s4 = cur_s4 + delta_s4
        new_S = new_s3 - P * new_s4
        
        # Acceptance probability
        # If new state is better, accept.
        # If worse, accept with prob exp((new_S - cur_S) / T)
        # Also, bias towards moves that reduce 4-AP violations?
        # The penalty P already does that.
        
        if new_S > cur_S or random.random() < math.exp((new_S - cur_S) / T):
            cur_s3 = new_s3
            cur_s4 = new_s4
            cur_S = new_S
            
            if cur_s4 == 0 and cur_s3 > best_valid_score:
                best_valid_score = cur_s3
                best_valid_set = in_set.copy()
        else:
            in_set[x] = not in_set[x]
            
        T *= alpha
        if T < 0.01:
            T = 0.01

    return [i for i in range(1, N + 1) if best_valid_set[i]]
