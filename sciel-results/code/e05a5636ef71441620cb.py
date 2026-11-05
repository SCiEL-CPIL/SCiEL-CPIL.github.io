import time
import random
import math

N = 256

# Precompute 3-AP neighbors for fast gain calculation
# neighbors3[x] = list of (y, z) such that {x, y, z} is a 3-AP
neighbors3 = [[] for _ in range(N + 1)]

# Precompute 4-AP conflicts
# For each x, list of masks of other 3 elements that with x form a 4-AP
# mask is a bitmask of indices 1..N
conflicts4 = [[] for _ in range(N + 1)]

def to_mask(vals):
    m = 0
    for v in vals:
        if 1 <= v <= N:
            m |= (1 << (v - 1))
    return m

for x in range(1, N + 1):
    # 3-APs containing x
    # x is first: x, x+d, x+2d
    for d in range(1, (N - x) // 2 + 1):
        neighbors3[x].append((x + d, x + 2 * d))
    # x is middle: x-d, x, x+d
    for d in range(1, min(x - 1, N - x) + 1):
        neighbors3[x].append((x - d, x + d))
    # x is last: x-2d, x-d, x
    for d in range(1, (x - 1) // 2 + 1):
        neighbors3[x].append((x - 2 * d, x - d))
        
    # 4-APs containing x
    # x is pos 1: x, x+d, x+2d, x+3d
    for d in range(1, (N - x) // 3 + 1):
        conflicts4[x].append(to_mask([x + d, x + 2 * d, x + 3 * d]))
    # x is pos 2: x-d, x, x+d, x+2d
    for d in range(1, min(x - 1, (N - x) // 2) + 1):
        conflicts4[x].append(to_mask([x - d, x + d, x + 2 * d]))
    # x is pos 3: x-2d, x-d, x, x+d
    for d in range(1, min((x - 1) // 2, N - x) + 1):
        conflicts4[x].append(to_mask([x - 2 * d, x - d, x + d]))
    # x is pos 4: x-3d, x-2d, x-d, x
    for d in range(1, (x - 1) // 3 + 1):
        conflicts4[x].append(to_mask([x - 3 * d, x - 2 * d, x - d]))

def is_in_set(A, x):
    return bool((A >> (x - 1)) & 1)

def has_conflict(A, x):
    """Check if adding x to A creates a 4-AP."""
    for mask in conflicts4[x]:
        if (A & mask) == mask:
            return True
    return False

def gain_add(A, x):
    """Calculate number of 3-APs formed by adding x to A."""
    g = 0
    for a, b in neighbors3[x]:
        if is_in_set(A, a) and is_in_set(A, b):
            g += 1
    return g

def count_3AP_total(A):
    """Count total 3-APs in set A (bitmask)."""
    count = 0
    temp = A
    while temp:
        lsb = temp & -temp
        b = lsb.bit_length()
        temp ^= lsb
        # b is the value (1-indexed)
        for a, c in neighbors3[b]:
            if is_in_set(A, a) and is_in_set(A, c):
                count += 1
    return count

def greedy_seed(rng):
    A = 0
    order = list(range(1, N + 1))
    rng.shuffle(order)
    for x in order:
        if not has_conflict(A, x):
            A |= (1 << (x - 1))
    return A

def simulated_annealing(A, deadline, rng, T_start=10.0, T_end=0.1, alpha=0.999):
    """
    Simulated Annealing to maximize 3-APs subject to no 4-APs.
    Moves: 
    1. Add a random x that doesn't create 4-AP.
    2. Remove a random x.
    3. Swap: Remove r, add x.
    """
    current_score = count_3AP_total(A)
    best_A = A
    best_score = current_score
    
    T = T_start
    
    while time.time() < deadline:
        # Cooling
        T = max(T_end, T * alpha)
        
        # Choose a move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add a random element
            candidates = [x for x in range(1, N + 1) if not is_in_set(A, x) and not has_conflict(A, x)]
            if not candidates:
                continue
            x = rng.choice(candidates)
            new_A = A | (1 << (x - 1))
            delta = gain_add(A, x)
            
        elif move_type < 0.7:
            # Try to remove a random element
            in_set = [x for x in range(1, N + 1) if is_in_set(A, x)]
            if not in_set:
                continue
            r = rng.choice(in_set)
            new_A = A & ~(1 << (r - 1))
            # Delta is negative of loss
            loss = 0
            for a, b in neighbors3[r]:
                if is_in_set(A, a) and is_in_set(A, b):
                    loss += 1
            delta = -loss
            
        else:
            # Swap: Remove r, add x
            in_set = [x for x in range(1, N + 1) if is_in_set(A, x)]
            if not in_set:
                continue
            r = rng.choice(in_set)
            A_temp = A & ~(1 << (r - 1))
            
            # Find candidates for x
            # To speed up, just pick random x not in A_temp
            x = rng.randint(1, N)
            if is_in_set(A_temp, x):
                continue
            if has_conflict(A_temp, x):
                continue
                
            new_A = A_temp | (1 << (x - 1))
            
            # Calculate delta
            # Loss from removing r (in A)
            loss_r = 0
            for a, b in neighbors3[r]:
                if is_in_set(A, a) and is_in_set(A, b):
                    loss_r += 1
            # Gain from adding x (in A_temp)
            gain_x = gain_add(A_temp, x)
            delta = gain_x - loss_r

        new_score = current_score + delta
        
        # Acceptance criterion
        if delta > 0:
            accept = True
        else:
            # Accept with probability exp(delta / T)
            # If delta is very negative, exp is 0
            if T > 0:
                prob = math.exp(delta / T)
            else:
                prob = 0
            accept = rng.random() < prob
        
        if accept:
            A = new_A
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_A = A
                
        # Small chance to restart or perturb if stuck? 
        # For now, just continue.
        
    return best_score, best_A

def construct():
    start = time.time()
    deadline = start + 55.0
    rng = random.Random(42)

    best_A = 0
    best_score = -1

    # Run multiple SA instances with different seeds/temps
    # Since we have 55s, we can do a few runs.
    # SA is stochastic, so multiple runs help.
    
    # We'll run SA for the remaining time, but split it into chunks
    # or just run one long SA and keep the best.
    # Given the landscape, one long run might be better than many short ones
    # if the cooling schedule is good. But multiple restarts help escape bad basins.
    
    # Strategy: 
    # 1. Generate a good seed with greedy.
    # 2. Run SA.
    # 3. If time permits, restart SA from the best found so far with a higher temp?
    #    Or just keep running SA on the incumbent.
    
    # Let's do a hybrid: 
    # Run SA for 20s, record best.
    # Run SA again from a new greedy seed for 20s, record best.
    # Use remaining time on the global best.
    
    # Simpler: Just run SA in a loop, restarting if we detect stagnation or just letting it run.
    # Actually, SA with a good schedule doesn't need explicit restarts if T is managed well.
    # But to be safe against getting stuck in a very bad local minimum early on,
    # we can do a few "epochs".
    
    epochs = 3
    time_per_epoch = (deadline - start) / epochs
    
    for i in range(epochs):
        epoch_deadline = start + (i + 1) * time_per_epoch
        A_seed = greedy_seed(rng)
        score, A_res = simulated_annealing(A_seed, epoch_deadline, rng, T_start=5.0, T_end=0.5, alpha=0.9995)
        if score > best_score:
            best_score = score
            best_A = A_res

    # Final polish on the best found
    if time.time() < deadline - 1.0 and best_A != 0:
        # Run SA again on the best incumbent with a lower starting temp to refine
        score, A_res = simulated_annealing(best_A, deadline, rng, T_start=1.0, T_end=0.01, alpha=0.9999)
        if score > best_score:
            best_score = score
            best_A = A_res

    if best_A == 0:
        best_A = 1 << 0 # Fallback {1}

    A_list = [x for x in range(1, N + 1) if is_in_set(best_A, x)]
    
    if not A_list:
        A_list = [1]
        
    return A_list

if __name__ == "__main__":
    res = construct()
    print(len(res))
