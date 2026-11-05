import time
import random
import math

N = 256

# Precompute 3-APs and 4-APs as bitmasks
# and build indices

three_aps = []
four_aps = []

# three_ap_indices[x] = list of indices of 3-APs containing x
# four_ap_indices[x] = list of indices of 4-APs containing x
three_ap_indices = [[] for _ in range(N + 1)]
four_ap_indices = [[] for _ in range(N + 1)]

def precompute():
    global three_aps, four_aps
    
    # Generate 3-APs
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            idx = len(three_aps)
            three_aps.append(mask)
            three_ap_indices[a].append(idx)
            three_ap_indices[b].append(idx)
            three_ap_indices[c].append(idx)
            
    # Generate 4-APs
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            idx = len(four_aps)
            four_aps.append(mask)
            four_ap_indices[a].append(idx)
            four_ap_indices[b].append(idx)
            four_ap_indices[c].append(idx)
            four_ap_indices[e].append(idx)

precompute()

def count_3aps(S_mask):
    """Count 3-APs in S_mask. Used for final verification or initialization."""
    count = 0
    for mask in three_aps:
        if (S_mask & mask) == mask:
            count += 1
    return count

def is_valid(S_mask):
    """Check if S_mask is 4-AP free."""
    for mask in four_aps:
        if (S_mask & mask) == mask:
            return False
    return True

def greedy_construct(seed):
    random.seed(seed)
    S_mask = 0
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        bit = 1 << (x - 1)
        new_S = S_mask | bit
        # Check validity incrementally
        valid = True
        for idx in four_ap_indices[x]:
            m = four_aps[idx]
            # Check if other 3 bits are in S_mask
            # m has 4 bits. One is x.
            # (S_mask & (m ^ bit)) == (m ^ bit)
            if (S_mask & (m ^ bit)) == (m ^ bit):
                valid = False
                break
        if valid:
            S_mask = new_S
            
    return S_mask

def simulated_annealing(S_mask, time_limit):
    start_time = time.time()
    deadline = start_time + time_limit
    
    current_S = S_mask
    current_score = count_3aps(S_mask)
    
    best_S = S_mask
    best_score = current_score
    
    T = 10.0
    cooling = 0.99
    
    # Precompute neighbors for faster moves?
    # We will just pick random x.
    
    while time.time() < deadline:
        if T < 0.01:
            T = 10.0 # Restart? Or just stop?
            # Let's do a few restarts or just keep going with low T
            # Actually, if T is very low, we are in local search mode.
            
        x = random.randint(1, N)
        bit = 1 << (x - 1)
        
        if current_S & bit:
            # Remove x
            new_S = current_S ^ bit
            # Check validity: Removing x can only remove 4-APs.
            # Since current_S is valid, new_S is always valid.
            # Calculate score change
            delta = 0
            for idx in three_ap_indices[x]:
                m = three_aps[idx]
                # Check if other 2 bits are in current_S
                # m has 3 bits. One is x.
                # (current_S & (m ^ bit)) == (m ^ bit)
                if (current_S & (m ^ bit)) == (m ^ bit):
                    delta -= 1
            new_score = current_score + delta
        else:
            # Add x
            new_S = current_S | bit
            # Check validity
            valid = True
            for idx in four_ap_indices[x]:
                m = four_aps[idx]
                if (current_S & (m ^ bit)) == (m ^ bit):
                    valid = False
                    break
            
            if not valid:
                # Reject
                continue
            
            # Calculate score change
            delta = 0
            for idx in three_ap_indices[x]:
                m = three_aps[idx]
                if (current_S & (m ^ bit)) == (m ^ bit):
                    delta += 1
            new_score = current_score + delta
        
        # Acceptance
        if new_score >= current_score:
            current_S = new_S
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_S = current_S
        else:
            # Accept with probability exp(delta / T)
            # delta is negative
            p = math.exp(delta / T)
            if random.random() < p:
                current_S = new_S
                current_score = new_score
        
        T *= cooling
        
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_result_mask = 0
    best_score = -1
    
    # Multiple restarts
    # Estimate time per restart
    # Let's do 10 restarts with 5s each?
    # Or adaptively.
    
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        remaining = deadline - time.time()
        if remaining < 2.0:
            # Not enough time for SA, just greedy
            S_mask = greedy_construct(i * 100 + 42)
            score = count_3aps(S_mask)
            if score > best_score:
                best_score = score
                best_result_mask = S_mask
            continue
            
        # Allocate time to SA
        # Leave 1s for next restart setup
        sa_time = remaining - 1.0
        if sa_time < 1.0:
            continue
            
        seed = i * 100 + 42
        S_mask = greedy_construct(seed)
        
        S_opt = simulated_annealing(S_mask, sa_time)
        score = count_3aps(S_opt)
        
        if score > best_score:
            best_score = score
            best_result_mask = S_opt
            
    # Convert mask to list
    result = []
    for i in range(N):
        if best_result_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)
