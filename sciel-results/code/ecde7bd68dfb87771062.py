import time
import random
import numpy as np

N = 256

def _precompute_numpy():
    """
    Precompute dependencies for 3-APs and 4-APs in a format suitable for NumPy vectorization.
    """
    # three_ap_u[x], three_ap_v[x]: indices u, v such that {x, u, v} is a 3-AP
    # If x is middle: u = x-d, v = x+d
    # If x is end: u = x-d, v = x-2d (or x+d, x+2d)
    three_ap_u = [None] * (N + 1)
    three_ap_v = [None] * (N + 1)
    
    for x in range(1, N + 1):
        us = []
        vs = []
        # x as middle
        for d in range(1, x):
            u = x - d
            v = x + d
            if v <= N:
                us.append(u)
                vs.append(v)
        # x as first
        for d in range(1, N + 1):
            u = x + d
            v = x + 2 * d
            if v <= N:
                us.append(u)
                vs.append(v)
        # x as last
        for d in range(1, x):
            u = x - d
            v = x - 2 * d
            if v >= 1:
                us.append(u)
                vs.append(v)
        
        three_ap_u[x] = np.array(us, dtype=np.int32)
        three_ap_v[x] = np.array(vs, dtype=np.int32)
        
    # four_ap_others[x]: array of shape (K, 3) containing indices of other 3 elements in 4-APs with x
    four_ap_others = [None] * (N + 1)
    
    # Generate all 4-APs
    all_4aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                all_4aps.append((a, b, c, e))
                
    # Map element to list of other triples
    tmp_maps = [[] for _ in range(N + 1)]
    for (a, b, c, e) in all_4aps:
        tmp_maps[a].append((b, c, e))
        tmp_maps[b].append((a, c, e))
        tmp_maps[c].append((a, b, e))
        tmp_maps[e].append((a, b, c))
        
    for x in range(1, N + 1):
        if tmp_maps[x]:
            four_ap_others[x] = np.array(tmp_maps[x], dtype=np.int32)
        else:
            four_ap_others[x] = np.empty((0, 3), dtype=np.int32)
            
    return three_ap_u, three_ap_v, four_ap_others

def _eval_addition_deltas(A, three_ap_u, three_ap_v):
    """
    Vectorized calculation of delta for adding each element y not in A.
    Returns an array of deltas for y=1..N.
    """
    deltas = np.zeros(N + 1, dtype=np.int32)
    for y in range(1, N + 1):
        if not A[y-1]: # 0-indexed in A
            us = three_ap_u[y]
            vs = three_ap_v[y]
            if len(us) > 0:
                # Count pairs (u, v) in A
                deltas[y] = np.sum(A[us-1] & A[vs-1])
    return deltas

def _eval_removal_deltas(A, three_ap_u, three_ap_v):
    """
    Vectorized calculation of delta for removing each element x in A.
    Returns an array of deltas (negative values) for x=1..N.
    """
    deltas = np.zeros(N + 1, dtype=np.int32)
    for x in range(1, N + 1):
        if A[x-1]:
            us = three_ap_u[x]
            vs = three_ap_v[x]
            if len(us) > 0:
                # Count 3-APs that will be broken
                lost = np.sum(A[us-1] & A[vs-1])
                deltas[x] = -lost
    return deltas

def _is_valid_addition(A, y, four_ap_others):
    """
    Check if adding y to A creates a 4-AP.
    """
    others = four_ap_others[y]
    if len(others) == 0:
        return True
    # Check if any row in 'others' has all 3 elements in A
    # A is 0-indexed, others are 1-indexed
    # We need to check if A[others[:,0]-1] & A[others[:,1]-1] & A[others[:,2]-1] has any True
    return not np.any(A[others[:,0]-1] & A[others[:,1]-1] & A[others[:,2]-1])

def _is_valid_removal(A, x):
    """
    Removing an element never creates a 4-AP, so it's always valid.
    """
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    three_ap_u, three_ap_v, four_ap_others = _precompute_numpy()
    
    # Helper to count total 3-APs
    def count_3aps(A):
        total = 0
        for x in range(1, N + 1):
            if A[x-1]:
                us = three_ap_u[x]
                vs = three_ap_v[x]
                if len(us) > 0:
                    total += np.sum(A[us-1] & A[vs-1])
        # Each 3-AP is counted 3 times (once for each element)
        return total // 3
        
    # Phase 1: Greedy Initialization
    A = np.zeros(N, dtype=bool)
    # Try multiple random orders for greedy
    best_A = None
    best_score = -1
    
    random.seed(42)
    for _ in range(10):
        order = list(range(1, N + 1))
        random.shuffle(order)
        A_curr = np.zeros(N, dtype=bool)
        for y in order:
            if _is_valid_addition(A_curr, y, four_ap_others):
                A_curr[y-1] = True
        score = count_3aps(A_curr)
        if score > best_score:
            best_score = score
            best_A = A_curr.copy()
            
    A = best_A
    current_score = best_score
    best_A = A.copy()
    best_score = current_score
    
    # Phase 2: Simulated Annealing
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline - 5.0 and T > T_min:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.6:
            # Try Addition
            unset_indices = np.where(~A)[0] + 1
            if len(unset_indices) == 0:
                T *= cooling
                continue
            # Pick a random candidate
            y = random.choice(unset_indices)
            if not _is_valid_addition(A, y, four_ap_others):
                T *= cooling
                continue
            
            us = three_ap_u[y]
            vs = three_ap_v[y]
            if len(us) > 0:
                delta = np.sum(A[us-1] & A[vs-1])
            else:
                delta = 0
                
            new_score = current_score + delta
            
            if new_score > current_score or (delta < 0 and random.random() < np.exp(delta / T)):
                A[y-1] = True
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_A = A.copy()
                    
        elif move_type < 0.8:
            # Try Removal
            set_indices = np.where(A)[0] + 1
            if len(set_indices) == 0:
                T *= cooling
                continue
            x = random.choice(set_indices)
            us = three_ap_u[x]
            vs = three_ap_v[x]
            if len(us) > 0:
                delta = -np.sum(A[us-1] & A[vs-1])
            else:
                delta = 0
                
            new_score = current_score + delta
            
            if new_score > current_score or (delta < 0 and random.random() < np.exp(delta / T)):
                A[x-1] = False
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_A = A.copy()
                    
        else:
            # Try Swap (Remove x, Add y)
            set_indices = np.where(A)[0] + 1
            unset_indices = np.where(~A)[0] + 1
            if len(set_indices) == 0 or len(unset_indices) == 0:
                T *= cooling
                continue
            x = random.choice(set_indices)
            y = random.choice(unset_indices)
            
            # Check validity of adding y to A without x
            A_temp = A.copy()
            A_temp[x-1] = False
            if not _is_valid_addition(A_temp, y, four_ap_others):
                T *= cooling
                continue
                
            # Calculate delta
            # Delta = (Gain from y in A without x) - (Loss from x in A)
            # Note: Gain from y might depend on x if x,y form a 3-AP?
            # If x and y are in a 3-AP with z, then:
            # In A: x, z present. y not.
            # In A' (A-x+y): x not, y, z present.
            # The 3-AP {x, z, w} is broken.
            # The 3-AP {y, z, v} is formed.
            # We need to be careful.
            
            # Loss from x: 3-APs containing x in A
            us_x = three_ap_u[x]
            vs_x = three_ap_v[x]
            loss_x = 0
            if len(us_x) > 0:
                loss_x = np.sum(A[us_x-1] & A[vs_x-1])
                
            # Gain from y: 3-APs containing y in A without x
            us_y = three_ap_u[y]
            vs_y = three_ap_v[y]
            gain_y = 0
            if len(us_y) > 0:
                gain_y = np.sum(A_temp[us_y-1] & A_temp[vs_y-1])
                
            delta = gain_y - loss_x
            new_score = current_score + delta
            
            if new_score > current_score or (delta < 0 and random.random() < np.exp(delta / T)):
                A_temp[y-1] = True
                A = A_temp
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_A = A.copy()
                    
        T *= cooling
        
    # Phase 3: Local Search Polish (Best Improvement)
    # Try to add any valid element that increases score
    improved = True
    while improved and time.time() < deadline - 1.0:
        improved = False
        unset_indices = np.where(~A)[0] + 1
        if len(unset_indices) == 0: break
        
        best_delta = 0
        best_y = -1
        for y in unset_indices:
            if _is_valid_addition(A, y, four_ap_others):
                us = three_ap_u[y]
                vs = three_ap_v[y]
                if len(us) > 0:
                    delta = np.sum(A[us-1] & A[vs-1])
                else:
                    delta = 0
                if delta > best_delta:
                    best_delta = delta
                    best_y = y
                    
        if best_y != -1:
            A[best_y-1] = True
            current_score += best_delta
            if current_score > best_score:
                best_score = current_score
                best_A = A.copy()
            improved = True
            
    # Convert back to list
    result = [i+1 for i in range(N) if best_A[i]]
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
