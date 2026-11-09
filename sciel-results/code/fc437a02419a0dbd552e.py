import numpy as np
import time
import random

N = 512
# Precompute APs
APs = []
aps_containing = [[] for _ in range(N + 1)]

for d in range(1, (N - 1) // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        ap = (a, a + d, a + 2 * d, a + 3 * d)
        idx = len(APs)
        APs.append(ap)
        for x in ap:
            aps_containing[x].append(idx)

# Convert to NumPy arrays for fast access
aps_containing_np = [np.array(lst, dtype=np.int32) for lst in aps_containing]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Initial solution: Greedy Hitting Set (fast version)
    # We can just start with a random set or a greedy one.
    # Let's use a simple greedy to get a good starting point.
    
    def greedy_initial():
        S = set()
        # Random order
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        # Use the fast check logic
        # We need to initialize counts to 0
        counts = np.zeros(len(APs), dtype=np.int32)
        in_S = np.zeros(N + 1, dtype=bool)
        
        for x in candidates:
            indices = aps_containing_np[x]
            # Check if any of these APs already have 3 elements in S
            # i.e., counts[indices] == 3
            if not np.any(counts[indices] == 3):
                S.add(x)
                in_S[x] = True
                counts[indices] += 1
        return S, counts, in_S

    best_S, best_counts, best_in_S = greedy_initial()
    best_size = len(best_S)
    
    # Simulated Annealing
    current_S, current_counts, current_in_S = best_S, best_counts.copy(), best_in_S.copy()
    current_size = best_size
    
    T = 1.0
    cooling = 0.999
    
    # Move types: 0=Add, 1=Remove, 2=Swap
    while time.time() < deadline:
        if T < 0.001:
            T = 1.0
            # Restart or perturb
            # Perturb: remove 10 random elements
            for _ in range(10):
                if current_S:
                    y = random.choice(list(current_S))
                    indices = aps_containing_np[y]
                    current_S.remove(y)
                    current_in_S[y] = False
                    current_counts[indices] -= 1
                    current_size -= 1
            T = 1.0
        
        move_type = random.random()
        if move_type < 0.3:
            # Add
            candidates = [x for x in range(1, N + 1) if not current_in_S[x]]
            if not candidates: continue
            x = random.choice(candidates)
            indices = aps_containing_np[x]
            if np.any(current_counts[indices] == 3):
                continue # Invalid
            # Valid Add
            delta_size = 1
            # Accept?
            # Always accept if size increases
            current_S.add(x)
            current_in_S[x] = True
            current_counts[indices] += 1
            current_size += 1
            if current_size > best_size:
                best_size = current_size
                best_S = set(current_S)
                best_counts = current_counts.copy()
                best_in_S = current_in_S.copy()
                
        elif move_type < 0.5:
            # Remove
            if not current_S: continue
            y = random.choice(list(current_S))
            indices = aps_containing_np[y]
            delta_size = -1
            # Accept if T is high enough
            # Metropolis: accept if exp(-delta/T) > random()
            # delta is change in "energy". Here energy = -size.
            # So delta_energy = -delta_size = 1.
            # Accept if random() < exp(-1/T)
            if random.random() < np.exp(-1.0 / T):
                current_S.remove(y)
                current_in_S[y] = False
                current_counts[indices] -= 1
                current_size -= 1
            # Else reject (do nothing)
            
        else:
            # Swap
            if not current_S: continue
            y = random.choice(list(current_S))
            candidates = [x for x in range(1, N + 1) if not current_in_S[x]]
            if not candidates: continue
            x = random.choice(candidates)
            
            # Try to remove y and add x
            # 1. Remove y
            indices_y = aps_containing_np[y]
            current_in_S[y] = False
            current_counts[indices_y] -= 1
            
            # 2. Check if x can be added
            indices_x = aps_containing_np[x]
            if not np.any(current_counts[indices_x] == 3):
                # Valid swap
                # Accept?
                # Size change is 0.
                # We accept with some probability to explore, or always?
                # Let's accept if random() < 0.5 or if T is high
                if random.random() < 0.5:
                    current_S.remove(y)
                    current_S.add(x)
                    current_in_S[x] = True
                    current_counts[indices_x] += 1
                    # current_size unchanged
                else:
                    # Revert y
                    current_S.add(y)
                    current_in_S[y] = True
                    current_counts[indices_y] += 1
            else:
                # Invalid swap, revert y
                current_S.add(y)
                current_in_S[y] = True
                current_counts[indices_y] += 1
        
        T *= cooling
        
    return sorted(best_S)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    S_set = set(res)
    valid = True
    for ap in APs:
        if all(x in S_set for x in ap):
            valid = False
            break
    print(f"Valid: {valid}")
