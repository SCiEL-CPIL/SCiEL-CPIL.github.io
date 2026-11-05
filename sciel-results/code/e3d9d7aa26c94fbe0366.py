"""Bitmask-based Simulated Annealing for 4-term-progression-free subsets of [1,512]."""
import time
import random
import numpy as np

# Constants
N = 512
DEADLINE = time.time() + 85  # Safety margin

def precompute_conflicts():
    """
    Precompute for each element x and each potential difference d, 
    the bitmask of elements that would form a 4-AP with x if added.
    
    Actually, for Simulated Annealing, it's better to precompute the 
    'conflict masks' for each x. 
    conflict_mask[x] is a bitmask where bit y is set if adding x 
    to a set containing y would create a 4-AP.
    
    Wait, a 4-AP involves 4 elements. 
    If we have a set S, and we want to add x, we need to check if 
    there exist a, b, c in S such that {a, b, c, x} is a 4-AP.
    
    This is equivalent to checking if x forms a 4-AP with any 3 existing elements.
    
    Let's precompute for each x, the list of "triplets" (a, b, c) such that 
    {a, b, c, x} is a 4-AP.
    Then, adding x is safe iff for all such triplets, S does not contain all of a, b, c.
    
    Alternatively, we can use a bitwise trick.
    For a fixed x, the condition that x creates a 4-AP with S is:
    There exists d > 0 such that:
    1. x = a + 3d => a = x-3d, need {x-3d, x-2d, x-d} in S
    2. x = a + 2d => a = x-2d, need {x-2d, x-d, x+d} in S
    3. x = a + d  => a = x-d,  need {x-d, x+d, x+2d} in S
    4. x = a      => a = x,    need {x+d, x+2d, x+3d} in S
    
    We can precompute for each x a list of bitmasks M_{x,k} such that 
    if (S & M_{x,k}) == M_{x,k}, then x creates a conflict.
    
    Since N=512, we can use two 256-bit integers or a Python int (arbitrary precision).
    Python ints are fast enough for bitwise ops on 512 bits.
    """
    # conflicts[x] will be a list of bitmasks. 
    # If any bitmask is a subset of S, then x cannot be added.
    conflicts = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        masks = set() # Avoid duplicate masks for same x
        
        # Case 1: x is the 4th term (a, a+d, a+2d, x)
        # x = a + 3d => a = x - 3d
        # d can range from 1 to (x-1)//3
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            b = a + d
            c = a + 2 * d
            # Elements: a, b, c
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            masks.add(mask)
            
        # Case 2: x is the 3rd term (a, a+d, x, a+3d)
        # x = a + 2d => a = x - 2d
        # d can range from 1 to (x-1)//2
        for d in range(1, (x - 1) // 2 + 1):
            a = x - 2 * d
            b = a + d
            c = a + 3 * d
            if c > N: continue
            # Elements: a, b, c
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            masks.add(mask)
            
        # Case 3: x is the 2nd term (a, x, a+2d, a+3d)
        # x = a + d => a = x - d
        # d can range from 1 to x-1
        # Also need a+3d <= N => x-d+3d = x+2d <= N => d <= (N-x)//2
        for d in range(1, min(x, (N - x) // 2) + 1):
            a = x - d
            b = x + d
            c = x + 2 * d
            # Elements: a, b, c
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            masks.add(mask)
            
        # Case 4: x is the 1st term (x, x+d, x+2d, x+3d)
        # d can range from 1 to (N-x)//3
        for d in range(1, (N - x) // 3 + 1):
            a = x + d
            b = x + 2 * d
            c = x + 3 * d
            # Elements: a, b, c
            mask = (1 << (a-1)) | (1 << (b-1)) | (1 << (c-1))
            masks.add(mask)
            
        conflicts[x] = list(masks)
        
    return conflicts

# Precompute at module level to save time in construct()
CONFLICTS = precompute_conflicts()

def is_valid_addition(state, x):
    """Check if x can be added to state (int bitmask)."""
    # x is 1-indexed, so bit index is x-1
    # If x is already in state, it's not an addition, but we assume we only call this for x not in state.
    for mask in CONFLICTS[x]:
        if (state & mask) == mask:
            return False
    return True

def count_valid_additions(state):
    """Count how many elements can be added to state."""
    count = 0
    # Iterate over all x not in state
    # state is an int. We can iterate 1..N
    for x in range(1, N + 1):
        if not (state & (1 << (x-1))):
            if is_valid_addition(state, x):
                count += 1
    return count

def greedy_init():
    """Fast greedy initialization using bitmasks."""
    state = 0
    # Order: try elements that are "least constrained" first? 
    # Or just random order for diversity?
    # Let's do a few greedy passes with different orders and pick the best.
    best_state = 0
    best_size = 0
    
    orders = [
        list(range(1, N+1)),
        list(range(N, 0, -1)),
        [x for x in range(1, N+1) if x % 2 == 1],
        [x for x in range(1, N+1) if x % 2 == 0],
    ]
    
    # Add random orders
    for _ in range(10):
        o = list(range(1, N+1))
        random.shuffle(o)
        orders.append(o)
        
    for order in orders:
        state = 0
        size = 0
        for x in order:
            if is_valid_addition(state, x):
                state |= (1 << (x-1))
                size += 1
        if size > best_size:
            best_size = size
            best_state = state
            
    return best_state, best_size

def simulated_annealing(start_state, start_size, time_budget=80.0):
    """Simulated Annealing to maximize size."""
    current_state = start_state
    current_size = start_size
    best_state = start_state
    best_size = start_size
    
    # Initial temperature
    # Energy function: -size (we want to minimize energy, so maximize size)
    # Delta E = -Delta Size
    # If Delta Size > 0, Delta E < 0 (always accept)
    # If Delta Size < 0, Delta E > 0 (accept with prob exp(-dE/T))
    # Let's define E = -size.
    # Move: Add x (if valid) -> Size +1 -> E -1. Always accept.
    # Move: Remove x -> Size -1 -> E +1. Accept with prob exp(-1/T).
    # Move: Swap x (in) for y (out) -> Size 0 -> E 0. Accept with prob 1? 
    #   Actually, swap doesn't change size, but might allow further additions.
    #   Let's stick to Add/Remove for simplicity, or Add/Remove/Remove+Add.
    
    T = 1.0
    cooling_rate = 0.9995
    
    start_time = time.time()
    steps = 0
    
    while time.time() - start_time < time_budget:
        # Choose a move
        # 1. Try to add a random element
        # 2. Try to remove a random element
        # 3. Try to swap
        
        move_type = random.random()
        
        if move_type < 0.5:
            # Try Add
            # Pick random x not in state
            # Efficiently: generate random 1..N, if in state, retry or skip
            # Better: maintain a list of available elements? 
            # For N=512, just picking random is fine.
            for _ in range(10): # Limit retries
                x = random.randint(1, N)
                if not (current_state & (1 << (x-1))):
                    break
            else:
                continue # State is full? Unlikely.
                
            if is_valid_addition(current_state, x):
                new_state = current_state | (1 << (x-1))
                new_size = current_size + 1
                # Always accept improvement
                current_state = new_state
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_state = current_state
            # If invalid, do nothing (or penalize? No, just skip)
            
        elif move_type < 0.8:
            # Try Remove
            # Pick random x in state
            # To pick random set bit, we can use a trick or just scan
            # For N=512, scanning is fast.
            # Or maintain a list of set bits? 
            # Let's just pick random 1..N and check if set.
            for _ in range(10):
                x = random.randint(1, N)
                if (current_state & (1 << (x-1))):
                    break
            else:
                continue # State is empty?
                
            new_state = current_state & ~(1 << (x-1))
            new_size = current_size - 1
            # Accept with probability exp(-dE/T) where dE = 1 (size decreases by 1, so E increases by 1)
            # P = exp(-1/T)
            if random.random() < np.exp(-1.0 / T):
                current_state = new_state
                current_size = new_size
                
        else:
            # Try Swap: Remove x, Add y
            # Pick x in state, y not in state
            for _ in range(10):
                x = random.randint(1, N)
                if (current_state & (1 << (x-1))):
                    break
            else:
                continue
                
            for _ in range(10):
                y = random.randint(1, N)
                if not (current_state & (1 << (y-1))):
                    break
            else:
                continue
                
            # Remove x, then check if y can be added
            temp_state = current_state & ~(1 << (x-1))
            if is_valid_addition(temp_state, y):
                new_state = temp_state | (1 << (y-1))
                new_size = current_size # Size unchanged
                # Since size is unchanged, we accept with prob 1? 
                # Or maybe we should accept with some probability to explore?
                # If we always accept, we might cycle. 
                # But swaps can unlock additions.
                # Let's accept with prob 0.5 to avoid cycling? 
                # Or just accept, as it's neutral in energy.
                if random.random() < 0.5:
                    current_state = new_state
                    # Size same, so best_size unchanged unless we add later
                    # But we might have enabled an addition. 
                    # We don't check for additions immediately here.
        
        # Cooling
        T *= cooling_rate
        if T < 0.01:
            T = 0.01 # Floor to allow some exploration
            
        steps += 1
        
        # Periodic restart or reheat?
        # If stuck (no improvement for many steps), reheat?
        # Simple: just keep cooling.
        
    return best_state, best_size

def construct():
    global DEADLINE
    start_time = time.time()
    DEADLINE = start_time + 85
    
    # 1. Greedy Initialization
    init_state, init_size = greedy_init()
    
    # 2. Simulated Annealing
    best_state, best_size = simulated_annealing(init_state, init_size, time_budget=75.0)
    
    # 3. Final Greedy Polish
    # Try to add any remaining elements in a smart order
    # Convert state to set for easier manipulation in final pass?
    # Or stick to bitmasks.
    
    # Let's do a final greedy pass on the best_state
    current_state = best_state
    current_size = best_size
    
    # Try adding elements in order of increasing "conflict count" or just random?
    # Random is safer to avoid bias.
    # Let's try multiple orders.
    for _ in range(5):
        order = list(range(1, N+1))
        random.shuffle(order)
        for x in order:
            if time.time() > DEADLINE - 1:
                break
            if not (current_state & (1 << (x-1))):
                if is_valid_addition(current_state, x):
                    current_state |= (1 << (x-1))
                    current_size += 1
        if current_size > best_size:
            best_size = current_size
            best_state = current_state
            
    # Convert to list
    result = []
    for x in range(1, N + 1):
        if best_state & (1 << (x-1)):
            result.append(x)
            
    # Verification (optional but good for safety)
    # The bitmask logic is exact, so it should be valid.
    # But let's double check with a simple set-based check if time permits.
    if time.time() < DEADLINE - 2:
        s = set(result)
        # Simple check
        valid = True
        for a in s:
            for b in s:
                d = b - a
                if d > 0:
                    if (a + 2*d) in s and (a + 3*d) in s:
                        valid = False
                        break
            if not valid:
                break
        if not valid:
            # Fallback: return empty or a known valid set
            # Base 3 construction
            b3 = []
            for x in range(1, N+1):
                t = x
                ok = True
                while t > 0:
                    if t % 3 == 2:
                        ok = False
                        break
                    t //= 3
                if ok:
                    b3.append(x)
            return b3
            
    return result

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    print(f"Size: {len(res)}, Time: {time.time()-t0:.2f}s")
