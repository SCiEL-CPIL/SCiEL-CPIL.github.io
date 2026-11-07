import time
import random
import math

N = 128
MAX_TIME = 85.0

def _can_add(s_set, x):
    """
    Check if adding x to s_set results in a valid set.
    s_set is a set of ints.
    Returns True if s_set U {x} is valid.
    """
    # x is not in s_set
    # We need to check if x creates a forbidden configuration.
    # A forbidden configuration is {a, b, c, a+b, a+c, b+c} with a,b,c distinct.
    # Since s_set is valid, any new conflict must involve x.
    
    # Case 1: x is one of the elements a, b, or c.
    # Let x = a. Then b, c must be in s_set.
    # The sums a+b, a+c, b+c must be in s_set U {x}.
    # a+b = x+b. a+c = x+c. b+c.
    # Since b, c are in s_set (and distinct from x), x+b and x+c cannot be x.
    # So we need x+b in s_set, x+c in s_set, and b+c in s_set.
    # Also b+c must be distinct from a,b,c? The problem says "six displayed values need not be mutually distinct".
    # But a,b,c are distinct members.
    # So we iterate pairs (b, c) in s_set.
    # To avoid O(N^3) in worst case, we can optimize.
    # But N=128, |s_set| ~ 60. 60^2 = 3600. 3600 * 3 lookups is fine.
    
    s_list = list(s_set)
    n = len(s_list)
    
    for i in range(n):
        b = s_list[i]
        if b + x > 128: # Optimization? No, sums can be anything, but must be in set.
             pass
        for j in range(i + 1, n):
            c = s_list[j]
            # Check if {x, b, c} is a valid triple in the new set
            # Sums: x+b, x+c, b+c
            if (x + b in s_set) and (x + c in s_set) and (b + c in s_set):
                return False
                
    # Case 2: x is one of the sums.
    # Let x = a + b. Then a, b, c must be in s_set.
    # The other sums a+c and b+c must be in s_set U {x}.
    # Since a, b, c are in s_set, a+c and b+c are not x (unless c=0 or c=a/b which is distinct).
    # Actually, if a+c = x, then a+c = a+b => c=b, but c distinct from b. So a+c != x.
    # So we need a+c in s_set and b+c in s_set.
    
    # Iterate pairs (a, b) in s_set such that a + b == x.
    # For each such pair, iterate c in s_set (c distinct from a, b).
    # Check if a+c in s_set and b+c in s_set.
    
    for i in range(n):
        a = s_list[i]
        b = x - a
        if b not in s_set:
            continue
        if a >= b: # Ensure a < b to avoid duplicate pairs
            continue
        # a and b are in s_set, a+b=x
        # Now find c in s_set, c != a, c != b
        for k in range(n):
            c = s_list[k]
            if c == a or c == b:
                continue
            if (a + c in s_set) and (b + c in s_set):
                return False
                
    return True

def _construct_greedy_random():
    s_set = set()
    # Shuffle candidates
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    for x in candidates:
        if _can_add(s_set, x):
            s_set.add(x)
    return s_set

def construct():
    start_time = time.time()
    best_s = set()
    best_size = 0
    
    # Multiple restarts of Simulated Annealing
    # Each restart starts from a random greedy solution
    
    num_restarts = 20 # Max number of restarts, limited by time
    
    for r in range(num_restarts):
        if time.time() - start_time > MAX_TIME - 10.0:
            break
            
        # Initialize
        current_s = _construct_greedy_random()
        current_size = len(current_s)
        
        if current_size > best_size:
            best_s = set(current_s)
            best_size = current_size
            
        # Simulated Annealing
        # Temperature schedule
        T = 10.0
        T_min = 0.01
        cooling_rate = 0.95
        
        # Max steps per restart
        max_steps = 20000
        
        for step in range(max_steps):
            if time.time() - start_time > MAX_TIME - 5.0:
                break
                
            if T < T_min:
                break
                
            # Generate move
            # 60% Add, 40% Remove (to allow shrinking)
            move_type = random.random()
            
            new_s = None
            
            if move_type < 0.6:
                # Try to add a random element
                candidates = [x for x in range(1, N + 1) if x not in current_s]
                if not candidates:
                    continue
                x = random.choice(candidates)
                if _can_add(current_s, x):
                    new_s = current_s | {x}
                    new_size = current_size + 1
                else:
                    # Invalid add, reject move? Or try another?
                    # For efficiency, just reject this step
                    continue
            else:
                # Try to remove a random element
                if not current_s:
                    continue
                x = random.choice(list(current_s))
                new_s = current_s - {x}
                new_size = current_size - 1
                
            if new_s is None:
                continue
                
            delta = new_size - current_size
            
            # Acceptance
            if delta > 0:
                # Always accept improvement
                current_s = new_s
                current_size = new_size
                if current_size > best_size:
                    best_s = set(current_s)
                    best_size = current_size
            elif delta == 0:
                # Accept with some probability to explore
                if random.random() < 0.5:
                    current_s = new_s
                    # current_size unchanged
            else:
                # delta < 0, accept with probability exp(delta / T)
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_s = new_s
                    current_size = new_size
            
            T *= cooling_rate
            
    # Final check
    if not best_s:
        # Fallback
        best_s = set(range(63, 129))
        
    # Verify best_s is valid (paranoia)
    # We assume it is, but let's do a quick check if time permits
    # For now, return best_s
    
    return [int(x) for x in best_s]
