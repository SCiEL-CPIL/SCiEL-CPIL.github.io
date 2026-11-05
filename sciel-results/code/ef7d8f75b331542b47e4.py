import time
import random
import math

def construct():
    start_time = time.time()
    # Safety margin: 60s limit, use 55s to be safe
    deadline = start_time + 55.0
    
    N = 256
    # in_set[i] is True if i is in the current set A
    # We use a list of booleans for O(1) access
    
    def get_score(A_list):
        return sum(1.0 / x for x in A_list)

    def can_add(y, in_set, A_list):
        """
        Check if y can be added to A (represented by in_set and A_list) 
        without forming a 3-AP.
        A_list is a list of elements currently in A.
        """
        # Case 1: y is the middle term.
        # a, y, c => a + c = 2y => c = 2y - a
        # For each a in A, check if c is in A.
        for a in A_list:
            c = 2 * y - a
            if 1 <= c <= N and in_set[c]:
                return False
        
        # Case 2: y is an endpoint.
        # y, b, c => y + c = 2b => c = 2b - y
        # a, b, y => a + y = 2b => a = 2b - y
        # In both cases, we check if there exists b in A such that 2b - y is in A.
        # Let z = 2b - y. If z is in A and z != y, we have a 3-AP (z, b, y) or (y, b, z).
        # Note: if z == y, then 2b = 2y => b = y, but y is not in A.
        for b in A_list:
            z = 2 * b - y
            if 1 <= z <= N and in_set[z]:
                return False
        
        return True

    def greedy_init():
        """
        Greedy construction: add smallest available numbers first if valid.
        This maximizes reciprocal sum quickly.
        """
        in_set = [False] * (N + 1)
        A_list = []
        for x in range(1, N + 1):
            if can_add(x, in_set, A_list):
                in_set[x] = True
                A_list.append(x)
        return A_list, in_set

    # Initialize with greedy solution
    A_list, in_set = greedy_init()
    current_score = get_score(A_list)
    
    # Track best solution
    best_A_list = list(A_list)
    best_score = current_score
    
    # Simulated Annealing Parameters
    T_start = 0.05
    T_end = 1e-6
    
    all_numbers = list(range(1, N + 1))
    
    # To speed up random selection from A_list and not in A_list
    # We can maintain a list of available numbers? 
    # Since N=256 is small, just picking random indices is fine.
    
    while time.time() < deadline:
        elapsed = time.time() - start_time
        total_time = deadline - start_time
        if elapsed >= total_time:
            break
            
        # Fraction of time passed
        f = elapsed / total_time
        # T decays from T_start to T_end
        # Use exponential decay
        T = T_start * math.exp(-10 * f)
        
        # Generate a random neighbor
        move_type = random.random()
        
        new_A_list = None
        new_in_set = None
        delta_score = 0.0
        valid_move = False
        
        if move_type < 0.7:
            # Swap: Pick x in A, y not in A
            if not A_list:
                continue
            x = random.choice(A_list)
            # Pick y from 1..N not in A
            # Efficient way: pick random y, if in A, retry a few times or just skip
            # Since N is small, we can pick from a precomputed complement list if we maintained it,
            # but maintaining it is overhead. Just pick random and check.
            y = random.randint(1, N)
            if in_set[y]:
                continue
            
            # Check if x can be removed and y added
            # Temporarily remove x
            new_in_set = list(in_set)
            new_in_set[x] = False
            new_A_list = list(A_list)
            new_A_list.remove(x)
            
            if can_add(y, new_in_set, new_A_list):
                new_in_set[y] = True
                new_A_list.append(y)
                valid_move = True
                delta_score = (1.0 / y) - (1.0 / x)
            else:
                # Move not possible
                valid_move = False
                
        elif move_type < 0.85:
            # Add: Pick y not in A
            y = random.randint(1, N)
            if in_set[y]:
                continue
            
            if can_add(y, in_set, A_list):
                new_in_set = list(in_set)
                new_in_set[y] = True
                new_A_list = list(A_list)
                new_A_list.append(y)
                valid_move = True
                delta_score = 1.0 / y
            else:
                valid_move = False
        else:
            # Remove: Pick x in A
            if not A_list:
                continue
            x = random.choice(A_list)
            new_in_set = list(in_set)
            new_in_set[x] = False
            new_A_list = list(A_list)
            new_A_list.remove(x)
            valid_move = True
            delta_score = -(1.0 / x)
            
        if not valid_move:
            continue
            
        # Acceptance criterion
        if delta_score > 0:
            accept = True
        else:
            # Metropolis criterion
            if T > 0:
                # Avoid overflow/underflow issues
                if delta_score / T < -100:
                    accept_prob = 0.0
                else:
                    accept_prob = math.exp(delta_score / T)
                accept = (random.random() < accept_prob)
            else:
                accept = False
                
        if accept:
            A_list = new_A_list
            in_set = new_in_set
            current_score += delta_score
            
            if current_score > best_score:
                best_score = current_score
                best_A_list = list(A_list)
        
    return best_A_list
