"""Combined program: fast greedy construction with multi-move local search."""
# EVOLVE-BLOCK-START
import time
import random

def _can_add(S_list, x):
    """
    Check if x can be added to S (given as S_list).
    For every three distinct members a, b, c in S ∪ {x}, a does not divide b*c.
    """
    n = len(S_list)
    
    # 1. x divides a*b for distinct a,b in S
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            if (a * b) % x == 0:
                return False
    
    # 2. a divides x*b for distinct a,b in S
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            if (x * b) % a == 0:
                return False
    
    return True

def _is_valid(S_list):
    """
    Check if S is a valid set.
    For every three distinct members a, b, c in S, a does not divide b*c.
    """
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if i == j:
                continue
            b = S_list[j]
            for k in range(n):
                if i == k or j == k:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _greedy_construct(order, deadline):
    """
    Greedily construct a set by iterating through 'order' and adding
    if the resulting set remains valid.
    """
    S = set()
    S_list = []
    
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(S_list, x):
            S.add(x)
            S_list.append(x)
            
    return S, S_list

def construct():
    deadline = time.time() + 55.0
    start_time = time.time()
    
    all_nums = list(range(2, 257))
    
    best_set = set()
    best_size = 0
    
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True
            
    primes = [x for x in all_nums if is_prime(x)]
    composites = [x for x in all_nums if not is_prime(x)]
    
    orders = [
        list(range(2, 257)),
        list(range(256, 1, -1)),
        primes + composites,
        composites + primes,
    ]
    
    rng = random.Random(42)
    for _ in range(30):
        shuffled = all_nums.copy()
        rng.shuffle(shuffled)
        orders.append(shuffled)
        
    for order in orders:
        if time.time() > deadline:
            break
        S, S_list = _greedy_construct(order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    current_set = best_set.copy()
    current_list = list(current_set)
    current_size = best_size
    
    non_members = [x for x in all_nums if x not in current_set]
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2.0:
        if not non_members:
            break
        improved = False
        
        r = random.random()
        if r < 0.6:
            # Try random Add
            idx = random.randint(0, len(non_members) - 1)
            x = non_members[idx]
            if _can_add(current_list, x):
                current_set.add(x)
                current_list.append(x)
                current_size += 1
                non_members.pop(idx)
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set.copy()
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        elif r < 0.85:
            # 1-for-1 Swap
            if len(current_set) > 0:
                y = random.choice(current_list)
                idx = random.randint(0, len(non_members) - 1)
                x = non_members[idx]
                temp_list = [v for v in current_list if v != y]
                if _can_add(temp_list, x):
                    current_set.remove(y)
                    current_set.add(x)
                    current_list = temp_list + [x]
                    non_members[idx] = y
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1
        else:
            # 2-for-2 Swap: remove two, add two
            if len(current_set) >= 2 and len(non_members) >= 2:
                y1, y2 = random.sample(current_list, 2)
                idx1, idx2 = random.sample(range(len(non_members)), 2)
                x1, x2 = non_members[idx1], non_members[idx2]
                temp_list = [v for v in current_list if v != y1 and v != y2]
                if _can_add(temp_list, x1) and _can_add(temp_list + [x1], x2):
                    current_set.discard(y1)
                    current_set.discard(y2)
                    current_set.add(x1)
                    current_set.add(x2)
                    current_list = temp_list + [x1, x2]
                    non_members[idx1] = y1
                    non_members[idx2] = y2
                    no_improve_count = 0
                    improved = True
                else:
                    no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            if random.random() < 0.5:
                current_set = best_set.copy()
                current_list = list(current_set)
                current_size = best_size
                non_members = [x for x in all_nums if x not in current_set]
    
    result = list(best_set)
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
