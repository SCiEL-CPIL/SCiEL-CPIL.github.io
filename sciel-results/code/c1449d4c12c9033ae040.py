"""Two Sidon sets with disjoint difference sets - SA with rebuild."""
# EVOLVE-BLOCK-START
import time
import random
import math

N = 256
MAX_SIZE = 64
DEADLINE = 55.0

def construct():
    start_time = time.time()
    
    best_A = set()
    best_B = set()
    best_size = 0
    
    A = set()
    B = set()
    sumA = 0
    sumB = 0
    diffA = 0
    diffB = 0
    
    def rebuild(side):
        nonlocal sumA, sumB, diffA, diffB
        if side == 0:
            sumA = 0
            diffA = 0
            for a in A:
                sumA |= (1 << (2*a))
                for b in A:
                    if b >= a:
                        sumA |= (1 << (a+b))
            for a in A:
                for b in A:
                    if a != b:
                        diffA |= (1 << abs(a-b))
        else:
            sumB = 0
            diffB = 0
            for a in B:
                sumB |= (1 << (2*a))
                for b in B:
                    if b >= a:
                        sumB |= (1 << (a+b))
            for a in B:
                for b in B:
                    if a != b:
                        diffB |= (1 << abs(a-b))

    def can_add(side, x):
        if side == 0:
            elems = A
            sums = sumA
            diffs = diffA
            other_diffs = diffB
        else:
            elems = B
            sums = sumB
            diffs = diffB
            other_diffs = diffA
            
        s2 = 2*x
        if (sums >> s2) & 1:
            return False
        for a in elems:
            s = x + a
            if (sums >> s) & 1:
                return False
            d = abs(x - a)
            if d > 0:
                if ((diffs >> d) & 1) or ((other_diffs >> d) & 1):
                    return False
        return True

    def do_add(side, x):
        if side == 0:
            A.add(x)
            rebuild(0)
        else:
            B.add(x)
            rebuild(1)

    def do_remove(side, x):
        if side == 0:
            A.remove(x)
            rebuild(0)
        else:
            B.remove(x)
            rebuild(1)

    # Greedy Init
    candidates = list(range(1, N+1))
    random.shuffle(candidates)
    for x in candidates:
        if len(A) < MAX_SIZE and can_add(0, x):
            do_add(0, x)
        if len(B) < MAX_SIZE and can_add(1, x):
            do_add(1, x)
            
    current_size = len(A) + len(B)
    if current_size > best_size:
        best_size = current_size
        best_A = set(A)
        best_B = set(B)
        
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    while time.time() - start_time < DEADLINE:
        # Random Move
        move_type = random.random()
        
        if move_type < 0.6:
            # Add
            side = random.randint(0, 1)
            if side == 0 and len(A) >= MAX_SIZE: continue
            if side == 1 and len(B) >= MAX_SIZE: continue
            
            x = random.randint(1, N)
            if (side == 0 and x in A) or (side == 1 and x in B):
                continue
                
            if can_add(side, x):
                do_add(side, x)
                new_size = len(A) + len(B)
                delta = new_size - current_size
                if delta > 0 or random.random() < math.exp(delta/T):
                    current_size = new_size
                    if current_size > best_size:
                        best_size = current_size
                        best_A = set(A)
                        best_B = set(B)
                else:
                    # Reject
                    do_remove(side, x)
        elif move_type < 0.8:
            # Remove
            side = random.randint(0, 1)
            if side == 0:
                if not A: continue
                x = random.choice(list(A))
                do_remove(0, x)
                new_size = len(A) + len(B)
                delta = new_size - current_size
                if delta > 0 or random.random() < math.exp(delta/T):
                    current_size = new_size
                else:
                    do_add(0, x)
            else:
                if not B: continue
                x = random.choice(list(B))
                do_remove(1, x)
                new_size = len(A) + len(B)
                delta = new_size - current_size
                if delta > 0 or random.random() < math.exp(delta/T):
                    current_size = new_size
                else:
                    do_add(1, x)
        else:
            # Swap: Remove x from side, Add y to side
            side = random.randint(0, 1)
            if side == 0:
                if not A: continue
                x = random.choice(list(A))
                y = random.randint(1, N)
                if y in A: continue
                
                # Temporarily remove x
                do_remove(0, x)
                
                if can_add(0, y):
                    do_add(0, y)
                    new_size = len(A) + len(B)
                    delta = new_size - current_size
                    # Size is same. Accept with prob 0.5 to explore
                    if random.random() < 0.5:
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_A = set(A)
                            best_B = set(B)
                    else:
                        # Reject: Remove y, Re-add x
                        do_remove(0, y)
                        do_add(0, x)
                else:
                    # Reject: Re-add x
                    do_add(0, x)
            else:
                if not B: continue
                x = random.choice(list(B))
                y = random.randint(1, N)
                if y in B: continue
                
                do_remove(1, x)
                
                if can_add(1, y):
                    do_add(1, y)
                    new_size = len(A) + len(B)
                    delta = new_size - current_size
                    if random.random() < 0.5:
                        current_size = new_size
                        if current_size > best_size:
                            best_size = current_size
                            best_A = set(A)
                            best_B = set(B)
                    else:
                        do_remove(1, y)
                        do_add(1, x)
                else:
                    do_add(1, x)

        T *= cooling
        if T < T_min:
            T = T_min
            
    return [list(best_A), list(best_B)]
# EVOLVE-BLOCK-END
