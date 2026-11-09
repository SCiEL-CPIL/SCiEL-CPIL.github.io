import random

def construct():
    N = 256
    
    # Base 3 set
    base3_set = set()
    for i in range(1, N+1):
        # Check if base 3 representation has only 0 and 1
        n = i
        valid = True
        while n > 0:
            if n % 3 == 2:
                valid = False
                break
            n //= 3
        if valid:
            base3_set.add(i)
            
    # Greedy set
    greedy_set = set()
    in_set_g = [False] * (N + 1)
    for x in range(1, N + 1):
        valid = True
        for a in range(1, x):
            if in_set_g[a]:
                # a, b, x
                s = a + x
                if s % 2 == 0:
                    b = s // 2
                    if a < b < x and in_set_g[b]:
                        valid = False
                        break
                # a, x, b
                b = 2*x - a
                if b > x and b <= N and in_set_g[b]:
                    valid = False
                    break
        if valid:
            in_set_g[x] = True
            greedy_set.add(x)
            
    # Pick the better one
    def score(s):
        return sum(1.0/i for i in s)
        
    s1 = score(base3_set)
    s2 = score(greedy_set)
    
    if s1 > s2:
        current_set = list(base3_set)
    else:
        current_set = list(greedy_set)
        
    in_set = [False] * (N + 1)
    for x in current_set:
        in_set[x] = True
        
    current_score = score(current_set)
    
    def can_add(x):
        # a, b, x
        for a in range(1, x):
            if in_set[a]:
                s = a + x
                if s % 2 == 0:
                    b = s // 2
                    if a < b < x and in_set[b]:
                        return False
                b = 2*x - a
                if b > x and b <= N and in_set[b]:
                    return False
        # x, a, b
        for a in range(x+1, N+1):
            if in_set[a]:
                b = 2*a - x
                if b > a and b <= N and in_set[b]:
                    return False
        return True

    # Local Search
    random.seed(42)
    for _ in range(50000):
        if random.random() < 0.7:
            # Try add
            y = random.randint(1, N)
            if not in_set[y]:
                if can_add(y):
                    in_set[y] = True
                    current_set.append(y)
                    current_score += 1.0/y
        else:
            # Try remove and add
            if current_set:
                x = random.choice(current_set)
                in_set[x] = False
                current_set.remove(x)
                current_score -= 1.0/x
                
                y = random.randint(1, N)
                if not in_set[y]:
                    if can_add(y):
                        in_set[y] = True
                        current_set.append(y)
                        current_score += 1.0/y
                    else:
                        # Revert
                        in_set[x] = True
                        current_set.append(x)
                        current_score += 1.0/x
                else:
                    # Revert
                    in_set[x] = True
                    current_set.append(x)
                    current_score += 1.0/x
                    
    return current_set
