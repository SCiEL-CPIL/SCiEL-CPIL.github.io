import itertools

def construct():
    max_target = 128
    smooth = []
    for i in range(1, max_target + 1):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth.append(i)
            
    # Precompute antichains for k=1 to 5
    # dict: k -> { sum: terms }
    # We want the first (smallest k) found
    antichain_sums = {k: {} for k in range(1, 6)}
    
    def is_antichain(terms):
        for i in range(len(terms)):
            for j in range(i+1, len(terms)):
                a, b = terms[i], terms[j]
                if a < b:
                    if b % a == 0: return False
                else:
                    if a % b == 0: return False
        return True

    # k=1
    for x in smooth:
        if x <= max_target:
            antichain_sums[1][x] = [x]
            
    # k=2
    for comb in itertools.combinations(smooth, 2):
        s = sum(comb)
        if s <= max_target:
            if s not in antichain_sums[2]: # Keep first found? Or any? 
                 # We just need one valid one.
                 if is_antichain(comb):
                     antichain_sums[2][s] = list(comb)
                     
    # k=3
    for comb in itertools.combinations(smooth, 3):
        s = sum(comb)
        if s <= max_target:
            if s not in antichain_sums[3]:
                 if is_antichain(comb):
                     antichain_sums[3][s] = list(comb)
                     
    # k=4
    for comb in itertools.combinations(smooth, 4):
        s = sum(comb)
        if s <= max_target:
            if s not in antichain_sums[4]:
                 if is_antichain(comb):
                     antichain_sums[4][s] = list(comb)
                     
    # k=5
    for comb in itertools.combinations(smooth, 5):
        s = sum(comb)
        if s <= max_target:
            if s not in antichain_sums[5]:
                 if is_antichain(comb):
                     antichain_sums[5][s] = list(comb)
                     
    answers = []
    for n in range(1, max_target + 1):
        found = None
        for k in range(1, 6):
            if n in antichain_sums[k]:
                found = antichain_sums[k][n]
                break
        # If not found, try a fallback search or larger k
        # But 5 is likely enough. 
        if found is None:
             # Fallback: simple search
             for k in range(6, 11):
                 # This is slow, but only for edge cases if any
                 for comb in itertools.combinations([x for x in smooth if x <= n], k):
                     if sum(comb) == n and is_antichain(comb):
                         found = list(comb)
                         break
                 if found: break
        answers.append(found)
        
    return answers
