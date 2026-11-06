import random

def construct():
    # Start with the interval {1..32}
    current_set = set(range(1, 33))
    
    def score(s):
        diffs = set()
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                d = lst[j] - lst[i]
                if d > 0:
                    if lst[i] + 2*d in s:
                        diffs.add(d)
        return len(diffs)
    
    best_score = score(current_set)
    best_set = set(current_set)
    
    # Local search
    random.seed(42)
    for _ in range(10000):
        # Pick a random element to remove and a random number to add
        a = random.choice(list(current_set))
        b = random.randint(1, 512)
        if b in current_set:
            continue
        new_set = set(current_set)
        new_set.remove(a)
        new_set.add(b)
        new_score = score(new_set)
        if new_score > best_score:
            best_score = new_score
            best_set = new_set
            current_set = new_set
    
    return sorted(best_set)
