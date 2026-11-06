def construct():
    import time
    from random import Random
    
    deadline = time.time() + 145
    
    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def can_add(S_set, x):
        for a in S_set:
            if (a + x) in S_set:
                return False
            if (x - a) in S_set:
                return False
        if (2 * x) in S_set:
            return False
        return True
    
    def greedy_build(elements, order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S
    
    def extend(S, elements):
        S = set(S)
        for x in elements:
            if x not in S and can_add(S, x):
                S.add(x)
        return S
    
    def refine(S, elements, max_rounds=50):
        S = set(S)
        for _ in range(max_rounds):
            improved = False
            for rem in list(S):
                S.remove(rem)
                for add in elements:
                    if add not in S and can_add(S, add):
                        S.add(add)
                        S = extend(S, elements)
                        improved = True
                        break
                if improved:
                    break
                S.add(rem)
            if not improved:
                break
        return S
    
    def solve_one(elements, rng):
        best = set()
        best_size = 0
        n = len(elements)
        
        orderings = [
            sorted(elements),
            sorted(elements, reverse=True),
            sorted(elements, key=abs),
            sorted(elements, key=abs, reverse=True),
        ]
        for _ in range(30):
            orderings.append(list(rng.sample(elements, n)))
        
        for order in orderings:
            S = greedy_build(elements, order)
            S = refine(S, elements)
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
        
        while time.time() < deadline - 5:
            order = list(rng.sample(elements, n))
            S = greedy_build(elements, order)
            S = refine(S, elements)
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
        
        return sorted(best)
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        elements = instance(size, seed)
        rng = Random(seed + 1000)
        result = solve_one(elements, rng)
        results.append(result)
    
    return results
