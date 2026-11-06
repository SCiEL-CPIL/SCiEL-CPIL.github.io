import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_set = []
    
    def check_and_add(S, sums_set, x):
        """Check if x can be added to S. Returns True if valid, updates sums_set if so."""
        # New sums: (x,x,x), (x,x,s) for s in S, (x,s,t) for s<=t in S
        new_sums = set()
        
        # (x,x,x)
        new_sums.add(3*x)
        
        # (x,x,s) for each s in S
        for s in S:
            new_sums.add(2*x + s)
        
        # (x,s,t) for s<=t in S (unordered pairs with replacement)
        for i in range(len(S)):
            for j in range(i, len(S)):
                new_sums.add(x + S[i] + S[j])
        
        # Check for collisions with existing sums
        if new_sums & sums_set:
            return False
        
        # Check for internal duplicates - but since we're using a set, 
        # we just need len(new_sums) == number of triples
        # Actually, we need to verify no duplicates within new_sums
        # Count expected number of new triples
        k = len(S)
        expected = 1 + k + k*(k+1)//2  # (xxx) + (xxs) + (xst)
        
        if len(new_sums) != expected:
            return False
        
        # Valid - add to sums_set
        sums_set.update(new_sums)
        return True
    
    def greedy_search(start_offset, step, max_val=4096):
        S = []
        sums_set = set()
        
        for x in range(start_offset, max_val + 1, step):
            if time.time() > deadline - 5:
                break
            if check_and_add(S, sums_set, x):
                S.append(x)
        
        return S
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Sequential from small values
    strategies.append(lambda: greedy_search(1, 1))
    
    # Strategy 2: Skip some small values
    strategies.append(lambda: greedy_search(1, 2))
    
    # Strategy 3: Start from different offsets
    for offset in range(1, 10):
        strategies.append(lambda o=offset: greedy_search(o, 1))
    
    # Run strategies and keep best
    for strat in strategies:
        if time.time() > deadline - 10:
            break
        result = strat()
        if len(result) > len(best_set):
            best_set = result
    
    # Randomized restarts with remaining time
    if time.time() < deadline - 15:
        random.seed(42)
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            # Random subset approach: pick random candidates and greedily build
            candidates = random.sample(range(1, 4097), min(200, 4096))
            candidates.sort()
            S = []
            sums_set = set()
            for x in candidates:
                if time.time() > deadline - 5:
                    break
                if check_and_add(S, sums_set, x):
                    S.append(x)
            if len(S) > len(best_set):
                best_set = S
    
    return best_set
