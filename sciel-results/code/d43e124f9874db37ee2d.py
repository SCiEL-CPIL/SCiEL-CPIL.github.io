import random
import time

N = 256

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 60s, but task says 30s. 
    # We aim to finish well before 30s to be safe.
    deadline = start_time + 25.0
    
    best_set = None
    best_size = float('inf')
    
    def greedy_blocking_restart(seed):
        random.seed(seed)
        S = []
        sums = set()
        # Use a list for S for fast iteration, set for membership if needed (not really needed for sums)
        
        while True:
            # Find best candidate
            best_x = -1
            best_score = -1
            tied_candidates = []
            
            # Iterate through all potential x
            for x in range(1, N + 1):
                if x in S:
                    continue
                
                # Check compatibility: 2x not in sums, and x+a not in sums for all a in S
                compatible = True
                if 2 * x in sums:
                    compatible = False
                else:
                    for a in S:
                        if x + a in sums:
                            compatible = False
                            break
                
                if not compatible:
                    continue
                
                # Calculate score: how many other compatible candidates y does x block?
                # y is blocked if x + y is in sums.
                # Note: We only care about y that are currently compatible.
                # If y is not compatible, it's already out.
                # But the "blocking" heuristic usually considers all potential future elements.
                # Let's count y in [1, N] \ S such that x + y is in sums.
                # This is a proxy for "dominance".
                score = 0
                for y in range(1, N + 1):
                    if y == x or y in S:
                        continue
                    # We don't strictly need y to be compatible right now for the "block" count,
                    # but it's a stronger signal if y is a current candidate.
                    # However, checking compatibility for y is expensive.
                    # Let's just check if x+y is in sums.
                    if x + y in sums:
                        score += 1
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    tied_candidates = [x]
                elif score == best_score:
                    tied_candidates.append(x)
            
            if not tied_candidates:
                break
            
            x_star = random.choice(tied_candidates)
            S.append(x_star)
            sums.add(2 * x_star)
            for a in S[:-1]:
                sums.add(x_star + a)
            
            # Time check
            if time.time() > deadline:
                break
        
        return S

    # Run multiple restarts
    # Since N=256 is small, we can afford a decent number of restarts.
    # The inner loop is O(N * |S| + N^2). 
    # |S| is typically ~10-20. N^2 is 65k.
    # So ~100k ops per step. ~20 steps. ~2M ops per restart.
    # Python does ~10M-50M simple ops/sec. So ~0.1s per restart.
    # We can do ~200 restarts in 20s.
    
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        S = greedy_blocking_restart(i)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
            
    # Add a few random order restarts for diversity
    for i in range(50):
        if time.time() > deadline:
            break
        random.seed(1000 + i)
        S = []
        sums = set()
        available = list(range(1, N + 1))
        random.shuffle(available)
        
        for x in available:
            if x in S:
                continue
            # Check compatibility
            compatible = True
            if 2 * x in sums:
                compatible = False
            else:
                for a in S:
                    if x + a in sums:
                        compatible = False
                        break
            if compatible:
                S.append(x)
                sums.add(2 * x)
                for a in S[:-1]:
                    sums.add(x + a)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S

    if best_set is None:
        # Fallback: simple ascending
        S = []
        sums = set()
        for x in range(1, N + 1):
            compatible = True
            if 2 * x in sums:
                compatible = False
            else:
                for a in S:
                    if x + a in sums:
                        compatible = False
                        break
            if compatible:
                S.append(x)
                sums.add(2 * x)
                for a in S[:-1]:
                    sums.add(x + a)
        best_set = S

    return best_set
