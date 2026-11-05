import random

N = 256

def construct():
    best_set = None
    best_size = N + 1
    
    for trial in range(300):
        S = set()
        sums = set()
        
        while True:
            U = []
            for x in range(1, N + 1):
                if x in S:
                    continue
                if 2 * x in sums:
                    continue
                conflict = False
                for a in S:
                    if x + a in sums:
                        conflict = True
                        break
                if not conflict:
                    U.append(x)
            
            if not U:
                break
            
            S_list = list(S)
            U_set = set(U)
            
            # Build BlockersByX
            # Map x -> set of y in U that are blocked by x
            BlockersByX = {}
            
            for y in U:
                # Type 1: 2y = x + a  => x = 2y - a
                for a in S_list:
                    x = 2 * y - a
                    if 1 <= x <= N and x not in S:
                        if x not in BlockersByX:
                            BlockersByX[x] = set()
                        BlockersByX[x].add(y)
                        
                # Type 2: y + x = s (s in sums) => x = s - y
                for s in sums:
                    x = s - y
                    if 1 <= x <= N and x not in S:
                        if x not in BlockersByX:
                            BlockersByX[x] = set()
                        BlockersByX[x].add(y)
                        
                # Type 3: y + a = 2x => x = (y + a) / 2
                for a in S_list:
                    val = y + a
                    if val % 2 == 0:
                        x = val // 2
                        if 1 <= x <= N and x not in S:
                            if x not in BlockersByX:
                                BlockersByX[x] = set()
                            BlockersByX[x].add(y)
                            
                # Type 4: y + a = x + b => x = y + a - b
                for a in S_list:
                    for b in S_list:
                        x = y + a - b
                        if 1 <= x <= N and x not in S:
                            if x not in BlockersByX:
                                BlockersByX[x] = set()
                            BlockersByX[x].add(y)
            
            # Find x in U with max |BlockersByX[x] \ {x}|
            best_x = None
            best_score = -1
            candidates = []
            
            for x in U:
                # x blocks y if y is in BlockersByX[x]
                # We want to count y in U, y != x
                score = 0
                if x in BlockersByX:
                    # BlockersByX[x] contains y in U
                    # We must exclude x itself if it's in the set (it shouldn't be, as we checked x not in S, but y is in U so y not in S. If y=x, it's in U. Does x block x? No, we are adding x. x is not in U anymore.
                    # The set BlockersByX[x] contains y such that adding x blocks y.
                    # We only care about y in U \ {x}.
                    # Since x is in U, and we are iterating x in U, we just subtract 1 if x is in the set.
                    # But x cannot block x in this context? 
                    # If x is added, x is in S. y=x is not in U.
                    # So we just count the size of the intersection with U \ {x}.
                    # Since BlockersByX[x] is built from U, all elements are in U.
                    # So score = len(BlockersByX[x]) - (1 if x in BlockersByX[x] else 0)
                    score = len(BlockersByX[x])
                    if x in BlockersByX[x]:
                        score -= 1
                else:
                    score = 0
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    candidates = [x]
                elif score == best_score:
                    candidates.append(x)
            
            chosen_x = random.choice(candidates)
            
            S.add(chosen_x)
            sums.add(2 * chosen_x)
            for a in S:
                if a != chosen_x:
                    sums.add(chosen_x + a)
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
            
    return best_set
