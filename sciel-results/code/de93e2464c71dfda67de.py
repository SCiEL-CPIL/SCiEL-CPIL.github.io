"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0
    best_S = None
    best_len = N + 1
    
    def is_sidon(S):
        seen = set()
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            if 2 * a in seen:
                return False
            seen.add(2 * a)
            for b in L[i + 1:]:
                s = a + b
                if s in seen:
                    return False
                seen.add(s)
        return True

    while time.time() < deadline:
        S = []
        Sset = set()
        T = set()
        addable = list(range(1, N + 1))
        
        while addable:
            # Pick a random element from addable
            idx = random.randint(0, len(addable) - 1)
            x = addable[idx]
            
            # Add x to S
            S.append(x)
            Sset.add(x)
            
            # Update T with new sums involving x
            T.add(2 * x)
            for s in S[:-1]:
                T.add(x + s)
            
            # Update addable: remove x and any y that conflicts with x
            new_addable = []
            for y in addable:
                if y == x:
                    continue
                # Check if y is still addable
                if 2 * y in T:
                    continue
                conflict = False
                for s in S:
                    if y + s in T:
                        conflict = True
                        break
                if not conflict:
                    new_addable.append(y)
            addable = new_addable
            
        # End of one greedy run
        if is_sidon(S):
            if len(S) < best_len:
                best_len = len(S)
                best_S = S[:]
                if best_len <= 16:
                    break

    if best_S is None:
        # Fallback: simple greedy
        S = []
        T = set()
        for x in range(1, N + 1):
            added = [x + a for a in S] + [2 * x]
            if len(set(added)) == len(added) and not T.intersection(added):
                S.append(x)
                T.update(added)
        return S
        
    return best_S
# EVOLVE-BLOCK-END
