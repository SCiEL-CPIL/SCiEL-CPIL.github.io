"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin
    
    # Generate smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    # Precompute divisibility relations
    # divs[a] = set of b > a such that a divides b
    # We need to check antichain: no term divides another
    # For a set of terms, check all pairs: for each pair (a,b) with a<b, b%a != 0
    
    def is_antichain(terms):
        """Check if terms form a divisibility antichain."""
        # Sort to ensure we only check a < b
        ts = sorted(terms)
        for i in range(len(ts)):
            a = ts[i]
            for j in range(i+1, len(ts)):
                b = ts[j]
                if b % a == 0:
                    return False
        return True
    
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: use a simple valid representation if we're running out of time
            # For now, just append an empty list as placeholder (shouldn't happen)
            answers.append([])
            continue
        
        answer = None
        # Try 1 term first (fastest)
        if n in smooth:
            answer = [n]
        else:
            # Try 2 terms
            if answer is None:
                for a, b in combinations([x for x in smooth if x <= n], 2):
                    if a + b == n and b % a != 0:
                        answer = [a, b]
                        break
            # Try 3 terms
            if answer is None:
                for terms in combinations([x for x in smooth if x <= n], 3):
                    if sum(terms) == n and is_antichain(terms):
                        answer = list(terms)
                        break
            # Try 4 terms
            if answer is None:
                for terms in combinations([x for x in smooth if x <= n], 4):
                    if sum(terms) == n and is_antichain(terms):
                        answer = list(terms)
                        break
            # Try 5 terms
            if answer is None:
                for terms in combinations([x for x in smooth if x <= n], 5):
                    if sum(terms) == n and is_antichain(terms):
                        answer = list(terms)
                        break
            # Try 6 terms
            if answer is None:
                for terms in combinations([x for x in smooth if x <= n], 6):
                    if sum(terms) == n and is_antichain(terms):
                        answer = list(terms)
                        break
            # Try 7 terms
            if answer is None:
                for terms in combinations([x for x in smooth if x <= n], 7):
                    if sum(terms) == n and is_antichain(terms):
                        answer = list(terms)
                        break
            # Try 8 terms
            if answer is None:
                for terms in combinations([x for x in smooth if x <= n], 8):
                    if sum(terms) == n and is_antichain(terms):
                        answer = list(terms)
                        break
            # Try 9 terms
            if answer is None:
                for terms in combinations([x for x in smooth if x <= n], 9):
                    if sum(terms) == n and is_antichain(terms):
                        answer = list(terms)
                        break
            # Try 10 terms
            if answer is None:
                for terms in combinations([x for x in smooth if x <= n], 10):
                    if sum(terms) == n and is_antichain(terms):
                        answer = list(terms)
                        break
        
        answers.append(answer)
    
    return answers
# EVOLVE-BLOCK-END
