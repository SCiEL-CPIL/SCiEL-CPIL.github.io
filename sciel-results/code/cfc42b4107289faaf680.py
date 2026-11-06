# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    best_set = []
    best_size = 0
    
    def can_add(S, x):
        """
        Check if x can be added to S without violating constraints.
        S is a list of distinct integers. x is not in S.
        Constraints:
        1. x does not divide sum of any 3 distinct elements in S.
        2. For any a in S, a does not divide sum of x and any 2 distinct elements in S \ {a}.
        """
        n = len(S)
        if n < 3:
            # If S has < 3 elements, no 3 distinct others exist for x to divide.
            # For a in S, others in S \ {a} has size < 2 if n < 3?
            # If n=2, S={u,v}. For a=u, others={v}. Can't pick 2.
            # If n=1, S={u}. For a=u, others={}. Can't pick 2.
            # So if n < 3, condition 2 is vacuously true?
            # Wait, condition 2 requires picking 2 from S \ {a}.
            # If n=2, S={u,v}. a=u. S\{a}={v}. Size 1. Can't pick 2.
            # So yes, if n < 3, adding x is always valid regarding existing elements?
            # Let's double check.
            # Condition: "No selected integer may divide the sum of any k=3 DISTINCT OTHER selected integers."
            # If we add x to S (size 2), new size 3.
            # Check x: others are 2 elements. Can't form sum of 3. OK.
            # Check a in S: others are x and the other 1 element in S. Total 2 others. Can't form sum of 3. OK.
            return True
            
        # Check 1: x divides sum of 3 in S
        for combo in combinations(S, 3):
            if sum(combo) % x == 0:
                return False
        
        # Check 2: a in S divides sum of x and 2 others in S
        for a in S:
            # Others for a are S \ {a} plus x.
            # We need to pick 3 from (S \ {a}) U {x}.
            # The triple must include x? No, the triple is from the NEW set.
            # The triple must NOT include a.
            # The triple is formed from (S U {x}) \ {a}.
            # Since x is new, the triples are:
            # 1. {x, u, v} where u, v in S \ {a}, u != v.
            # 2. {u, v, w} where u, v, w in S \ {a}. 
            #    But these triples were already checked in the previous state of S?
            #    Wait, S is the OLD set. The triples {u,v,w} from S\{a} were checked when S was formed?
            #    No, when S was formed, a was in S. The check for a in S (old) ensured a does not divide sum of any 3 in S\{a}.
            #    So we only need to check triples involving x.
            #    Triples involving x and not a: {x, u, v} with u, v in S \ {a}.
            
            # Construct list of candidates for u, v: S without a
            # Instead of list comprehension, we can iterate and skip a.
            for i in range(n):
                u = S[i]
                if u == a:
                    continue
                for j in range(i + 1, n):
                    v = S[j]
                    if v == a:
                        continue
                    # u and v are distinct and not a.
                    # Triple is (x, u, v).
                    if (x + u + v) % a == 0:
                        return False
        return True

    def greedy_construct(order):
        S = []
        for x in order:
            if time.time() > deadline:
                break
            if can_add(S, x):
                S.append(x)
        return S

    # Try different orders
    # 1. Descending (Large numbers are often "safer" divisors? Or summands?)
    # 2. Ascending
    # 3. Random restarts
    
    random.seed(42)
    
    # Fixed orders
    orders = [
        list(range(192, 0, -1)),
        list(range(1, 193))
    ]
    
    # Random restarts
    # We can do many of these now because can_add is O(N^2) ~ 2000 ops.
    # 85s / (192 * 2000 ops) ... 
    # Each greedy pass is 192 iterations. Each iter is ~2000 ops.
    # 192 * 2000 = 384,000 ops per pass.
    # Python does ~10M-50M simple ops/sec? Maybe less. Let's say 1M.
    # 384 passes/sec? 
    # 85s * 384 = 32,000 restarts. This is plenty.
    
    max_restarts = 5000
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        
        perm = list(range(1, 193))
        random.shuffle(perm)
        S = greedy_construct(perm)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        # Optional: Local search? 
        # If we have time, we can try to swap elements to improve size.
        # But greedy with random restarts is usually strong for these.
        
    return best_set
# EVOLVE-BLOCK-END
