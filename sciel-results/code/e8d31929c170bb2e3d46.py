import random
import time
from itertools import combinations_with_replacement

def construct():
        start_time = time.time()
        time_limit = 55.0
        best_set = []
        best_size = 0
        
        def run_greedy(order):
            S = []
            pair_sums = set()
            triple_sums = set()
            
            for x in order:
                # Check pair sums
                # new_pairs = {2*x} | {x+y for y in S}
                # If any in pair_sums, reject
                # We can check 2*x first
                if 2*x in pair_sums:
                    continue
                # Check x+y
                valid = True
                new_pairs = {2*x}
                for y in S:
                    s = x + y
                    if s in pair_sums:
                        valid = False
                        break
                    new_pairs.add(s)
                
                if not valid:
                    continue

                # Check triple sums
                # new_triples = {3*x} | {2*x+y for y in S} | {x+y+z for y,z in S}
                if 3*x in triple_sums:
                    continue
                
                valid = True
                new_triples = {3*x}
                
                # Check 2*x+y
                for y in S:
                    s = 2*x + y
                    if s in triple_sums:
                        valid = False
                        break
                    new_triples.add(s)
                
                if not valid:
                    continue
                
                # Check x+y+z
                # We need to generate all y+z sums from S
                # and check if x + (y+z) is in triple_sums
                # Also check for duplicates within new_triples
                # Duplicates:
                # 3x vs 2x+y -> x=y (no)
                # 3x vs x+y+z -> 2x=y+z (checked: 2x not in pair_sums? 
                #   Wait. pair_sums contains y+z. If 2x is in pair_sums, we rejected earlier.
                #   So 2x != y+z. So 3x != x+y+z. Safe.)
                # 2x+y vs 2x+y' -> y=y' (no)
                # 2x+y vs x+y'+z' -> x+y = y'+z'.
                #   This means x+y is a pair sum from S.
                #   We checked if x+y is in pair_sums.
                #   pair_sums contains y'+z'.
                #   So if x+y in pair_sums, we rejected earlier.
                #   So 2x+y != x+y'+z'. Safe.
                # x+y+z vs x+y'+z' -> y+z = y'+z'.
                #   Pair sums in S are unique. So y,z is same pair.
                #   So no duplicates.
                
                # So we just need to check against triple_sums.
                for y_idx in range(len(S)):
                    y = S[y_idx]
                    for z_idx in range(y_idx, len(S)):
                        z = S[z_idx]
                        s = x + y + z
                        if s in triple_sums:
                            valid = False
                            break
                        new_triples.add(s)
                    if not valid:
                        break
                
                if not valid:
                    continue

                # Accept
                S.append(x)
                pair_sums.update(new_pairs)
                triple_sums.update(new_triples)
            
            return S

        # Initial runs
        current_set = run_greedy(range(1, 4097))
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

        current_set = run_greedy(range(4096, 0, -1))
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set

        # Randomized runs
        while time.time() - start_time < time_limit:
            candidates = list(range(1, 4097))
            random.shuffle(candidates)
            current_set = run_greedy(candidates)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set
                
        return best_set
