import time
import random

def construct():
        start_time = time.time()
        deadline = 55.0
        
        # Generate 7-smooth numbers <= 10000
        primes = [2, 3, 5, 7]
        pool = set()
        def gen(val, idx):
            if idx == len(primes):
                if 1 <= val <= 10000:
                    pool.add(val)
                return
            p = primes[idx]
            while val <= 10000:
                gen(val, idx+1)
                val *= p
        
        gen(1, 0)
        P = list(pool)
        P.sort()
        
        # Initial A: Powers of 2
        A = [2**i for i in range(20)]
        A_set = set(A)
        
        def calc_score(A_list):
            s = set()
            p = set()
            n = len(A_list)
            for i in range(n):
                ai = A_list[i]
                for j in range(i, n):
                    aj = A_list[j]
                    s.add(ai + aj)
                    p.add(ai * aj)
            return max(len(s), len(p))

        current_score = calc_score(A)
        
        # Local Search
        no_improve_count = 0
        while (time.time() - start_time) < deadline:
            improved = False
            # Try all swaps
            for i in range(20):
                a_old = A[i]
                # Randomize order of P? Or just iterate.
                # To avoid bias, maybe shuffle P copy?
                # Shuffling 1000 items is fast.
                P_copy = P[:]
                random.shuffle(P_copy)
                
                for b_new in P_copy:
                    if b_new in A_set:
                        continue
                    
                    # Swap
                    A[i] = b_new
                    new_score = calc_score(A)
                    
                    if new_score < current_score:
                        current_score = new_score
                        A_set.remove(a_old)
                        A_set.add(b_new)
                        improved = True
                        # Accept and break? Or continue to find better swap for this i?
                        # Steepest descent for this i:
                        # Let's continue checking to find the BEST swap for this i.
                        # But that requires keeping track of best.
                        # First improvement is faster. Let's try first improvement first.
                        break 
                    else:
                        A[i] = a_old
                
                if improved:
                    break # Restart pass? Or continue?
                    # If we break, we do a full pass again.
                    # This is "First Improvement" with restart.
            
            if not improved:
                # Stuck. Perturb.
                # Swap 2 random elements with random pool elements
                if (time.time() - start_time) > deadline - 5:
                    break
                
                idx1, idx2 = random.sample(range(20), 2)
                a1, a2 = A[idx1], A[idx2]
                
                # Pick random b1, b2 from P
                b1 = random.choice(P)
                b2 = random.choice(P)
                # Ensure distinctness and not in A (or handle carefully)
                # Simple perturbation: just swap
                A[idx1] = b1
                A[idx2] = b2
                
                # Fix duplicates if b1==b2 or in A
                # This is messy. Better:
                # Remove a1, a2 from A_set. Add b1, b2 if not present.
                # If b1 in A_set (and b1 != a1, b1 != a2), we have a duplicate.
                # Let's just use a safer perturbation:
                # Pick b1 not in A. Swap a1 with b1.
                # Pick b2 not in A (after first swap). Swap a2 with b2.
                
                A_set.remove(a1)
                b1 = random.choice([x for x in P if x not in A_set])
                A[idx1] = b1
                A_set.add(b1)
                
                A_set.remove(a2)
                b2 = random.choice([x for x in P if x not in A_set])
                A[idx2] = b2
                A_set.add(b2)
                
                current_score = calc_score(A)
                no_improve_count += 1
                if no_improve_count > 10:
                    break
        
        return A
