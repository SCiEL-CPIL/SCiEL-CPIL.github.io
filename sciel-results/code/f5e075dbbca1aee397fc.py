import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # 1. Generate primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    num_primes = len(primes)
    
    # Precompute residues for each prime: residues[p][r] = list of n in 1..1000
    # We only care about coverage of 1..1000 for the score, but the problem says "largest L".
    # The evaluator caps at 10^5 but the task says "Covering 1..L by prime residue classes... primes up to 1000".
    # Usually, these covering systems are designed to cover a specific range. 
    # The score is "largest L such that every integer 1..L lies in some class".
    # If we cover 1..1000, L is at least 1000.
    # However, the classes are periodic. If we cover 1..1000, we might cover more.
    # But checking up to 10^5 is expensive.
    # Let's assume the primary goal is to cover 1..1000 effectively, as L>=999 is the target.
    # We will optimize for coverage of 1..1000.
    
    # We represent the state as:
    # - A set of active primes
    # - For each active prime, a chosen residue r
    # 
    # Objective: Maximize the smallest uncovered integer in 1..1000 (i.e., maximize L).
    # Constraint: Sum(1/p) <= 1.
    
    # Initial State:
    # Start with a greedy solution to have a good baseline.
    # Greedy: Pick primes that cover the most *new* numbers, subject to budget.
    
    # To make SA fast, we need to evaluate the "score" (L) quickly.
    # Calculating L requires checking 1..1000.
    # We can maintain a bitset or an array of booleans for 1..1000.
    
    # Let's use a numpy array for fast coverage checks? 
    # Or just a list of booleans.
    # With 1000 elements, updating is fast.
    
    # State:
    # active_primes: list of primes currently used
    # residues: dict {prime: residue}
    
    # Budget check:
    # We need to track cost.
    
    # Let's define a helper to calculate the current L from a coverage array.
    def get_L(covered):
        L = 0
        for i in range(1, 1001):
            if covered[i]:
                L = i
            else:
                break
        return L

    # Helper to update coverage array when changing a prime's residue or adding/removing prime.
    # This is tricky to do incrementally if we just store the final `covered` array,
    # because we need to know if a number is covered by *other* classes.
    # Better representation:
    # For each number n in 1..1000, store the count of classes covering it.
    # count[n] = number of (p, r) such that n % p == r.
    # A number is covered if count[n] > 0.
    # When we change residue of p from r_old to r_new:
    #   For all n in 1..1000 where n % p == r_old: count[n] -= 1
    #   For all n in 1..1000 where n % p == r_new: count[n] += 1
    # When we add prime p with residue r:
    #   For all n in 1..1000 where n % p == r: count[n] += 1
    # When we remove prime p with residue r:
    #   For all n in 1..1000 where n % p == r: count[n] -= 1
    
    # Precompute for each prime p, the list of indices for each residue.
    # residues_idx[p][r] = [n1, n2, ...]
    residues_idx = {}
    for p in primes:
        res_list = [[] for _ in range(p)]
        for n in range(1, 1001):
            res_list[n % p].append(n)
        residues_idx[p] = res_list

    # Initialize Greedy Solution
    # We want to maximize L.
    # Greedy Set Cover heuristic:
    # 1. Start with empty set.
    # 2. At each step, find (p, r) that maximizes new coverage, subject to budget.
    # 3. Add it.
    
    used_primes = set()
    residues = {}
    counts = [0] * 1001 # counts[n] for n in 1..1000
    current_cost = Fraction(0)
    
    # To speed up greedy, we can just pick random residues for a set of primes?
    # No, greedy is better for init.
    
    # Simple Greedy:
    # We have budget 1.
    # Small primes are expensive.
    # Let's try to cover 1..1000.
    
    # Actually, a known good covering system for 1..1000 with budget 1 might use specific primes.
    # Let's run a quick greedy to get a starting point.
    
    def greedy_init():
        u_p = set()
        res = {}
        c = [0] * 1001
        cost = Fraction(0)
        
        # Try adding primes one by one.
        # Which primes to try?
        # Maybe all primes, but pick the best residue.
        
        # We can't add ALL primes because sum(1/p) > 1.
        # Sum(1/p) for p<=1000 is ~ 1.9.
        # We need to select a subset.
        
        # Heuristic: 
        # Iterate through primes. For each prime, if we can afford it, 
        # pick the residue that covers the most currently uncovered numbers.
        # If it covers 0 new numbers, maybe skip it? Or include it if it helps later?
        # In set cover, we usually only add if it helps.
        
        # But SA can swap residues later.
        # Let's just pick a random subset of primes that fits the budget, 
        # and random residues, then let SA fix it.
        # Or better: Greedy selection of primes.
        
        # Let's sort primes by 1/p (descending) i.e. p ascending.
        # Small primes are "dense" but expensive.
        
        # Let's try a randomized greedy:
        # Shuffle primes.
        # For each prime, if cost + 1/p <= 1:
        #    Pick residue r that maximizes new coverage.
        #    If new coverage > 0, add it.
        
        random.shuffle(primes)
        for p in primes:
            if cost + Fraction(1, p) > 1:
                continue
            
            best_r = -1
            best_gain = 0
            for r in range(p):
                gain = 0
                for n in residues_idx[p][r]:
                    if c[n] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r != -1 and best_gain > 0:
                u_p.add(p)
                res[p] = best_r
                cost += Fraction(1, p)
                for n in residues_idx[p][best_r]:
                    c[n] += 1
                    
        return u_p, res, c, cost

    used_primes, residues, counts, current_cost = greedy_init()
    
    # Calculate initial L
    def calc_L(counts):
        L = 0
        for i in range(1, 1001):
            if counts[i] > 0:
                L = i
            else:
                break
        return L
        
    best_L = calc_L(counts)
    current_L = best_L
    
    best_state = {
        'primes': set(used_primes),
        'residues': dict(residues),
        'cost': current_cost
    }
    
    # Simulated Annealing
    # Temperature schedule
    T = 1.0
    cooling = 0.99
    
    # Moves:
    # 1. Change residue of an existing prime.
    # 2. Swap a prime: remove p1, add p2 (if budget allows).
    # 3. Add a prime (if budget allows).
    # 4. Remove a prime (to free budget for others? or just simplify).
    
    # We need to be careful with time.
    # Each move requires updating counts and recalculating L.
    # Recalculating L is O(1000).
    # Updating counts is O(1000/p) * 2 (for old and new residue).
    
    # To speed up L calculation, we can maintain a "gaps" structure?
    # Or just accept O(1000) per move.
    # 1000 moves/sec * 100s = 100k moves.
    # O(1000) per move -> 100M ops. Python might be slow.
    # Let's optimize L calculation.
    
    # Instead of scanning 1..1000, we can maintain a set of uncovered numbers?
    # Uncovered numbers are those with count[n] == 0.
    # If we maintain a set `uncovered`, we can find min(uncovered) quickly?
    # `min(uncovered)` is O(size). If size is small, it's fast.
    # Initially, many are uncovered. As we cover more, it gets faster.
    # But updating the set on every count change is O(1) per element changed.
    
    # Let's use a set of uncovered indices.
    uncovered = set(i for i in range(1, 1001) if counts[i] == 0)
    
    def get_L_from_uncovered(uncovered_set):
        if not uncovered_set:
            return 1000
        return min(uncovered_set) - 1
        
    # Verify initial L
    current_L = get_L_from_uncovered(uncovered)
    best_L = current_L
    
    # SA Loop
    while time.time() < deadline:
        if T < 0.01:
            break
            
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.7:
            # Change residue of an existing prime
            if not used_primes:
                continue
            p = random.choice(list(used_primes))
            old_r = residues[p]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            
            # Apply move
            # Decrement old_r
            for n in residues_idx[p][old_r]:
                if counts[n] == 1:
                    uncovered.add(n)
                counts[n] -= 1
            
            # Increment new_r
            for n in residues_idx[p][new_r]:
                if counts[n] == 0:
                    uncovered.discard(n)
                counts[n] += 1
            
            residues[p] = new_r
            
            new_L = get_L_from_uncovered(uncovered)
            delta_L = new_L - current_L
            
            # Accept if better, or with probability exp(delta/T)
            if delta_L >= 0 or random.random() < pow(2.718, delta_L / T):
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_state = {
                        'primes': set(used_primes),
                        'residues': dict(residues),
                        'cost': current_cost
                    }
            else:
                # Revert
                for n in residues_idx[p][new_r]:
                    if counts[n] == 1:
                        uncovered.add(n)
                    counts[n] -= 1
                    
                for n in residues_idx[p][old_r]:
                    if counts[n] == 0:
                        uncovered.discard(n)
                    counts[n] += 1
                    
                residues[p] = old_r
                
        elif move_type < 0.9:
            # Swap prime: remove p1, add p2
            if len(used_primes) < 2:
                continue
            p1 = random.choice(list(used_primes))
            p2 = random.choice(primes)
            if p1 == p2:
                continue
            
            # Check if we can swap
            # Cost change: -1/p1 + 1/p2
            cost_change = Fraction(1, p2) - Fraction(1, p1)
            if current_cost + cost_change > 1:
                continue
                
            old_r1 = residues[p1]
            new_r2 = random.randint(0, p2 - 1)
            
            # Apply: Remove p1, Add p2
            for n in residues_idx[p1][old_r1]:
                if counts[n] == 1:
                    uncovered.add(n)
                counts[n] -= 1
                
            for n in residues_idx[p2][new_r2]:
                if counts[n] == 0:
                    uncovered.discard(n)
                counts[n] += 1
                
            used_primes.remove(p1)
            used_primes.add(p2)
            del residues[p1]
            residues[p2] = new_r2
            current_cost += cost_change
            
            new_L = get_L_from_uncovered(uncovered)
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < pow(2.718, delta_L / T):
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_state = {
                        'primes': set(used_primes),
                        'residues': dict(residues),
                        'cost': current_cost
                    }
            else:
                # Revert
                for n in residues_idx[p2][new_r2]:
                    if counts[n] == 1:
                        uncovered.add(n)
                    counts[n] -= 1
                    
                for n in residues_idx[p1][old_r1]:
                    if counts[n] == 0:
                        uncovered.discard(n)
                    counts[n] += 1
                    
                used_primes.add(p1)
                used_primes.remove(p2)
                residues[p1] = old_r1
                del residues[p2]
                current_cost -= cost_change
                
        else:
            # Add a prime (if budget allows)
            # Find a prime not in used_primes
            available = [p for p in primes if p not in used_primes]
            if not available:
                continue
            p = random.choice(available)
            if current_cost + Fraction(1, p) > 1:
                continue
                
            new_r = random.randint(0, p - 1)
            
            for n in residues_idx[p][new_r]:
                if counts[n] == 0:
                    uncovered.discard(n)
                counts[n] += 1
                
            used_primes.add(p)
            residues[p] = new_r
            current_cost += Fraction(1, p)
            
            new_L = get_L_from_uncovered(uncovered)
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < pow(2.718, delta_L / T):
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_state = {
                        'primes': set(used_primes),
                        'residues': dict(residues),
                        'cost': current_cost
                    }
            else:
                # Revert
                for n in residues_idx[p][new_r]:
                    if counts[n] == 1:
                        uncovered.add(n)
                    counts[n] -= 1
                    
                used_primes.remove(p)
                del residues[p]
                current_cost -= Fraction(1, p)
                
        T *= cooling
        
    # Construct final result from best_state
    final_classes = []
    for p, r in best_state['residues'].items():
        final_classes.append([p, r])
        
    return final_classes

# EVOLVE-BLOCK-END
