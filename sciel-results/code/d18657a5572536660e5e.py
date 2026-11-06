"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit

    def get_sums(selected):
        sums = set()
        for i in range(len(selected)):
            a = selected[i]
            sums.add(2 * a)
            for j in range(i + 1, len(selected)):
                sums.add(a + selected[j])
        return sums

    def is_sidon(selected):
        if not selected:
            return True
        sums = get_sums(selected)
        expected = len(selected) * (len(selected) + 1) // 2
        return len(sums) == expected

    def get_extendable(selected, sums):
        extendable = []
        selected_set = set(selected)
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            new_sums = {2 * x}
            new_sums.update(x + y for y in selected)
            
            if new_sums & sums:
                continue
            
            extendable.append(x)
        return extendable

    def generate_dense_sidon():
        """
        Generates a large, dense Sidon set using a greedy random permutation.
        This set will serve as the 'universe' of candidate vertices for the 
        minimization phase.
        """
        S = []
        S_set = set()
        Sums = set()
        
        # Randomize order to get diverse structures
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            new_sums = {2 * x}
            new_sums.update(x + s for s in S)
            if new_sums & Sums:
                continue
            S.append(x)
            S_set.add(x)
            Sums.update(new_sums)
            
        return S, S_set, Sums

    def minimize_maximal_set(S_initial):
        """
        Given a Sidon set S, we try to remove elements to minimize size 
        while maintaining maximality.
        
        Strategy:
        1. Check if S is maximal. If not, we must add elements (or restart).
           Actually, if we start from a maximal set (constructed by greedy 
           blocking), removing elements might break maximality.
           
        Alternative View:
        Let U be the set of all elements in [1, N] that are "blocked" by S.
        S is maximal iff every x in [1, N] \ S is blocked by S.
        
        We want to find a subset S' of S such that:
        1. S' is Sidon.
        2. Every x in [1, N] \ S' is blocked by S'.
        
        Note: If x is in S, it's trivially "covered" by being in S'.
        If x is not in S, it must be blocked by S'.
        
        This is a Hitting Set problem on the "blocking constraints".
        For each x not in S, let B_x be the set of s in S such that 
        {x} ∪ (S \ {s}) is NOT Sidon? No, that's complex.
        
        Simpler Constraint:
        x is blocked by S if adding x to S creates a sum collision.
        Collision happens if:
        1. 2x is in Sums(S)
        2. x + s is in Sums(S) for some s in S.
        
        If we remove a set R from S, does x remain blocked?
        x is unblocked in S\R if:
        1. 2x is not in Sums(S\R)
        2. x + s is not in Sums(S\R) for all s in S\R.
        
        This is hard to check incrementally for all x.
        
        Let's use a local search approach on the "removal" decision.
        Start with S_max (a maximal Sidon set).
        Iteratively try to remove an element s from S_max.
        If S_max \ {s} is still maximal, keep it removed.
        If not, we must restore it, OR we must find another element to remove 
        that compensates? No, removing elements only makes it *easier* to be 
        non-maximal (fewer blocks). So if S is maximal, S\{s} might not be.
        
        Wait. If S is maximal, every x not in S is blocked by S.
        If we remove s, some x might become unblocked.
        If x becomes unblocked, S\{s} is not maximal.
        
        So we can only remove s if NO x becomes unblocked.
        This is a "greedy removal" strategy.
        
        To improve, we can try to "swap": remove s, and if it breaks maximality 
        for some x, can we add a different element t (not in S) to block x?
        But adding t increases size. We want to minimize size.
        
        Actually, the problem is: Find smallest S such that S is Sidon and Maximal.
        
        Let's try a different structural approach:
        1. Generate a random maximal Sidon set S0.
        2. Identify "redundant" elements. An element s is redundant if 
           S0 \ {s} is still maximal.
        3. Remove all redundant elements (iteratively, as removing one might 
           make others redundant).
        4. This yields a "minimal" maximal set (inclusion-minimal).
        5. Now, try to "shift" elements. Replace s with t (t not in S) if 
           size doesn't increase? No, we want to decrease size.
           
        Let's stick to the removal heuristic, but add a "repair" step if we 
        get stuck or if we want to explore different structures.
        
        Actually, a better strategy for "minimize cardinality":
        1. Start with a random permutation.
        2. Greedily build a Sidon set S.
        3. Ensure maximality by adding any missing element that is blocked? 
           No, if we just build a Sidon set, it might not be maximal.
           To make it maximal, we must add elements until no more can be added.
           This usually increases size.
           
        The key insight: "Maximal" is a constraint. "Minimal size" is the objective.
        
        Let's use Simulated Annealing on the set of elements, with a penalty 
        for non-maximality.
        
        State: A set S.
        Objective: |S| + Penalty * (Number of unblocked elements).
        Penalty must be high enough to enforce maximality.
        
        Moves:
        1. Add an element x (if Sidon property holds).
        2. Remove an element x.
        3. Swap x for y.
        
        We accept moves that improve the objective, or with probability exp(-dE/T).
        
        This allows us to temporarily sacrifice maximality (high penalty) to 
        reduce size, and then "repair" maximality by adding back necessary 
        blockers, potentially finding a smaller configuration where the 
        blockers are more "efficient" (one element blocks many others).
        """
        
        S = list(S_initial)
        S_set = set(S)
        
        def evaluate(S_list):
            """Returns (is_sidon, is_maximal, size)"""
            if not S_list:
                return True, True, 0
            sums = get_sums(S_list)
            is_sid = len(sums) == len(S_list) * (len(S_list) + 1) // 2
            if not is_sid:
                return False, False, len(S_list)
            
            # Check maximality
            s_set = set(S_list)
            unblocked_count = 0
            for x in range(1, N + 1):
                if x in s_set:
                    continue
                new_sums = {2 * x}
                new_sums.update(x + y for y in S_list)
                if not (new_sums & sums):
                    unblocked_count += 1
            
            return True, unblocked_count == 0, len(S_list)

        def cost(S_list):
            """Cost function: Size + Penalty * Unblocked"""
            is_sid, is_max, size = evaluate(S_list)
            if not is_sid:
                return float('inf')
            
            # Calculate unblocked count efficiently? 
            # For SA, we need fast evaluation.
            # Let's just count unblocked.
            sums = get_sums(S_list)
            s_set = set(S_list)
            unblocked = 0
            for x in range(1, N + 1):
                if x in s_set:
                    continue
                new_sums = {2 * x}
                new_sums.update(x + y for y in S_list)
                if not (new_sums & sums):
                    unblocked += 1
            
            penalty = 100 # High penalty to prioritize maximality
            return size + penalty * unblocked

        current_S = S[:]
        current_cost = cost(current_S)
        best_S = current_S[:]
        best_cost = current_cost
        
        T = 10.0
        cooling = 0.999
        
        steps = 0
        while time.time() < deadline - 1.0 and T > 0.1:
            steps += 1
            if steps % 100 == 0:
                T *= cooling
                
            # Generate a neighbor
            move_type = random.choice(['add', 'remove', 'swap'])
            new_S = current_S[:]
            
            if move_type == 'add':
                # Pick a random missing element
                missing = [x for x in range(1, N + 1) if x not in set(current_S)]
                if not missing: continue
                x = random.choice(missing)
                new_S.append(x)
                
            elif move_type == 'remove':
                if not current_S: continue
                x = random.choice(current_S)
                new_S.remove(x)
                
            else: # swap
                if not current_S: continue
                missing = [x for x in range(1, N + 1) if x not in set(current_S)]
                if not missing: continue
                x = random.choice(current_S)
                y = random.choice(missing)
                new_S.remove(x)
                new_S.append(y)
                
            new_cost = cost(new_S)
            
            if new_cost < current_cost:
                current_S = new_S
                current_cost = new_cost
                if current_cost < best_cost:
                    best_S = current_S[:]
                    best_cost = current_cost
            else:
                # Accept with probability
                delta = new_cost - current_cost
                if T > 0:
                    p = math.exp(-delta / T)
                    if random.random() < p:
                        current_S = new_S
                        current_cost = new_cost

        return best_S

    best_set = None
    best_size = N + 1
    
    # Run multiple restarts with different initial dense sets
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline - 2.0:
            break
            
        # Generate a random maximal Sidon set
        # 1. Greedy random Sidon
        S_init, _, _ = generate_dense_sidon()
        
        # 2. Make it maximal
        # Check for unblocked elements and add them?
        # Actually, if we just use the SA, it can handle non-maximal starts.
        # But starting from a valid Sidon set is good.
        
        S_candidate = minimize_maximal_set(S_init)
        
        if S_candidate:
            # Verify
            if is_sidon(S_candidate):
                sums = get_sums(S_candidate)
                if not get_extendable(S_candidate, sums):
                    if len(S_candidate) < best_size:
                        best_size = len(S_candidate)
                        best_set = S_candidate
                        if best_size <= 14:
                            break

    if best_set is None:
        # Fallback: Simple greedy maximal
        S = []
        Sums = set()
        alive = set(range(1, N + 1))
        while alive:
            x = min(alive)
            new_sums_x = {2 * x}
            new_sums_x.update(x + s for s in S)
            if new_sums_x & Sums:
                alive.remove(x)
                continue
            
            S.append(x)
            Sums.update(new_sums_x)
            alive.remove(x)
            
            to_remove = []
            for y in alive:
                if 2 * y in Sums:
                    to_remove.append(y)
                    continue
                blocked = False
                for s in S:
                    if (y + s) in Sums:
                        blocked = True
                        break
                if blocked:
                    to_remove.append(y)
            for y in to_remove:
                alive.remove(y)
        
        best_set = S

    return best_set

# EVOLVE-BLOCK-END
