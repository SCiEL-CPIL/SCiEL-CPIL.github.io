import random
import time
from collections import defaultdict

def construct():
    MAX_VAL = 128
    MIN_VAL = 2
    CANDIDATES = list(range(MIN_VAL, MAX_VAL + 1))
    
    def get_collision_info(seq):
        """
        Returns a list of (product, [list of (i, j) indices]) for all products that appear more than once.
        Also returns the total count of collisions (excess items).
        """
        if not seq:
            return [], 0
        prods = defaultdict(list)
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                prods[p].append((i, j))
        
        collisions = []
        collision_count = 0
        for p, indices in prods.items():
            if len(indices) > 1:
                collisions.append((p, indices))
                collision_count += len(indices) - 1
        return collisions, collision_count

    def is_valid(seq):
        """Check if sequence has no duplicate consecutive block products."""
        if not seq:
            return True
        prods = set()
        n = len(seq)
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def greedy_init(shuffle=False):
        """
        Greedily build a sequence.
        If shuffle is True, try multiple random orders.
        """
        best_seq = []
        if not shuffle:
            for x in CANDIDATES:
                new_seq = sorted(best_seq + [x])
                if is_valid(new_seq):
                    best_seq = new_seq
            return best_seq
        
        # Try a few random shuffles to find a better greedy path
        for _ in range(20):
            shuffled = CANDIDATES[:]
            random.shuffle(shuffled)
            seq = []
            for x in shuffled:
                new_seq = sorted(seq + [x])
                if is_valid(new_seq):
                    seq = new_seq
            if len(seq) > len(best_seq):
                best_seq = seq
        return best_seq

    # Start with a good greedy solution
    current_seq = greedy_init(shuffle=True)
    best_seq = current_seq[:]
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s buffer
    
    # Local Search: Targeted Collision Resolution
    while time.time() - start_time < time_limit:
        collisions, collision_count = get_collision_info(current_seq)
        
        # If valid, try to extend
        if collision_count == 0:
            # Try to add a new element
            unused = [x for x in CANDIDATES if x not in current_seq]
            if not unused:
                break
            
            # Try adding the smallest unused first, or random
            # Greedy extension
            added = False
            for x in sorted(unused):
                new_seq = sorted(current_seq + [x])
                if is_valid(new_seq):
                    current_seq = new_seq
                    if len(current_seq) > len(best_seq):
                        best_seq = current_seq[:]
                    added = True
                    break
            
            if not added:
                # Try random swaps to find a better structure that allows extension
                # Or just accept the current state as a local optimum and try to break it
                # Let's try a random swap to see if we can find a valid state with a different composition
                if current_seq:
                    x = random.choice(current_seq)
                    y = random.choice(unused)
                    new_seq = sorted([v for v in current_seq if v != x] + [y])
                    if is_valid(new_seq):
                        current_seq = new_seq
                        if len(current_seq) > len(best_seq):
                            best_seq = current_seq[:]
                else:
                    break
        else:
            # Resolve collisions
            # Pick a random collision
            _, indices = random.choice(collisions)
            # Pick two indices from the collision that are distinct subarrays
            # We want to modify the sequence to break the equality.
            # Strategy: Swap one of the elements involved in these subarrays with an unused element.
            
            # Collect all elements that appear in the colliding subarrays
            involved_elements = set()
            for i, j in indices:
                for k in range(i, j+1):
                    involved_elements.add(current_seq[k])
            
            unused = [x for x in CANDIDATES if x not in current_seq]
            
            improved = False
            # Try swapping each involved element with each unused element
            # Limit the number of attempts to avoid O(N^2) blowup in tight loops
            for elem in list(involved_elements):
                for new_elem in unused:
                    if elem == new_elem:
                        continue
                    new_seq = sorted([v for v in current_seq if v != elem] + [new_elem])
                    new_collisions, new_count = get_collision_info(new_seq)
                    
                    # Accept if collisions decrease
                    if new_count < collision_count:
                        current_seq = new_seq
                        collision_count = new_count
                        improved = True
                        break
                    # Accept if same collisions but maybe better structure? 
                    # For now, strict improvement to ensure convergence, 
                    # but we might get stuck. 
                    # Let's allow sideways moves with low probability if stuck?
                    # For simplicity and speed, let's stick to strict improvement first.
                    # If we get stuck, we can restart or randomize.
                
                if improved:
                    break
            
            if not improved:
                # Stuck in a local minimum with collisions.
                # Try a random move: remove a random element from a colliding subarray
                # and replace with a random unused element, even if it doesn't immediately reduce collisions,
                # but we need to be careful not to loop forever.
                # Let's try to remove one of the involved elements and see if we can fill it with something else.
                if unused and current_seq:
                    elem = random.choice(list(involved_elements))
                    new_elem = random.choice(unused)
                    new_seq = sorted([v for v in current_seq if v != elem] + [new_elem])
                    new_collisions, new_count = get_collision_info(new_seq)
                    # Accept if it doesn't get much worse (e.g. <= collision_count + 1)
                    if new_count <= collision_count + 1:
                        current_seq = new_seq
                        collision_count = new_count
                else:
                    # No unused elements or no seq, break
                    break

    return best_seq
