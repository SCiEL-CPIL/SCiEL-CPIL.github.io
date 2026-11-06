<<<<<<< SEARCH
def _greedy_construct(primes, n_max=N):
    """
    Construct a large admissible set by greedy inclusion.
    Start with all integers in [0, n_max], then remove elements to fix
    each prime whose residue classes are all covered.
    Uses a heuristic: remove the class with the minimum count, with random tie-breaking.
    """
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    for p in small_primes:
        if p > len(points):
            break
        class_counts = Counter(v % p for v in points)
        if len(class_counts) == p:
            min_count = min(class_counts.values())
            min_classes = [r for r, c in class_counts.items() if c == min_count]
            # Randomly choose among min-count classes to diversify
            remove_class = random.choice(min_classes)
            points = {v for v in points if v % p != remove_class}
    
    return sorted(points)
=======
def _greedy_construct(primes, n_max=N):
    """
    Construct a large admissible set by greedy inclusion.
    Start with all integers in [0, n_max], then remove elements to fix
    each prime whose residue classes are all covered.
    Uses a conflict-aware heuristic: for each prime that needs a class removed,
    choose the class whose removal causes the least damage to other primes'
    admissibility (i.e., minimizes how many other primes become 'tight').
    """
    points = set(range(n_max + 1))
    small_primes = [p for p in primes if p <= n_max + 1]
    
    # Precompute residue class membership for speed
    # For each prime p, map residue r -> list of points with v % p == r
    # We'll rebuild this as points change, but cache when possible
    
    for p in small_primes:
        if p > len(points):
            break
        class_counts = Counter(v % p for v in points)
        if len(class_counts) < p:
            continue  # Already admissible for this prime
        
        # All p classes present. Choose which class to remove.
        # Heuristic: for each candidate class r, compute the "damage" to other primes.
        # Damage = number of other primes q where removing class r mod p would
        # make q lose its only "safety margin" (i.e., q would become tight).
        
        # Build class membership for current p
        class_members = {r: [] for r in range(p)}
        for v in points:
            class_members[v % p].append(v)
        
        best_r = None
        best_score = float('inf')
        
        for r in range(p):
            members = class_members[r]
            if not members:
                continue
            
            # Compute damage: for each other prime q, check if removing these members
            # would make q tight (all q residues present with no margin)
            damage = 0
            for q in small_primes:
                if q == p or q > len(points):
                    continue
                # Check if q is currently tight (all q residues present)
                q_classes = Counter(v % q for v in points)
                if len(q_classes) < q:
                    continue  # q not tight, no damage possible
                
                # For q to remain non-tight after removing members, at least one
                # residue class mod q must be entirely contained in members.
                # Check: is there a residue s mod q such that ALL points with v%q==s
                # are in members?
                member_set = set(members)
                saved = False
                for s in range(q):
                    # Check if all points with v%q==s are in member_set
                    all_in = True
                    for v in points:
                        if v % q == s and v not in member_set:
                            all_in = False
                            break
                    if all_in:
                        saved = True
                        break
                if not saved:
                    damage += 1
            
            # We want to minimize damage (remove class that hurts other primes least)
            # Tie-break by smaller class size (remove fewer elements)
            score = (damage, len(members))
            if best_r is None or score < best_score:
                best_score = score
                best_r = r
        
        if best_r is not None:
            points = {v for v in points if v % p != best_r}
    
    return sorted(points)
>>>>>>> REPLACE

