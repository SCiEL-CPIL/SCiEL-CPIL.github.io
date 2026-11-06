<<<<<<< SEARCH
def _compute_counts(residues, primes, max_check=2000):
    """Compute coverage counts for x=1..max_check."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    return counts

def _find_L(counts):
    """Find largest L such that all x in 1..L have count >= 2."""
    # Find first x where counts[x] < 2
    below = np.where(counts[1:] < 2)[0]
    if len(below) == 0:
        return len(counts) - 1
    return below[0]

def _score_full(residues, primes, max_check=2000):
    """Full score computation."""
    counts = _compute_counts(residues, primes, max_check)
    return _find_L(counts)

def _update_counts(counts, old_r, new_r, p, max_check=2000):
    """Update counts when changing residue for prime p from old_r to new_r."""
    # Remove old coverage
    old_start = old_r if old_r > 0 else p
    if old_start <= max_check:
        counts[old_start::p] -= 1
    # Add new coverage
    new_start = new_r if new_r > 0 else p
    if new_start <= max_check:
        counts[new_start::p] += 1
=======
def _compute_counts(residues, primes, max_check=2000):
    """Compute coverage counts for x=1..max_check."""
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, primes):
        start = r if r > 0 else p
        if start <= max_check:
            counts[start::p] += 1
    return counts

def _find_L(counts):
    """Find largest L such that all x in 1..L have count >= 2."""
    below = np.where(counts[1:] < 2)[0]
    if len(below) == 0:
        return len(counts) - 1
    return below[0]

def _update_counts(counts, old_r, new_r, p, max_check=2000):
    """Update counts when changing residue for prime p from old_r to new_r."""
    old_start = old_r if old_r > 0 else p
    if old_start <= max_check:
        counts[old_start::p] -= 1
    new_start = new_r if new_r > 0 else p
    if new_start <= max_check:
        counts[new_start::p] += 1

def _evaluate_candidates(counts, p, candidates, max_check=2000):
    """
    Evaluate multiple candidate residues for prime p simultaneously.
    Returns array of scores for each candidate.
    """
    n_cands = len(candidates)
    scores = np.zeros(n_cands, dtype=np.int32)
    
    for idx, new_r in enumerate(candidates):
        # Create a temporary count by adjusting for this candidate
        # We need to compute the effect of changing from current to new_r
        # But we don't know current_r here, so we'll handle it differently
        pass
    
    # Better approach: for each candidate, compute the new counts and find L
    # This is still O(n_cands * max_check) but with numpy it's fast
    results = []
    for new_r in candidates:
        # We need to know the old residue to compute the delta
        # This function is called with the current counts already reflecting the old residue
        # So we need to remove old and add new
        # But we don't have old_r here... 
        # Let's restructure: pass in the current residue too
        pass
    
    return scores
>>>>>>> REPLACE

