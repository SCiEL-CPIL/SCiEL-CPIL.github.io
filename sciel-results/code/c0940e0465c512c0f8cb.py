import numpy as np
import time
import itertools
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    # Precompute all t-subsets and their orbits under Z19 (cyclic group of order 19)
    # Since 19 is prime, the cyclic group Z19 acts transitively on points
    # A t-subset's orbit size is 19/gcd(19, stabilizer_size), but since 19 is prime,
    # orbit size is either 1 (if subset is invariant, impossible for 1<t<19) or 19
    
    # For t=4, k=7, v=19: all t-subsets have orbit size 19
    # So there are C(19,4)/19 = 3060/19 = 161 orbits of t-subsets
    
    # Precompute t-subset indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Compute orbit representatives for t-subsets under Z19
    # Orbit of a t-subset S: {(S + i) mod 19 : i in 0..18}
    orbit_reps = []
    orbit_to_subsets = defaultdict(list)
    seen_orbits = set()
    
    for subset in all_t_subsets:
        # Canonical representative: lexicographically smallest in orbit
        orbit = set()
        for shift in range(v):
            shifted = tuple(sorted((x + shift) % v for x in subset))
            orbit.add(shifted)
        rep = min(orbit)
        if rep not in seen_orbits:
            seen_orbits.add(rep)
            orbit_reps.append(rep)
            orbit_to_subsets[rep] = list(orbit)
    
    # For each orbit representative, store all t-subsets in that orbit
    orbit_rep_list = orbit_reps
    orbit_to_indices = {}
    for rep, subsets in orbit_to_subsets.items():
        indices = [subset_to_idx[s] for s in subsets]
        orbit_to_indices[rep] = indices
    
    # Precompute all k-subsets and group by orbit
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Group k-subsets by orbit
    k_orbit_reps = []
    k_orbit_to_subsets = defaultdict(list)
    seen_k_orbits = set()
    
    for subset in all_k_subsets:
        orbit = set()
        for shift in range(v):
            shifted = tuple(sorted((x + shift) % v for x in subset))
            orbit.add(shifted)
        rep = min(orbit)
        if rep not in seen_k_orbits:
            seen_k_orbits.add(rep)
            k_orbit_reps.append(rep)
            k_orbit_to_subsets[rep] = list(orbit)
    
    # For each k-orbit, compute which t-orbit representatives it covers
    # A k-subset covers a t-subset if t-subset ⊆ k-subset
    # Two k-subsets in the same orbit cover the same set of t-orbit reps (up to shift)
    
    k_orbit_covered_t_orbits = {}
    k_orbit_index = {}
    for i, rep in enumerate(k_orbit_reps):
        k_orbit_index[rep] = i
        # Take one representative k-subset
        rep_set = set(rep)
        covered = set()
        for t_rep in orbit_rep_list:
            # Check if any t-subset in t_rep's orbit is contained in rep_set
            # By symmetry, check if t_rep itself (or some shift) is contained
            # Actually, we need to check: does there exist a shift s such that
            # (t_rep + s) mod v ⊆ (rep + s') mod v for some s'?
            # This is equivalent to: is t_rep in the orbit of some t-subset of rep?
            # Simpler: just check all t-subsets of rep and see which orbits they belong to
            pass
        
        # Alternative: for the representative k-subset, find all t-subsets contained in it
        covered_orbits = set()
        for t_sub in itertools.combinations(rep, t):
            t_tup = tuple(t_sub)
            # Find which orbit this t-subset belongs to
            t_orbit = set()
            for shift in range(v):
                shifted = tuple(sorted((x + shift) % v for x in t_tup))
                t_orbit.add(shifted)
            t_rep = min(t_orbit)
            covered_orbits.add(t_rep)
        
        k_orbit_covered_t_orbits[i] = covered_orbits
    
    # Now: we need to select k-orbits (and how many copies from each) to cover all t-orbits
    # This is a set cover problem at the orbit level!
    
    num_k_orbits = len(k_orbit_reps)
    num_t_orbits = len(orbit_rep_list)
    
    # Incidence matrix: k_orbit i covers t_orbit j?
    k_orbit_to_t_orbit = np.zeros((num_k_orbits, num_t_orbits), dtype=bool)
    for i in range(num_k_orbits):
        covered = k_orbit_covered_t_orbits[i]
        for t_rep in covered:
            # Find index of t_rep in orbit_rep_list
            if t_rep in [r for r in orbit_rep_list]:
                j = orbit_rep_list.index(t_rep)
                k_orbit_to_t_orbit[i, j] = True
    
    # Greedy set cover at orbit level: select k-orbits to cover all t-orbits
    uncovered_t_orbits = set(range(num_t_orbits))
    selected_k_orbits = []
    
    while uncovered_t_orbits:
        if time.time() > deadline - 20.0:
            break
        
        best_gain = -1
        best_k_orbit = -1
        
        for i in range(num_k_orbits):
            if i in selected_k_orbits:
                continue
            gain = sum(1 for j in uncovered_t_orbits if k_orbit_to_t_orbit[i, j])
            if gain > best_gain:
                best_gain = gain
                best_k_orbit = i
        
        if best_k_orbit == -1 or best_gain == 0:
            break
        
        selected_k_orbits.append(best_k_orbit)
        uncovered_t_orbits.discard(j for j in uncovered_t_orbits if k_orbit_to_t_orbit[best_k_orbit, j])
    
    # Now instantiate: for each selected k-orbit, choose one representative and all its shifts
    # But we may need multiple copies from some orbits to cover everything
    
    # Build the initial solution: for each selected k-orbit, add all 19 shifts
    # This gives 19 * len(selected_k_orbits) blocks
    
    initial_blocks = []
    for k_orbit_idx in selected_k_orbits:
        rep = k_orbit_reps[k_orbit_idx]
        for shift in range(v):
            block = tuple(sorted((x + shift) % v for x in rep))
            initial_blocks.append(list(block))
    
    # Remove duplicates
    seen = set()
    unique_blocks = []
    for b in initial_blocks:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            unique_blocks.append(b)
    
    # Now use vectorized greedy to repair and optimize
    # Compute coverage
    all_t_subsets_list = list(all_t_subsets)
    t_sub_idx = {s: i for i, s in enumerate(all_t_subsets_list)}
    
    # For each block, precompute which t-subsets it covers
    def block_to_t_indices(block):
        return [t_sub_idx[tuple(s)] for s in itertools.combinations(block, t)]
    
    # Build coverage
    coverage = np.zeros(num_t_subsets, dtype=np.int32)
    block_t_indices = {}
    for b in unique_blocks:
        t_idx = block_to_t_indices(b)
        block_t_indices[tuple(b)] = t_idx
        coverage[t_idx] += 1
    
    # Check if we have a valid covering
    if np.all(coverage > 0):
        # Try to remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(unique_blocks)):
                if time.time() > deadline - 15.0:
                    return unique_blocks
                b = unique_blocks[i]
                t_idx = block_t_indices[tuple(b)]
                redundant = all(coverage[j] > 1 for j in t_idx)
                if redundant:
                    coverage[t_idx] -= 1
                    unique_blocks.pop(i)
                    del block_t_indices[tuple(b)]
                    changed = True
                    break
    
    # Now do local search: try to replace blocks to reduce count
    # Use ILS: perturb by removing random blocks, then repair
    
    best_blocks = unique_blocks[:]
    best_count = len(unique_blocks)
    
    # Precompute all k-subsets as indices for faster access
    k_sub_to_idx = {s: i for i, s in enumerate(all_k_subsets)}
    k_sub_t_indices = [block_to_t_indices(list(s)) for s in all_k_subsets]
    
    # ILS
    current_blocks = best_blocks[:]
    current_count = best_count
    
    for iteration in range(300):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb: remove 3-8 random blocks
        perturb_size = random.randint(3, min(8, len(current_blocks) - 1))
        if len(current_blocks) <= perturb_size + 1:
            perturb_size = max(1, len(current_blocks) // 3)
        
        to_remove = random.sample(range(len(current_blocks)), perturb_size)
        to_remove_set = set(to_remove)
        
        perturbed = [b for i, b in enumerate(current_blocks) if i not in to_remove_set]
        perturbed_t = {tuple(b): block_t_indices.get(tuple(b), block_to_t_indices(b)) for b in perturbed}
        
        # Recompute coverage for perturbed
        cov = np.zeros(num_t_subsets, dtype=np.int32)
        for b in perturbed:
            t_idx = perturbed_t[tuple(b)]
            cov[t_idx] += 1
        
        # Repair: greedily add blocks to cover uncovered t-subsets
        uncovered = np.where(cov == 0)[0]
        
        while len(uncovered) > 0:
            if time.time() > deadline - 10.0:
                break
            
            best_gain = -1
            best_block = None
            
            for ks_idx in range(num_k_subsets):
                t_idx = k_sub_t_indices[ks_idx]
                gain = sum(1 for j in uncovered if cov[j] == 0 and (t_idx.count(j) > 0))
                # Faster: use set intersection
                if gain > best_gain:
                    best_gain = gain
                    best_block = ks_idx
            
            if best_block is None or best_gain == 0:
                break
            
            new_block = list(all_k_subsets[best_block])
            if new_block not in perturbed:
                perturbed.append(new_block)
                t_idx = k_sub_t_indices[best_block]
                cov[t_idx] += 1
                uncovered = np.where(cov == 0)[0]
        
        if len(uncovered) == 0 and len(perturbed) < best_count:
            best_blocks = perturbed[:]
            best_count = len(perturbed)
            current_blocks = perturbed[:]
        
        elif len(uncovered) == 0:
            current_blocks = perturbed[:]
    
    # Final cleanup: remove redundant blocks
    if len(best_blocks) > 0:
        cov = np.zeros(num_t_subsets, dtype=np.int32)
        bti = {}
        for b in best_blocks:
            t_idx = [t_sub_idx[tuple(s)] for s in itertools.combinations(b, t)]
            bti[tuple(b)] = t_idx
            cov[t_idx] += 1
        
        changed = True
        while changed:
            changed = False
            for i in range(len(best_blocks)):
                b = best_blocks[i]
                t_idx = bti[tuple(b)]
                if all(cov[j] > 1 for j in t_idx):
                    cov[t_idx] -= 1
                    best_blocks.pop(i)
                    del bti[tuple(b)]
                    changed = True
                    break
    
    return best_blocks

