<<<<<<< SEARCH
def _beam_grow(sequence, dimension, beam_width, rng):
    """Beam search chain growth: keep top-k partial chains by contact count."""
    start = (0,) * dimension
    # Each beam entry: (chain, occupied_dict, energy)
    beams = [([start], {start: 0}, 0)]
    
    for i in range(1, len(sequence)):
        candidates = []
        for chain, occupied, e in beams:
            last = chain[-1]
            for site in _neighbours(last):
                if site in occupied:
                    continue
                # Compute new contacts if we place residue i at site
                gain = 0
                if sequence[i] == "H":
                    for q in _neighbours(site):
                        j = occupied.get(q)
                        if j is not None and sequence[j] == "H":
                            gain += 1
                # Also add small bonus for free space (helps avoid dead ends)
                free = 0
                for q in _neighbours(site):
                    if q not in occupied:
                        free += 1
                new_e = e - gain
                candidates.append((new_e, -free, site, chain, occupied))
        
        if not candidates:
            return None
        
        # Sort by energy (ascending = more negative = better), then by free space
        candidates.sort(key=lambda x: (x[0], -x[1]))
        
        # Keep top beam_width unique chains
        new_beams = []
        seen = set()
        for new_e, neg_free, site, chain, occupied in candidates:
            new_chain = chain + [site]
            key = tuple(new_chain)
            if key in seen:
                continue
            seen.add(key)
            new_occ = dict(occupied)
            new_occ[site] = i
            new_beams.append((new_chain, new_occ, new_e))
            if len(new_beams) >= beam_width:
                break
        
        beams = new_beams
        if not beams:
            return None
    
    # Return best full chain
    best = min(beams, key=lambda x: x[2])
    return best[0]
=======
def _greedy_grow(sequence, dimension, rng, width=5):
    """Fast greedy growth with random restarts. Much faster than full beam search."""
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        last = chain[-1]
        candidates = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    j = occupied.get(q)
                    if j is not None and sequence[j] == "H":
                        gain += 1
            # Add small random noise to avoid getting stuck in local minima
            noise = rng.random() * 0.1
            candidates.append((gain + noise, site))
        
        if not candidates:
            return None
            
        # Sort by gain (descending)
        candidates.sort(key=lambda x: -x[0])
        
        # Pick from top few
        top_k = min(width, len(candidates))
        _, site = candidates[rng.randint(0, top_k - 1)]
        
        chain.append(site)
        occupied[site] = i
        
    return chain
>>>>>>> REPLACE

