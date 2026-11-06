    def solve_for_moduli(moduli, time_limit):
        # Returns list of pairs or None
        # Uses backtracking with ordering
        # Order moduli by ascending value (more constrained)
        moduli = sorted(moduli)
        
        def backtrack(idx, pairs):
            if time.time() > time_limit:
                return None
            if idx == len(moduli):
                return pairs[:]
            
            m = moduli[idx]
            # Get valid residues
            # Optimization: Precompute gcds with existing pairs?
            valid = []
            for r in range(m):
                ok = True
                for r2, m2 in pairs:
                    if (r - r2) % math.gcd(m, m2) == 0:
                        ok = False
                        break
                if ok:
                    valid.append(r)
            
            if not valid:
                return None
            
            # Heuristic: Choose residue that minimizes future conflicts?
            # Or just random shuffle to avoid bias
            random.shuffle(valid)
            
            for r in valid:
                res = backtrack(idx + 1, pairs + [(r, m)])
                if res:
                    return res
            return None
            
        return backtrack(0, [])
    