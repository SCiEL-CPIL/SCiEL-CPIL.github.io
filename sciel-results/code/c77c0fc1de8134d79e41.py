import time
import math
import random

def count_mono_4ap(colors):
    n = 256
    c = colors
    count = 0
    for d in range(1, 86):
        max_a = n - 3*d
        for a in range(1, max_a+1):
            i = a-1
            col = c[i]
            if col == c[i+d] == c[i+2*d] == c[i+3*d]:
                count += 1
    return count

def thue_morse(n):
    seq = [0]*n
    for i in range(1, n):
        seq[i] = seq[i >> 1] ^ (i & 1)
    return seq

def build_affected(n=256):
    """For each position p (0-indexed), store list of (d, a) such that p is in the 4-AP a, a+d, a+2d, a+3d"""
    affected = [[] for _ in range(n)]
    for d in range(1, 86):
        max_a = n - 3*d
        for a in range(1, max_a+1):
            i0 = a-1
            for j in range(4):
                p = i0 + j*d
                if p < n:
                    affected[p].append((d, a))
    return affected

def build_ap_list(n=256):
    """Return list of 4-APs as tuples of 0-indexed positions"""
    aps = []
    for d in range(1, 86):
        max_a = n - 3*d
        for a in range(1, max_a+1):
            i0 = a-1
            aps.append((i0, i0+d, i0+2*d, i0+3*d))
    return aps

def delta_mono(c, p, affected):
    """Compute change in mono-4AP count if we flip position p (without modifying c)"""
    old = c[p]
    delta = 0
    for (d, a) in affected[p]:
        i0 = a - 1
        # Check if currently mono
        was_mono = (c[i0] == c[i0+d] == c[i0+2*d] == c[i0+3*d])
        # After flip: p changes, so check if all four would be same
        # The four positions are i0, i0+d, i0+2d, i0+3d
        # p is one of these. After flipping p, the value at p becomes 1-old
        vals = [c[i0], c[i0+d], c[i0+2*d], c[i0+3*d]]
        # Find which index p corresponds to
        # p = i0 + j*d for j in 0..3
        # Just check: after flip, is it mono?
        # The flipped value is 1-old
        # If was_mono: all four were 'old'. After flip, one becomes 1-old != old, so NOT mono. delta -= 1
        # If not was_mono: check if after flip all four become same
        # For it to become mono after flip, the other three must all equal 1-old
        other_vals = []
        for idx in range(4):
            pos = i0 + idx*d
            if pos != p:
                other_vals.append(c[pos])
        if was_mono:
            # Becomes non-mono
            delta -= 1
        else:
            # Check if all other three equal 1-old
            if len(other_vals) == 3 and all(v == 1-old for v in other_vals):
                delta += 1
    return delta

def local_search(colors, deadline, affected):
    c = list(colors)
    cur = count_mono_4ap(c)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for p in range(256):
            if time.time() > deadline:
                break
            d = delta_mono(c, p, affected)
            if d < 0:
                c[p] = 1 - c[p]
                cur += d
                improved = True
    return c, cur

def multi_bit_local_search(colors, deadline, affected, ap_list):
    """Local search that also tries flipping pairs of bits"""
    c = list(colors)
    cur = count_mono_4ap(c)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Single flips
        for p in range(256):
            if time.time() > deadline:
                break
            d = delta_mono(c, p, affected)
            if d < 0:
                c[p] = 1 - c[p]
                cur += d
                improved = True
        # Pair flips (limited)
        if time.time() < deadline:
            for p1 in range(256):
                if time.time() > deadline:
                    break
                for p2 in range(p1+1, 256):
                    if time.time() > deadline:
                        break
                    # Compute delta for flipping both p1 and p2
                    old1 = c[p1]
                    old2 = c[p2]
                    c[p1] = 1 - old1
                    c[p2] = 1 - old2
                    # Count change
                    delta = 0
                    affected_aps = set()
                    for (d, a) in affected[p1]:
                        affected_aps.add((d, a))
                    for (d, a) in affected[p2]:
                        affected_aps.add((d, a))
                    for (d, a) in affected_aps:
                        i0 = a - 1
                        # Check before (we already flipped, so revert to check)
                        # Actually let's compute more carefully
                        # Save current state
                        c[p1] = old1
                        c[p2] = old2
                        was_mono = (c[i0] == c[i0+d] == c[i0+2*d] == c[i0+3*d])
                        c[p1] = 1 - old1
                        c[p2] = 1 - old2
                        is_mono_now = (c[i0] == c[i0+d] == c[i0+2*d] == c[i0+3*d])
                        c[p1] = old1
                        c[p2] = old2
                        if is_mono_now and not was_mono:
                            delta += 1
                        elif was_mono and not is_mono_now:
                            delta -= 1
                    if delta < 0:
                        c[p1] = 1 - old1
                        c[p2] = 1 - old2
                        cur += delta
                        improved = True
    return c, cur

def spectral_refine(colors, deadline, ap_list):
    """
    Build the 4-AP incidence structure and use spectral methods to guide refinement.
    We construct a matrix M where M[i,j] = 1 if positions i,j are in the same 4-AP.
    Then we use the top eigenvectors to identify patterns to break.
    
    Alternative: Use the 4-AP constraint matrix to compute a "pressure" on each position
    and use it to guide flips.
    """
    c = list(colors)
    n = 256
    cur = count_mono_4ap(c)
    
    # Build adjacency: for each 4-AP, all pairs of positions in it are "coupled"
    # We'll compute a weighted degree for each position based on how many mono-APs it participates in
    affected = build_affected(n)
    
    # Compute current mono status for each AP
    ap_mono = []
    for ap in ap_list:
        i0, i1, i2, i3 = ap
        mono = (c[i0] == c[i1] == c[i2] == c[i3])
        ap_mono.append(mono)
    
    # Iteratively: find positions that are in many mono-APs, try flipping them
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Compute pressure: number of mono-APs each position is in
        pressure = [0] * n
        for idx, ap in enumerate(ap_list):
            if ap_mono[idx]:
                for pos in ap:
                    pressure[pos] += 1
        
        # Try flipping the highest-pressure position
        if max(pressure) == 0:
            break
        
        # Try all positions, pick the one with best delta
        best_p = -1
        best_delta = 0
        for p in range(n):
            if pressure[p] == 0:
                continue
            d = delta_mono(c, p, affected)
            if d < best_delta:
                best_delta = d
                best_p = p
        
        if best_p >= 0 and best_delta < 0:
            c[best_p] = 1 - c[best_p]
            cur += best_delta
            # Update ap_mono
            for idx, ap in enumerate(ap_list):
                if best_p in ap:
                    i0, i1, i2, i3 = ap
                    ap_mono[idx] = (c[i0] == c[i1] == c[i2] == c[i3])
            improved = True
    
    # Now do standard local search to polish
    c, cur = local_search(c, deadline, affected)
    return c, cur

def construct():
    start = time.time()
    deadline = start + 140
    
    candidates = []
    candidates.append(thue_morse(256))
    candidates.append([(i//2)%2 for i in range(256)])
    candidates.append([(i//3)%2 for i in range(256)])
    candidates.append([(i//4)%2 for i in range(256)])
    tm_full = thue_morse(257)
    candidates.append(tm_full[1:257])
    def v2_color(n):
        out = []
        for i in range(1, n+1):
            x = i
            p = 0
            while x % 2 == 0:
                x //= 2
                p += 1
            out.append(p % 2)
        return out
    candidates.append(v2_color(256))
    candidates.append([(i % 2) ^ ((i//2) % 2) for i in range(256)])
    # Add shifted Thue-Morse variants
    tm2 = thue_morse(260)
    candidates.append(tm2[2:258])
    candidates.append(tm2[4:260])
    # Add complement of Thue-Morse
    candidates.append([1-x for x in thue_morse(256)])
    # Add XOR with higher bit patterns
    candidates.append([(thue_morse(256)[i]) ^ ((i>>2)&1) for i in range(256)])
    candidates.append([(thue_morse(256)[i]) ^ ((i>>3)&1) for i in range(256)])
    candidates.append([(thue_morse(256)[i]) ^ ((i>>4)&1) for i in range(256)])
    
    best = None
    best_count = float('inf')
    for cand in candidates:
        if time.time() > deadline:
            break
        cnt = count_mono_4ap(cand)
        if cnt < best_count:
            best_count = cnt
            best = list(cand)
    
    if best is None:
        best = list(candidates[0])
        best_count = count_mono_4ap(best)
    
    affected = build_affected(256)
    ap_list = build_ap_list(256)
    
    # Spectral refinement first
    best, best_count = spectral_refine(best, deadline, ap_list)
    
    # Then local search to polish
    best, best_count = local_search(best, deadline, affected)
    
    return best
