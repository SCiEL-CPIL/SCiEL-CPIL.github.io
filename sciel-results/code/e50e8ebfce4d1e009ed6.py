import time
import math

def _compatible(residues, m_new, r_new):
    for m_old, r_old in residues.items():
        g = math.gcd(m_new, m_old)
        if r_new % g == r_old % g:
            return False
    return True

def _find_residue(residues, m_new):
    for r in range(m_new):
        if _compatible(residues, m_new, r):
            return r
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130

    residues = {}
    for m in range(20, 201):
        if time.time() > deadline:
            break
        r = _find_residue(residues, m)
        if r is not None:
            residues[m] = r

    # Local improvement passes
    for iteration in range(100):
        if time.time() > deadline:
            break
        improved = False
        for m in range(20, 201):
            if time.time() > deadline:
                break
            if m in residues:
                continue
            r = _find_residue(residues, m)
            if r is not None:
                residues[m] = r
                improved = True
                continue
            for m2 in sorted(residues.keys(), key=lambda x: 1.0/x, reverse=True):
                if 1.0/m >= 1.0/m2:
                    continue
                temp = {k: v for k, v in residues.items() if k != m2}
                r2 = _find_residue(temp, m)
                if r2 is not None:
                    residues[m] = r2
                    del residues[m2]
                    improved = True
                    break
        if not improved:
            break

    return [[m, r] for m, r in sorted(residues.items())]
