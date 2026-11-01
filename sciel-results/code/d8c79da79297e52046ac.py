<<<<<<< SEARCH
def construct(k, g):
    start_time = time.time()
    deadline = 140.0

    if k != 3:
        return _luw_construct(k, g)

    # Try Voltage Lift of Dodecahedron
    result = _voltage_lift_dodecahedron(k, g, deadline)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # Fallback to LUW
    return _luw_construct(k, g)
=======
def construct(k, g):
    start_time = time.time()
    deadline = 140.0

    if k != 3:
        return _luw_construct(k, g)

    # Try Tutte-Coxeter style construction first (may give smaller n)
    result = _tutte_coxeter_cage(k, g, deadline)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # Fallback to LUW (deterministic, known to work for girth 16)
    return _luw_construct(k, g)
>>>>>>> REPLACE
