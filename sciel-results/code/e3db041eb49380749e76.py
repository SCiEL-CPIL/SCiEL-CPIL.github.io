<<<<<<< SEARCH
def construct():
    t0 = time.time()
    deadline = 115.0

    result = _build_6x6_block_strassen()

    if time.time() - t0 < deadline - 5:
        try:
            if not _validate_decomposition(result["U"], result["V"], result["W"]):
                result = _build_schoolbook()
        except Exception:
            pass

    if time.time() - t0 > deadline:
        result = _build_schoolbook()

    return result
=======
def construct():
    t0 = time.time()
    deadline = 115.0

    # Try Kronecker product: 2x2 Strassen (rank 7) x 3x3 Laderman (rank 23) = rank 161
    result = _build_6x6_kronecker()
    
    if result is not None:
        if time.time() - t0 < deadline - 5:
            try:
                if not _validate_decomposition(result["U"], result["V"], result["W"]):
                    result = _build_6x6_block_strassen()
            except Exception:
                result = _build_6x6_block_strassen()
    else:
        result = _build_6x6_block_strassen()

    if time.time() - t0 < deadline - 5:
        try:
            if not _validate_decomposition(result["U"], result["V"], result["W"]):
                result = _build_schoolbook()
        except Exception:
            result = _build_schoolbook()

    if time.time() - t0 > deadline:
        result = _build_schoolbook()

    return result
>>>>>>> REPLACE

