<<<<<<< SEARCH
def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    # Generate diverse initial designs
    initial_designs = []
    
    # Single layer designs
    initial_designs.append((["L"], [centre / (4.0 * n_L)]))
    initial_designs.append((["H"], [centre / (4.0 * n_H)]))
    
    # Two-layer designs
    initial_designs.append((["L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Three-layer designs
    initial_designs.append((["H", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Four-layer designs
    initial_designs.append((["H", "L", "H", "L"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    initial_designs.append((["L", "H", "L", "H"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # Five-layer designs
    initial_designs.append((["H", "L", "H", "L", "H"], 
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H), 
                            centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["L", "H", "L", "H", "L"], 
                           [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L), 
                            centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Six-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Seven-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H)]))
    
    # Eight-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Nine-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H)]))
    
    # Ten-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L", "H", "L"],
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_H),
                            centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H), centre / (4.0 * n_L),
                            centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Non-standard patterns
    # HLLH pattern
    initial_designs.append((["H", "L", "L", "H"], [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    initial_designs.append((["H", "L", "L", "H", "L", "L", "H"], 
                           [centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), 
                            centre / (4.0 * n_H), centre / (4.0 * n_L), centre / (4.0 * n_L), centre / (4.0 * n_H)]))
    
    # LHL pattern
    initial_designs.append((["L", "H", "L"], [centre / (4.0 * n_L), centre / (4.0 * n_H), centre / (4.0 * n_L)]))
    
    # Evaluate all initial designs and keep the best
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    for mats, thicks in initial_designs:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _evaluate_design(mats, thicks, problem, wavelengths)
        if score < best_score:
            best_score = score
            best_materials = list(mats)
            best_thicknesses = list(thicks)
    
    # Fallback if no valid initial design
    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)
    
    # Optimize the best initial design
    best_materials, best_thicknesses, best_score = _optimize_thicknesses(
        best_materials, best_thicknesses, problem, wavelengths, deadline
    )
    
    # Try other promising sequences with shorter optimization
    # Generate additional candidate sequences
    candidate_sequences = []
    
    # Alternating sequences of various lengths
    for n_layers in range(1, min(15, max_layers + 1)):
        # Starting with H
        seq = []
        for i in range(n_layers):
            seq.append("H" if i % 2 == 0 else "L")
        candidate_sequences.append(seq)
        
        # Starting with L
        seq = []
        for i in range(n_layers):
            seq.append("L" if i % 2 == 0 else "H")
        candidate_sequences.append(seq)
    
    # Some non-alternating patterns
    for n_layers in [4, 6, 8, 10, 12]:
        # HLLH repeated
        if n_layers % 4 == 0:
            seq = []
            for i in range(n_layers):
                pos = i % 4
                if pos == 0 or pos == 3:
                    seq.append("H")
                else:
                    seq.append("L")
            candidate_sequences.append(seq)
        
        # LHHL repeated
        if n_layers % 4 == 0:
            seq = []
            for i in range(n_layers):
                pos = i % 4
                if pos == 0 or pos == 1:
                    seq.append("L" if pos == 0 else "H")
                else:
                    seq.append("H" if pos == 2 else "L")
            # Simplified: just add a few more patterns
            pass
    
    # Add some specific patterns
    candidate_sequences.append(["H", "L", "L", "H", "L", "L", "H"])
    candidate_sequences.append(["L", "H", "H", "L", "H", "H", "L"])
    candidate_sequences.append(["H", "L", "H", "L", "L", "H", "L", "H"])
    candidate_sequences.append(["L", "H", "L", "H", "H", "L", "H", "L"])
    
    # Deduplicate and filter
    seen = set()
    unique_candidates = []
    for seq in candidate_sequences:
        if len(seq) > max_layers:
            continue
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(seq)
    
    # Evaluate and optimize each candidate
    for seq in unique_candidates:
        if time.time() > deadline:
            break
        
        if seq == best_materials:
            continue
        
        thicks = [centre / (4.0 * materials_dict[m]) for m in seq]
        
        if not _is_valid(seq, thicks, problem):
            continue
        
        score = _evaluate_design(seq, thicks, problem, wavelengths)
        
        # Only optimize if promising
        if score < best_score + 3.0:
            mats, thicks, score = _optimize_thicknesses(
                seq, thicks, problem, wavelengths, deadline, max_iter=1500
            )
            if score < best_score:
                best_score = score
                best_materials = list(seq)
                best_thicknesses = list(thicks)
    
    # Final refinement on the best design
    if time.time() < deadline - 5.0:
        best_materials, best_thicknesses, best_score = _optimize_thicknesses(
            best_materials, best_thicknesses, problem, wavelengths, deadline, max_iter=2000
        )
    
    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
=======
def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    materials_dict = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    n_L = materials_dict["L"]
    n_H = materials_dict["H"]
    
    centre = 0.5 * (band_nm[0] + band_nm[1])
    
    # Generate diverse initial designs with better optical path considerations
    initial_designs = []
    
    # Use quarter-wave at different reference wavelengths for broadband coverage
    ref_wavelengths = [450, 500, 550, 600, 650]
    
    # Single layer designs
    for ref_wl in ref_wavelengths:
        initial_designs.append((["L"], [ref_wl / (4.0 * n_L)]))
        initial_designs.append((["H"], [ref_wl / (4.0 * n_H)]))
    
    # Two-layer designs - optimize for different center wavelengths
    for ref_wl in ref_wavelengths:
        initial_designs.append((["L", "H"], [ref_wl / (4.0 * n_L), ref_wl / (4.0 * n_H)]))
        initial_designs.append((["H", "L"], [ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L)]))
    
    # Three-layer designs
    for ref_wl in [500, 550, 600]:
        initial_designs.append((["H", "L", "H"], 
                               [ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L), ref_wl / (4.0 * n_H)]))
        initial_designs.append((["L", "H", "L"], 
                               [ref_wl / (4.0 * n_L), ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L)]))
    
    # Four-layer designs
    for ref_wl in [500, 550, 600]:
        initial_designs.append((["H", "L", "H", "L"], 
                               [ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L), 
                                ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L)]))
        initial_designs.append((["L", "H", "L", "H"], 
                               [ref_wl / (4.0 * n_L), ref_wl / (4.0 * n_H), 
                                ref_wl / (4.0 * n_L), ref_wl / (4.0 * n_H)]))
    
    # Five-layer designs
    for ref_wl in [500, 550, 600]:
        initial_designs.append((["H", "L", "H", "L", "H"], 
                               [ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L), 
                                ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L), 
                                ref_wl / (4.0 * n_H)]))
        initial_designs.append((["L", "H", "L", "H", "L"], 
                               [ref_wl / (4.0 * n_L), ref_wl / (4.0 * n_H), 
                                ref_wl / (4.0 * n_L), ref_wl / (4.0 * n_H), 
                                ref_wl / (4.0 * n_L)]))
    
    # Six-layer designs
    for ref_wl in [500, 550]:
        initial_designs.append((["H", "L", "H", "L", "H", "L"],
                               [ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L), 
                                ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L), 
                                ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L)]))
        initial_designs.append((["L", "H", "L", "H", "L", "H"],
                               [ref_wl / (4.0 * n_L), ref_wl / (4.0 * n_H), 
                                ref_wl / (4.0 * n_L), ref_wl / (4.0 * n_H), 
                                ref_wl / (4.0 * n_L), ref_wl / (4.0 * n_H)]))
    
    # Seven-layer designs
    for ref_wl in [500, 550]:
        initial_designs.append((["H", "L", "H", "L", "H", "L", "H"],
                               [ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L), 
                                ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L), 
                                ref_wl / (4.0 * n_H), ref_wl / (4.0 * n_L), 
                                ref_wl / (4.0 * n_H)]))
    
    # Eight-layer designs
    initial_designs.append((["H", "L", "H", "L", "H", "L", "H", "L"],
                           [550 / (4.0 * n_H), 550 / (4.0 * n_L), 
                            550 / (4.0 * n_H), 550 / (4.0 * n_L), 
                            550 / (4.0 * n_H), 550 / (4.0 * n_L), 
                            550 / (4.0 * n_H), 550 / (4.0 * n_L)]))
    
    # Non-standard patterns
    initial_designs.append((["H", "L", "L", "H"], 
                           [550 / (4.0 * n_H), 550 / (4.0 * n_L), 
                            550 / (4.0 * n_L), 550 / (4.0 * n_H)]))
    initial_designs.append((["L", "H", "H", "L"], 
                           [550 / (4.0 * n_L), 550 / (4.0 * n_H), 
                            550 / (4.0 * n_H), 550 / (4.0 * n_L)]))
    
    # Evaluate all initial designs and keep the best
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    for mats, thicks in initial_designs:
        if not _is_valid(mats, thicks, problem):
            continue
        score = _evaluate_design(mats, thicks, problem, wavelengths)
        if score < best_score:
            best_score = score
            best_materials = list(mats)
            best_thicknesses = list(thicks)
    
    # Fallback if no valid initial design
    if best_materials is None:
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
        best_score = _evaluate_design(best_materials, best_thicknesses, problem, wavelengths)
    
    # Optimize the best initial design with Nelder-Mead
    best_materials, best_thicknesses, best_score = _nelder_mead_optimize(
        best_materials, best_thicknesses, problem, wavelengths, deadline
    )
    
    # Try other promising sequences
    candidate_sequences = []
    
    # Alternating sequences of various lengths
    for n_layers in range(2, min(12, max_layers + 1)):
        # Starting with H
        seq = []
        for i in range(n_layers):
            seq.append("H" if i % 2 == 0 else "L")
        candidate_sequences.append(seq)
        
        # Starting with L
        seq = []
        for i in range(n_layers):
            seq.append("L" if i % 2 == 0 else "H")
        candidate_sequences.append(seq)
    
    # Non-alternating patterns
    candidate_sequences.extend([
        ["H", "L", "L", "H"],
        ["L", "H", "H", "L"],
        ["H", "L", "L", "H", "L", "L", "H"],
        ["L", "H", "H", "L", "H", "H", "L"],
        ["H", "L", "H", "L", "L", "H", "L", "H"],
        ["L", "H", "L", "H", "H", "L", "H", "L"],
    ])
    
    # Deduplicate and filter
    seen = set()
    unique_candidates = []
    for seq in candidate_sequences:
        if len(seq) > max_layers:
            continue
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(seq)
    
    # Evaluate and optimize each candidate with Nelder-Mead
    for seq in unique_candidates:
        if time.time() > deadline - 10.0:
            break
        
        if seq == best_materials:
            continue
        
        # Use center wavelength for initial guess
        thicks = [centre / (4.0 * materials_dict[m]) for m in seq]
        
        if not _is_valid(seq, thicks, problem):
            continue
        
        score = _evaluate_design(seq, thicks, problem, wavelengths)
        
        # Only optimize if promising
        if score < best_score + 5.0:
            mats, thicks, score = _nelder_mead_optimize(
                seq, thicks, problem, wavelengths, deadline - 5.0, max_iter=1500
            )
            if score < best_score:
                best_score = score
                best_materials = list(seq)
                best_thicknesses = list(thicks)
    
    # Final refinement on the best design with tighter tolerance
    if time.time() < deadline - 10.0:
        best_materials, best_thicknesses, best_score = _nelder_mead_optimize(
            best_materials, best_thicknesses, problem, wavelengths, deadline - 5.0, max_iter=2000
        )
    
    # Final validation
    if not _is_valid(best_materials, best_thicknesses, problem):
        best_materials = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    return best_materials, best_thicknesses
>>>>>>> REPLACE
