    # Robust ID extraction
    # If 'id' is present, use it. Otherwise, use index.
    processed_vertices = []
    for i, v in enumerate(vertices):
        if 'id' in v:
            vid = v['id']
        else:
            vid = i
        processed_vertices.append((vid, v))
        
    # Rebuild structures with robust IDs
    v_ids = [vid for vid, _ in processed_vertices]
    # Note: If IDs are not 0..N-1, parents must reference these IDs.
    # If parents are indices, this will break.
    # Given "topologically ordered parents", it is highly likely parents are IDs or Indices consistent with the list.
    # I will assume parents are IDs that match the 'id' field or the index.
    
    # To be safe, if 'id' is not in v, I assume v is just a dict with 'parents' and 'is_output', and the ID is the index.
    # If 'id' is in v, I use it.
    
    # Let's assume the standard format for these generated tasks:
    # [ { 'id': 0, 'parents': [], 'is_output': False }, ... ]
    # OR
    # [ { 'parents': [], 'is_output': False }, ... ] where ID is index.
    
    # I will stick to the previous logic but add a fallback.
