    def can_add_after_remove(e_add, e_rem):
        for fid in edge_faces[e_add]:
            if face_count[fid] >= 3:
                # Check if e_rem is in this face
                # e_rem is in fid if e_add and e_rem share a face
                # We can precompute edge conflicts or just check
                # face_edges[fid] contains 4 edges
                if e_rem not in face_edges[fid]:
                    return False
        return True
