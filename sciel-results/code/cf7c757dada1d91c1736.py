                for i in range(len(circuit) - 1):
                    u, v = circuit[i], circuit[i+1]
                    e = tuple(sorted((u, v)))
                    if e in original_edges_set:
                        if not current_path:
                            current_path = [u]
                            current_vertices = {u}
                        
                        if v in current_vertices:
                            if len(current_path) > 1:
                                path_segments.append(current_path)
                            current_path = [u, v]
                            current_vertices = {u, v}
                        else:
                            current_path.append(v)
                            current_vertices.add(v)
                    else:
                        # Virtual edge: break the path
                        if current_path and len(current_path) > 1:
                            path_segments.append(current_path)
                        current_path = []
                        current_vertices = set()
