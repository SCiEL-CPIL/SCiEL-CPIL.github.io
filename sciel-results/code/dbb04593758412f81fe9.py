        if not children:
            # Leaf
            taxa = node_to_taxa.get(node, set())
            if taxa:
                taxon_id = list(taxa)[0]
                return labels[taxon_id]
            else:
                return f"n{node}"
