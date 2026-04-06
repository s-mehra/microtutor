"""Layered DAG layout for knowledge graph visualization."""

from __future__ import annotations

from microtutor.graph import ConceptGraph


class LayeredLayout:
    """Compute fixed (x, y) positions for a DAG using layered placement.

    Algorithm:
    1. Assign each node to a layer based on its topological depth
    2. Within each layer, order nodes to minimize edge crossings
       (greedy barycenter heuristic)
    3. Compute (x, y) from layer index and position within layer

    The layout is computed ONCE when a course loads. Only colors/styles
    change at runtime -- positions are static.
    """

    def __init__(
        self,
        canvas_width: float = 700.0,
        canvas_height: float = 480.0,
        padding: float = 60.0,
        node_radius: float = 22.0,
    ):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.padding = padding
        self.node_radius = node_radius

    def compute(self, graph: ConceptGraph) -> dict[str, tuple[float, float]]:
        """Return {concept_id: (x, y)} positions."""
        all_ids = graph.get_all_concept_ids()
        if not all_ids:
            return {}

        # Step 1: Assign layers by topological depth
        layers: dict[int, list[str]] = {}
        for cid in all_ids:
            depth = graph.get_topological_depth(cid)
            layers.setdefault(depth, []).append(cid)

        max_depth = max(layers.keys()) if layers else 0

        # Step 2: Barycenter ordering within each layer
        ordered_layers = self._barycenter_ordering(layers, graph, max_depth)

        # Step 3: Compute pixel positions
        usable_w = self.canvas_width - 2 * self.padding
        usable_h = self.canvas_height - 2 * self.padding

        positions = {}
        for depth, layer in ordered_layers.items():
            n = len(layer)
            y = self.padding + (depth / max(max_depth, 1)) * usable_h

            for i, cid in enumerate(layer):
                if n == 1:
                    x = self.canvas_width / 2
                else:
                    x = self.padding + (i / (n - 1)) * usable_w
                positions[cid] = (x, y)

        return positions

    def compute_from_depths(
        self,
        concepts: list[dict],
        edges: list[tuple[str, str]],
    ) -> dict[str, tuple[float, float]]:
        """Compute layout from state data (no ConceptGraph needed).

        Args:
            concepts: list of dicts with 'id' and 'depth' keys
            edges: list of (source_id, target_id) tuples
        """
        if not concepts:
            return {}

        # Build layers from depth values
        layers: dict[int, list[str]] = {}
        for c in concepts:
            layers.setdefault(c["depth"], []).append(c["id"])

        max_depth = max(layers.keys()) if layers else 0

        # Build parent map for barycenter ordering
        parent_map: dict[str, list[str]] = {c["id"]: [] for c in concepts}
        for src, tgt in edges:
            if tgt in parent_map:
                parent_map[tgt].append(src)

        # Barycenter ordering
        x_index: dict[str, float] = {}
        ordered: dict[int, list[str]] = {}

        for d in range(max_depth + 1):
            layer = layers.get(d, [])
            if d == 0:
                ordered[d] = layer
                for i, cid in enumerate(layer):
                    x_index[cid] = float(i)
            else:
                def bary(cid, _pm=parent_map, _xi=x_index):
                    parents = _pm.get(cid, [])
                    if not parents:
                        return 0.0
                    return sum(_xi.get(p, 0.0) for p in parents) / len(parents)

                layer_sorted = sorted(layer, key=bary)
                ordered[d] = layer_sorted
                for i, cid in enumerate(layer_sorted):
                    x_index[cid] = float(i)

        # Compute pixel positions
        usable_w = self.canvas_width - 2 * self.padding
        usable_h = self.canvas_height - 2 * self.padding

        positions = {}
        for depth, layer in ordered.items():
            n = len(layer)
            y = self.padding + (depth / max(max_depth, 1)) * usable_h
            for i, cid in enumerate(layer):
                x = self.canvas_width / 2 if n == 1 else self.padding + (i / (n - 1)) * usable_w
                positions[cid] = (x, y)

        return positions

    def _barycenter_ordering(
        self, layers: dict[int, list[str]], graph: ConceptGraph, max_depth: int
    ) -> dict[int, list[str]]:
        """Order nodes within each layer by average parent position."""
        ordered = {0: layers.get(0, [])}
        x_index: dict[str, float] = {cid: float(i) for i, cid in enumerate(ordered[0])}

        for d in range(1, max_depth + 1):
            layer = layers.get(d, [])

            def bary(cid, _g=graph, _xi=x_index):
                preds = _g.get_prerequisites(cid)
                if not preds:
                    return 0.0
                return sum(_xi.get(p, 0.0) for p in preds) / len(preds)

            layer_sorted = sorted(layer, key=bary)
            ordered[d] = layer_sorted
            for i, cid in enumerate(layer_sorted):
                x_index[cid] = float(i)

        return ordered
