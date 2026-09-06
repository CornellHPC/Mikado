#!/usr/bin/env python
"""Convert a GRG file into the lower-triangular CSR matrix ``M = I - A``.

The GRG linear operator's UP traversal computes, in stable-height (topological)
order::

    node_values = b + A @ node_values        # A strictly block-lower-triangular

which is algebraically the triangular solve ``(I - A) x = b``.  This script
materializes ``M = I - A`` as a single CSR matrix in stable-height order and
writes it to a compact little-endian binary file that the companion
``sptrsv.cu`` benchmark loads and solves with cuSPARSE SpTrSV.

Binary ``.csr`` layout (little-endian)::

    magic        : 8 bytes  = b"GRGTRSV1"
    index_bytes  : int32    = 4 (int32 col indices) or 8 (int64 col indices)
    nrows        : int64    (= num_nodes)
    ncols        : int64    (= num_nodes)
    nnz          : int64
    indptr       : (nrows + 1) x int64
    indices      : nnz x (index_bytes)
    values       : nnz x float64

Usage::

    grg_to_csr.py <input.grg> <output.csr>
"""

from __future__ import annotations

import argparse
import struct
import sys

import numpy as np
import pygrgl
import scipy.sparse as sp

# Reuse the exact ordering logic the SpMM compiler uses so the matrix matches
# the reference operator's node permutation.
from pygrgl_spmv.grg.compile import _build_stable_height_order, _compute_node_levels

MAGIC = b"GRGTRSV1"


def build_lower_triangular(grg) -> sp.csr_matrix:
    """Build ``M = I - A`` in stable-height order as a sorted CSR matrix."""
    num_nodes = int(grg.num_nodes)
    num_edges = int(grg.num_edges)

    # Stable height order: children before parents.  inv_node_perm maps an
    # original NodeID to its row/column index in the reordered matrix.
    node_levels = _compute_node_levels(
        grg, num_nodes, node_id_scratch_dtype=np.int64
    )
    _node_perm, inv_node_perm, _level_offsets = _build_stable_height_order(node_levels)
    inv = np.asarray(inv_node_perm, dtype=np.int64)

    # Collect one (-1) entry per down-edge parent -> child.  Preallocate to
    # num_edges to avoid Python-list overhead on large graphs.
    rows = np.empty(num_edges, dtype=np.int64)
    cols = np.empty(num_edges, dtype=np.int64)
    pos = 0
    for parent_id in range(num_nodes):
        children = grg.get_down_edges(parent_id)
        if not len(children):
            continue
        child_arr = np.asarray(children, dtype=np.int64)
        n = child_arr.size
        rows[pos : pos + n] = inv[parent_id]
        cols[pos : pos + n] = inv[child_arr]
        pos += n
    rows = rows[:pos]
    cols = cols[:pos]

    # col < row is guaranteed (children sit at strictly lower levels), so the
    # result is strictly lower-triangular before the identity is added.
    minus_a = sp.coo_matrix(
        (np.full(pos, -1.0, dtype=np.float64), (rows, cols)),
        shape=(num_nodes, num_nodes),
    )
    m = (sp.identity(num_nodes, format="csr", dtype=np.float64) - minus_a).tocsr()
    m.sort_indices()
    return m


def write_csr(path: str, m: sp.csr_matrix) -> None:
    """Serialize a CSR matrix to the binary ``.csr`` format."""
    nrows, ncols = m.shape
    # cuSPARSE requires csrRowOffsetsType == csrColIndType, so indptr and
    # indices share one width, picked to hold both the node count and nnz.
    if max(ncols, int(m.nnz)) < 2**31:
        index_bytes = 4
        index_dtype = np.int32
    else:
        index_bytes = 8
        index_dtype = np.int64
    indptr = np.ascontiguousarray(m.indptr, dtype=index_dtype)
    indices = np.ascontiguousarray(m.indices, dtype=index_dtype)
    values = np.ascontiguousarray(m.data, dtype=np.float64)

    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<i", index_bytes))
        fh.write(struct.pack("<qqq", nrows, ncols, int(m.nnz)))
        indptr.tofile(fh)
        indices.tofile(fh)
        values.tofile(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_grg", help="path to input .grg file")
    parser.add_argument("output_csr", help="path to output .csr file")
    args = parser.parse_args(argv)

    grg = pygrgl.load_immutable_grg(args.input_grg, load_up_edges=False)
    print(
        f"loaded {args.input_grg}: "
        f"num_nodes={grg.num_nodes} num_edges={grg.num_edges} "
        f"num_samples={grg.num_samples} num_mutations={grg.num_mutations}"
    )

    m = build_lower_triangular(grg)
    write_csr(args.output_csr, m)

    expected_nnz = int(grg.num_nodes) + int(grg.num_edges)
    index_bytes = 4 if max(m.shape[1], int(m.nnz)) < 2**31 else 8
    print(
        f"wrote {args.output_csr}: n={m.shape[0]} nnz={m.nnz} "
        f"(expected num_nodes+num_edges={expected_nnz}) "
        f"index_bytes={index_bytes}"
    )
    if m.nnz != expected_nnz:
        print(
            "WARNING: nnz != num_nodes + num_edges; the graph may contain "
            "duplicate down-edges or self-loops.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
