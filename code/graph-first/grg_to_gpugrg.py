#!/usr/bin/env python
"""Convert a GRG file into the legacy GPUGRG binary format (``.gpugrg``).

The legacy GRGL GPU backend (``pygrgl.GPUGRG``) is normally built in RAM from a
loaded GRG via ``pygrgl.grg_to_gpu``. This script performs that conversion once
and serializes the result with ``pygrgl.store_gpu_grg`` so the
``examples/kernel/legacy_gpugrg.py`` benchmark can load a precomputed
``.gpugrg`` directly (``pygrgl.load_gpu_grg``) instead of re-converting on every
run -- mirroring how the cuSPARSE / SpTrSV backends consume precomputed
``.grg_spmv`` / ``.csr`` artifacts.

Requires a CUDA-enabled pygrgl in which ``grg_to_gpu`` / ``store_gpu_grg`` are
available (the conversion runs on the GPU).

Usage::

    grg_to_gpugrg.py <input.grg> <output.gpugrg>
"""

from __future__ import annotations

import argparse

import pygrgl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input_grg", help="path to input .grg file")
    parser.add_argument("output_gpugrg", help="path to output .gpugrg file")
    args = parser.parse_args(argv)

    grg = pygrgl.load_immutable_grg(args.input_grg, load_up_edges=True)
    print(
        f"loaded {args.input_grg}: "
        f"num_samples={grg.num_samples} num_mutations={grg.num_mutations} "
        f"num_individuals={grg.num_individuals}"
    )

    gpugrg = pygrgl.grg_to_gpu(grg)
    pygrgl.store_gpu_grg(gpugrg, args.output_gpugrg)
    print(
        f"wrote {args.output_gpugrg}: "
        f"num_samples={gpugrg.num_samples} num_mutations={gpugrg.num_mutations}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
