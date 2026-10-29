from __future__ import annotations
import argparse
import logging
import json
import sys
from pathlib import Path
from emmy.compiler.graph import Graph

from ast_visitor import MetaStruct, NodeVisitor, PTXAssembly
from codegen import Emitter

from ast_visitor import NodeVisitor
from codegen import Emitter
from meta import MetaStruct
from ptx_ir import PTXAssembly

def compile_kernel(ir_path: Path) -> str:
    graph = Graph.from_dict(json.loads(ir_path.read_text()))
    meta = MetaStruct(json.loads((ir_path.parent / "meta.json").read_text()))
    ctx = PTXAssembly(meta.signature)
    visitor = NodeVisitor(ctx, meta)
    visitor.visit(list(graph.nodes.values()))
    return Emitter(ctx, meta, visitor.bman).emit()

def main() -> None:
    parser = argparse.ArgumentParser(description="Compile an Emmy kernel IR to PTX.")
    parser.add_argument("ir", type=Path, help="path to <kernel>.ir.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug log on stderr")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, stream=sys.stderr)
    sys.stdout.write(compile_kernel(args.ir))


if __name__ == "__main__":
    main()