#!/usr/bin/env python3
"""
ptx_formatter.py — Make PTX assembly human-readable.

Transforms raw cuobjdump PTX output into formatted, annotated files
with demangled names, section headers, and instruction comments.
"""

import re
import os
from typing import Optional
from symbol_utils import demangle_symbol, classify_kernel, format_kernel_header, extract_kernel_metadata


# PTX instruction categories for annotation
INSTRUCTION_CATEGORIES = {
    "ld": "LOAD",
    "st": "STORE",
    "add": "ARITH",
    "sub": "ARITH",
    "mul": "ARITH",
    "fma": "ARITH (fused multiply-add)",
    "mad": "ARITH (multiply-add)",
    "div": "ARITH (division)",
    "setp": "COMPARE",
    "bra": "BRANCH",
    "bar": "SYNC (barrier)",
    "atom": "ATOMIC",
    "shfl": "WARP SHUFFLE",
    "vote": "WARP VOTE",
    "mov": "MOVE",
    "cvt": "CONVERT",
    "cvta": "ADDRESS CONVERT",
    "shr": "SHIFT",
    "shl": "SHIFT",
    "and": "LOGIC",
    "or": "LOGIC",
    "xor": "LOGIC",
    "not": "LOGIC",
    "prmt": "BYTE PERMUTE",
    "popc": "BIT COUNT",
    "bfe": "BIT EXTRACT",
    "bfi": "BIT INSERT",
    "tex": "TEXTURE",
    "suld": "SURFACE LOAD",
    "sust": "SURFACE STORE",
    "red": "REDUCTION",
    "prefetch": "PREFETCH",
    "ret": "RETURN",
    "call": "CALL",
    "exit": "EXIT",
}


def annotate_instruction(line: str) -> Optional[str]:
    """Add an annotation comment to a PTX instruction line."""
    stripped = line.strip()
    if not stripped or stripped.startswith("//") or stripped.startswith("."):
        return None

    # Get the instruction mnemonic
    parts = stripped.split()
    if not parts:
        return None

    mnemonic = parts[0].rstrip(";").rstrip(",")

    # Handle predicated instructions
    if mnemonic.startswith("@"):
        if len(parts) > 1:
            mnemonic = parts[1].rstrip(";").rstrip(",")
        else:
            return None

    # Look up category
    base = mnemonic.split(".")[0]
    for prefix, category in INSTRUCTION_CATEGORIES.items():
        if base.startswith(prefix):
            return category

    return None


def format_ptx_section(ptx_text: str, max_lines_per_kernel: int = 0) -> str:
    """
    Format PTX text with annotations and section headers.

    Args:
        ptx_text: Raw PTX from cuobjdump
        max_lines_per_kernel: 0 = full, >0 = truncate to N lines
    """
    kernels = extract_kernel_metadata(ptx_text)
    if not kernels:
        return _format_raw_ptx(ptx_text)

    output = []
    output.append("=" * 72)
    output.append(f"  PTX Dump — {len(kernels)} kernel(s) found")
    output.append("=" * 72)
    output.append("")

    # Summary table
    output.append("  Kernel Summary:")
    output.append(f"  {'#':>3s}  {'Category':<20s}  {'Registers':>9s}  {'Lines':>6s}  Name")
    output.append(f"  {'─' * 3}  {'─' * 20}  {'─' * 9}  {'─' * 6}  {'─' * 30}")
    for i, k in enumerate(kernels):
        name_short = k["demangled_name"][:50]
        output.append(
            f"  {i + 1:>3d}  {k['classification']:<20s}  "
            f"{k['registers']:>9d}  {k['lines']:>6d}  {name_short}"
        )
    output.append("")
    output.append("=" * 72)
    output.append("")

    # Full PTX for each kernel (from the raw text)
    kernel_blocks = _split_ptx_into_kernels(ptx_text)

    for i, (meta, block) in enumerate(zip(kernels, kernel_blocks)):
        output.append(format_kernel_header(meta))
        output.append("")

        lines = block.split("\n")
        if max_lines_per_kernel > 0 and len(lines) > max_lines_per_kernel:
            lines = lines[:max_lines_per_kernel]
            lines.append(f"\n  ... ({len(block.split(chr(10))) - max_lines_per_kernel} more lines truncated)\n")

        for line in lines:
            annotation = annotate_instruction(line)
            if annotation:
                # Right-align annotation
                padded = line.rstrip()
                if len(padded) < 60:
                    padding = " " * (60 - len(padded))
                else:
                    padding = "  "
                output.append(f"{padded}{padding}// {annotation}")
            else:
                output.append(line)

        output.append("")

    return "\n".join(output)


def _split_ptx_into_kernels(ptx_text: str) -> list:
    """Split PTX text into per-kernel blocks."""
    blocks = []
    current = []
    header = []
    seen_first_entry = False

    for line in ptx_text.split("\n"):
        # Match .entry or .visible .entry
        is_entry = re.match(r"(\.visible\s+)?\.entry\s+", line)
        if is_entry:
            if not seen_first_entry:
                seen_first_entry = True
            else:
                # Save previous kernel block
                blocks.append("\n".join(current))
            current = [line]
        elif seen_first_entry:
            current.append(line)
        else:
            header.append(line)

    if current:
        blocks.append("\n".join(current))

    return blocks


def _format_raw_ptx(ptx_text: str) -> str:
    """Fallback: format raw PTX with basic annotations."""
    output = []
    output.append("=" * 72)
    output.append("  PTX Dump (raw)")
    output.append("=" * 72)
    output.append("")

    for line in ptx_text.split("\n"):
        annotation = annotate_instruction(line)
        if annotation:
            padded = line.rstrip()
            padding = " " * max(1, 60 - len(padded))
            output.append(f"{padded}{padding}// {annotation}")
        else:
            output.append(line)

    return "\n".join(output)


def write_formatted_ptx(ptx_text: str, output_dir: str, prefix: str = "dump",
                        split_per_kernel: bool = True,
                        max_lines_per_kernel: int = 0) -> list:
    """
    Write formatted PTX to files.

    Args:
        ptx_text: Raw PTX from cuobjdump
        output_dir: Directory to write to
        prefix: Filename prefix
        split_per_kernel: If True, write one file per kernel
        max_lines_per_kernel: 0 = full

    Returns:
        List of written file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    written = []

    kernels = extract_kernel_metadata(ptx_text)

    if split_per_kernel and kernels:
        kernel_blocks = _split_ptx_into_kernels(ptx_text)

        for i, (meta, block) in enumerate(zip(kernels, kernel_blocks)):
            # Sanitize filename
            safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", meta["demangled_name"])[:80]
            filename = f"{prefix}_{i + 1:03d}_{safe_name}.ptx"
            filepath = os.path.join(output_dir, filename)

            formatted = format_kernel_header(meta) + "\n\n"
            for line in block.split("\n"):
                annotation = annotate_instruction(line)
                if annotation:
                    padded = line.rstrip()
                    padding = " " * max(1, 60 - len(padded))
                    formatted += f"{padded}{padding}// {annotation}\n"
                else:
                    formatted += line + "\n"

            with open(filepath, "w") as f:
                f.write(formatted)
            written.append(filepath)

    # Always write combined file
    combined = format_ptx_section(ptx_text, max_lines_per_kernel)
    combined_path = os.path.join(output_dir, f"{prefix}_all.ptx")
    with open(combined_path, "w") as f:
        f.write(combined)
    written.insert(0, combined_path)

    return written
