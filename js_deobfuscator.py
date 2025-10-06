#!/usr/bin/env python3
"""
Fixed JavaScript Deobfuscator for _0x-style string table obfuscation

Key fixes:
1. Aggressive asterisk removal (multiple passes)
2. Correct offset calculation with multiple pattern attempts
3. Proper rotation simulation matching IIFE arithmetic
4. Handles both _0x and *0x in resolver calls
"""

from __future__ import annotations
import re
import sys
import os
import argparse
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class DeobfuscationResult:
    code: str
    mapping: Dict[str, Optional[str]]
    resolver_calls_before: int
    resolver_calls_after: int
    passes_completed: int
    syntax_fixes_applied: int


class SyntaxFixer:
    """Enhanced syntax fixer with multiple aggressive passes."""
    
    @staticmethod
    def fix_syntax(js: str) -> Tuple[str, int]:
        """Fix asterisk syntax errors with multiple passes."""
        original = js
        
        # Run multiple passes to catch all cases
        for iteration in range(10):
            before = js
            
            # Replace all *0x patterns
            js = re.sub(r'\*0x', '_0x', js)
            
            # Fix keyword declarations
            js = re.sub(r'\b(const|let|var|function)\s+\*', r'\1 _', js)
            
            # If no changes, we're done
            if js == before:
                break
        
        # Count how many asterisks were fixed
        fixes = original.count('*0x')
        
        return js, fixes


class StringTableExtractor:
    """Extracts string tables from obfuscated code."""
    
    @staticmethod
    def find_table(js: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """Find the string table array."""
        # Pattern: const _0x3bec45 = ['...', '...', ...]
        pattern = re.compile(
            r"(?:const|let|var)\s+(_0x[a-fA-F0-9_]+)\s*=\s*(\[[^\]]*\])",
            re.DOTALL
        )
        match = pattern.search(js)
        
        if not match:
            return None, None
        
        arr_name = match.group(1)
        arr_literal = match.group(2)
        
        return arr_name, StringTableExtractor._parse_string_array(arr_literal)
    
    @staticmethod
    def _parse_string_array(arr_literal: str) -> List[str]:
        """Extract strings from array literal."""
        items = re.findall(
            r"'((?:\\'|[^'])*)'|\"((?:\\\"|[^\"])*)\"",
            arr_literal
        )
        
        result = []
        for single_quoted, double_quoted in items:
            string_value = single_quoted if single_quoted else double_quoted
            decoded = StringTableExtractor._decode_string(string_value)
            result.append(decoded)
        
        return result
    
    @staticmethod
    def _decode_string(s: str) -> str:
        """Decode JavaScript escape sequences."""
        # Hex escapes
        s = re.sub(r'\\x([0-9a-fA-F]{2})', 
                   lambda m: chr(int(m.group(1), 16)), s)
        
        # Unicode escapes
        s = re.sub(r'\\u([0-9a-fA-F]{4})', 
                   lambda m: chr(int(m.group(1), 16)), s)
        
        # Standard escapes
        escape_map = {
            '\\n': '\n', '\\r': '\r', '\\t': '\t',
            '\\b': '\b', '\\f': '\f', '\\v': '\v',
            "\\'": "'", '\\"': '"', '\\\\': '\\'
        }
        
        for escaped, actual in escape_map.items():
            s = s.replace(escaped, actual)
        
        return s


class ResolverAnalyzer:
    """Analyzes resolver functions and IIFE patterns."""
    
    @staticmethod
    def find_offset(js: str) -> Optional[int]:
        """
        Find the offset used in index calculations.
        Tries multiple patterns.
        """
        patterns = [
            # Pattern 1: _idx = _idx - (expr)
            r"_0x[0-9a-fA-F_]+\s*=\s*_0x[0-9a-fA-F_]+\s*-\s*\(\s*([^)]+)\)",
            # Pattern 2: Inside array access with subtraction
            r"\[_0x[0-9a-fA-F_]+\]\s*;\s*return[^;]*-\s*\(\s*([^)]+)\)",
            # Pattern 3: Direct calculation
            r"function.*?\{\s*_0x[0-9a-fA-F_]+\s*=\s*_0x[0-9a-fA-F_]+\s*-\s*\(\s*([^)]+)\)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, js, re.DOTALL)
            if match:
                expr = match.group(1)
                result = ResolverAnalyzer._eval_hex_expression(expr)
                if result is not None:
                    return result
        
        return None
    
    @staticmethod
    def _eval_hex_expression(expr: str) -> Optional[int]:
        """Safely evaluate hex arithmetic expressions."""
        try:
            # Replace hex literals with decimals
            expr_py = re.sub(
                r'0x[0-9a-fA-F]+',
                lambda m: str(int(m.group(0), 16)),
                expr
            )
            # Safe eval
            return int(eval(expr_py, {"__builtins__": {}}))
        except:
            return None
    
    @staticmethod
    def extract_iife_info(js: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Extract IIFE rotation target and arithmetic expression.
        
        Returns:
            (target_value, expression_body)
        """
        # Find IIFE call: }(_0xcdde, <target_expression>));
        call_pattern = re.compile(
            r"\}\s*\(\s*_0x[a-fA-F0-9_]+\s*,\s*([^)]+)\)\s*\)",
            re.DOTALL
        )
        match = call_pattern.search(js)
        
        target = None
        if match:
            target = ResolverAnalyzer._eval_hex_expression(match.group(1))
        
        # Find the arithmetic expression in IIFE body
        # Pattern: const _0xVAR = <big_expression>; if (...)
        body_pattern = re.compile(
            r"\(function[^{]*\{([^}]+)\}\s*\(_0x",
            re.DOTALL
        )
        body_match = body_pattern.search(js)
        
        big_expr = None
        if body_match:
            body = body_match.group(1)
            # Extract the calculation expression
            expr_pattern = re.compile(
                r"const\s+_0x[a-fA-F0-9_]+\s*=\s*(.+?);\s*if",
                re.DOTALL
            )
            expr_match = expr_pattern.search(body)
            if expr_match:
                big_expr = expr_match.group(1).strip()
        
        return target, big_expr


class RotationSolver:
    """Determines correct rotation for string tables."""
    
    @staticmethod
    def find_rotation(
        table: List[str],
        offset: int,
        big_expr: Optional[str],
        target: Optional[int]
    ) -> Tuple[List[str], int]:
        """
        Find the rotation that makes big_expr === target.
        
        Returns:
            (rotated_table, rotations_applied)
        """
        n = len(table)
        
        # If we have both expression and target, try exact match
        if big_expr and target is not None:
            rotated = table.copy()
            
            for rotation in range(n):
                value = RotationSolver._evaluate_expression(big_expr, rotated, offset)
                
                if value is not None and abs(value - target) < 1e-6:
                    return rotated, rotation
                
                # Rotate
                rotated.append(rotated.pop(0))
        
        # No match found, return original
        return table.copy(), 0
    
    @staticmethod
    def _evaluate_expression(expr: str, table: List[str], offset: int) -> Optional[float]:
        """
        Evaluate the IIFE arithmetic expression with current table rotation.
        
        The expression contains patterns like:
        - parseInt(_0xFUNC(0xHEX))
        - _0xFUNC(0xHEX)
        - 0xHEX literals
        - arithmetic operators
        """
        if not expr:
            return None
        
        # Find all hex indices used in the expression
        hex_pattern = r'0x[0-9a-fA-F]+'
        hex_tokens = list(set(re.findall(hex_pattern, expr)))
        
        # Resolve each hex to its table string, then to a number
        substitutions = {}
        for hex_val in hex_tokens:
            # Get string from table
            string_val = RotationSolver._resolve_index(table, offset, hex_val)
            # Parse as integer (JavaScript parseInt behavior)
            int_val = RotationSolver._parse_int_like(string_val)
            substitutions[hex_val] = int_val
        
        # Replace patterns in expression
        expr_sub = expr
        
        # Replace parseInt(_0xFUNC(0xHEX)) with the number
        expr_sub = re.sub(
            r'parseInt\s*\(\s*_0x[a-fA-F0-9_]+\s*\(\s*(0x[0-9a-fA-F]+)\s*\)\s*\)',
            lambda m: str(substitutions.get(m.group(1), 0)),
            expr_sub
        )
        
        # Replace _0xFUNC(0xHEX) with the number
        expr_sub = re.sub(
            r'_0x[a-fA-F0-9_]+\s*\(\s*(0x[0-9a-fA-F]+)\s*\)',
            lambda m: str(substitutions.get(m.group(1), 0)),
            expr_sub
        )
        
        # Replace standalone hex literals with decimals
        expr_sub = re.sub(
            r'0x[0-9a-fA-F]+',
            lambda m: str(int(m.group(0), 16)),
            expr_sub
        )
        
        # Evaluate
        try:
            return float(eval(expr_sub, {"__builtins__": {}}))
        except:
            return None
    
    @staticmethod
    def _resolve_index(table: List[str], offset: int, hex_index: str) -> Optional[str]:
        """Resolve hex index to table string."""
        dec = int(hex_index, 16)
        idx = dec - offset
        if 0 <= idx < len(table):
            return table[idx]
        return None
    
    @staticmethod
    def _parse_int_like(s: Optional[str]) -> int:
        """Mimic JavaScript parseInt behavior."""
        if not s:
            return 0
        match = re.match(r'[-+]?\d+', s)
        if match:
            try:
                return int(match.group(0))
            except:
                return 0
        return 0


class CodeTransformer:
    """Transforms obfuscated code to readable form."""
    
    @staticmethod
    def replace_resolver_calls(
        js: str,
        table: List[str],
        offset: int
    ) -> Tuple[str, Dict[str, Optional[str]]]:
        """
        Replace all resolver calls with string literals.
        Handles both _0x and *0x prefixes.
        """
        mapping: Dict[str, Optional[str]] = {}
        
        def replacement(match):
            hex_val = match.group(1)
            
            dec = int(hex_val, 16)
            idx = dec - offset
            
            if 0 <= idx < len(table):
                literal = table[idx]
            else:
                literal = None
            
            mapping[hex_val] = literal
            
            if literal is None:
                return match.group(0)
            
            return CodeTransformer._create_js_string_literal(literal)
        
        # Match resolver calls with EITHER _0x or *0x prefix
        # Pattern: (_0x...|*0x...)(0xHEX)
        pattern = re.compile(r'(?:_0x|[*]0x)[a-fA-F0-9_]+\s*\(\s*(0x[0-9a-fA-F]+)\s*\)')
        
        return pattern.sub(replacement, js), mapping
    
    @staticmethod
    def _create_js_string_literal(s: str) -> str:
        """Create properly escaped JavaScript string."""
        has_single = "'" in s
        has_double = '"' in s
        
        if has_single and not has_double:
            escaped = s.replace('\\', '\\\\').replace('"', '\\"')
            escaped = escaped.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return f'"{escaped}"'
        else:
            escaped = s.replace('\\', '\\\\').replace("'", "\\'")
            escaped = escaped.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return f"'{escaped}'"
    
    @staticmethod
    def collapse_concatenations(js: str) -> str:
        """Collapse adjacent string concatenations."""
        for _ in range(100):
            original = js
            
            # Single quotes
            js = re.sub(
                r"'([^'\\]*(?:\\.[^'\\]*)*)'\s*\+\s*'([^'\\]*(?:\\.[^'\\]*)*)'",
                lambda m: f"'{m.group(1)}{m.group(2)}'",
                js
            )
            
            # Double quotes
            js = re.sub(
                r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*\+\s*"([^"\\]*(?:\\.[^"\\]*)*)"',
                lambda m: f'"{m.group(1)}{m.group(2)}"',
                js
            )
            
            if js == original:
                break
        
        return js


class Deobfuscator:
    """Main deobfuscator."""
    
    def __init__(self):
        pass
    
    def deobfuscate(
        self,
        js: str,
        repeat: bool = False,
        max_iterations: int = 6,
        collapse_concat: bool = False
    ) -> DeobfuscationResult:
        """Perform deobfuscation."""
        # Fix syntax first
        js, syntax_fixes = SyntaxFixer.fix_syntax(js)
        
        initial_calls = self._count_resolver_calls(js)
        overall_mapping: Dict[str, Optional[str]] = {}
        current_js = js
        passes = 0
        
        if repeat:
            for _ in range(max(1, max_iterations)):
                passes += 1
                new_js, mapping, applied = self._single_pass(current_js)
                
                if not applied:
                    break
                
                overall_mapping.update(mapping)
                current_js = new_js
                
                if self._count_resolver_calls(current_js) == 0:
                    break
        else:
            passes = 1
            new_js, mapping, applied = self._single_pass(current_js)
            if applied:
                overall_mapping.update(mapping)
                current_js = new_js
        
        if collapse_concat:
            current_js = CodeTransformer.collapse_concatenations(current_js)
        
        final_calls = self._count_resolver_calls(current_js)
        
        return DeobfuscationResult(
            code=current_js,
            mapping=overall_mapping,
            resolver_calls_before=initial_calls,
            resolver_calls_after=final_calls,
            passes_completed=passes,
            syntax_fixes_applied=syntax_fixes
        )
    
    def _single_pass(self, js: str) -> Tuple[str, Dict[str, Optional[str]], bool]:
        """Perform a single deobfuscation pass."""
        # Extract string table
        arr_name, table = StringTableExtractor.find_table(js)
        if table is None:
            return js, {}, False
        
        print(f"  Found string table with {len(table)} entries")
        
        # Find offset
        offset = ResolverAnalyzer.find_offset(js)
        if offset is None:
            print("  Warning: Could not find offset, using 0")
            offset = 0
        else:
            print(f"  Calculated offset: {offset} (0x{offset:x})")
        
        # Find rotation
        target, big_expr = ResolverAnalyzer.extract_iife_info(js)
        if target is not None:
            print(f"  IIFE target value: {target}")
        
        final_table, rotations = RotationSolver.find_rotation(
            table, offset, big_expr, target
        )
        print(f"  Applied {rotations} rotation(s)")
        
        # Transform code
        deob_js, mapping = CodeTransformer.replace_resolver_calls(
            js, final_table, offset
        )
        
        return deob_js, mapping, True
    
    @staticmethod
    def _count_resolver_calls(js: str) -> int:
        """Count remaining resolver calls."""
        return len(re.findall(
            r'(?:_0x|[*]0x)[a-fA-F0-9_]+\s*\(\s*0x[0-9a-fA-F]+\s*\)',
            js
        ))


def main():
    parser = argparse.ArgumentParser(
        description="JavaScript deobfuscator for _0x-style string table obfuscation"
    )
    
    parser.add_argument("input", help="Path to obfuscated JavaScript file")
    parser.add_argument("--out", "-o", help="Output file path")
    parser.add_argument("--repeat", action="store_true", help="Run multiple passes")
    parser.add_argument("--max-iter", type=int, default=6, help="Max iterations")
    parser.add_argument("--collapse-concat", action="store_true", help="Collapse string concatenations")
    parser.add_argument("--mapping", action="store_true", help="Print mapping table")
    parser.add_argument("--preview", action="store_true", help="Print preview only")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1
    
    # Read input
    with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
        js_code = f.read()
    
    print(f"Processing: {args.input}")
    print()
    
    # Deobfuscate
    deobfuscator = Deobfuscator()
    result = deobfuscator.deobfuscate(
        js_code,
        repeat=args.repeat,
        max_iterations=args.max_iter,
        collapse_concat=args.collapse_concat
    )
    
    # Print results
    print()
    print("=" * 60)
    print(f"Completed {result.passes_completed} pass(es)")
    print(f"Syntax fixes: {result.syntax_fixes_applied}")
    print(f"Resolver calls: {result.resolver_calls_before} → {result.resolver_calls_after}")
    print(f"Unique mappings: {len(result.mapping)}")
    print("=" * 60)
    
    # Show mapping if requested
    if args.mapping:
        print("\nMapping Table:")
        print("-" * 60)
        for hex_val in sorted(result.mapping.keys(), key=lambda x: int(x, 16)):
            literal = result.mapping[hex_val]
            print(f"  {hex_val} → {repr(literal)}")
        print("-" * 60)
    
    # Preview or save
    if args.preview:
        print("\nPreview (first 2000 characters):")
        print("-" * 60)
        print(result.code[:2000])
        if len(result.code) > 2000:
            print("\n... (truncated) ...")
        print("-" * 60)
    else:
        output_path = args.out if args.out else args.input + ".deobf.js"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.code)
        print(f"\nOutput written to: {output_path}")
    
    print("\nWARNING: Review deobfuscated code before execution!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
