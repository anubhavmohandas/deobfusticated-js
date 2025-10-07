#!/usr/bin/env python3
"""
Enhanced JavaScript Deobfuscator with Advanced Features

Features:
1. _0x-style string table deobfuscation
2. Duplicate pattern removal (junk string removal)
3. Base64 and encoding detection/decoding
4. Code beautification and cleanup
5. Multi-pass iterative improvements
"""

from __future__ import annotations
import re
import sys
import os
import argparse
import base64
import json
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from collections import Counter


@dataclass
class DeobfuscationResult:
    code: str
    mapping: Dict[str, Optional[str]]
    resolver_calls_before: int
    resolver_calls_after: int
    passes_completed: int
    syntax_fixes_applied: int
    junk_patterns_removed: int
    encodings_decoded: int
    beautification_applied: bool


class SyntaxFixer:
    """Enhanced syntax fixer with multiple aggressive passes."""
    
    @staticmethod
    def fix_syntax(js: str) -> Tuple[str, int]:
        """Fix asterisk syntax errors with multiple passes."""
        original = js
        
        for iteration in range(10):
            before = js
            js = re.sub(r'\*0x', '_0x', js)
            js = re.sub(r'\b(const|let|var|function)\s+\*', r'\1 _', js)
            if js == before:
                break
        
        fixes = original.count('*0x')
        return js, fixes


class JunkPatternDetector:
    """Detects and removes junk/duplicate patterns from obfuscated code."""
    
    @staticmethod
    def find_junk_patterns(js: str, min_length: int = 3, min_occurrences: int = 10) -> List[str]:
        """
        Find repetitive junk patterns in strings.
        Returns patterns that appear suspiciously often.
        """
        # Extract all string literals
        strings = re.findall(r"'([^'\\]*(?:\\.[^'\\]*)*)'|\"([^\"\\]*(?:\\.[^\"\\]*)*)\"", js)
        all_strings = [s1 or s2 for s1, s2 in strings]
        
        # Find n-grams that appear frequently
        junk_candidates: Counter = Counter()
        
        for string in all_strings:
            # Extract substrings of various lengths
            for length in range(min_length, min(len(string) + 1, 8)):
                for i in range(len(string) - length + 1):
                    ngram = string[i:i+length]
                    # Only consider patterns with mix of letters
                    if re.match(r'^[A-Za-z]{3,}$', ngram):
                        junk_candidates[ngram] += 1
        
        # Filter to high-frequency patterns
        junk_patterns = [
            pattern for pattern, count in junk_candidates.items()
            if count >= min_occurrences
        ]
        
        # Sort by length (longer first) to avoid removing substrings
        junk_patterns.sort(key=len, reverse=True)
        
        return junk_patterns
    
    @staticmethod
    def remove_junk_patterns(js: str, patterns: List[str]) -> Tuple[str, int]:
        """
        Remove junk patterns from strings.
        Returns cleaned code and number of removals.
        """
        if not patterns:
            return js, 0
        
        removals = 0
        
        def clean_string(match):
            nonlocal removals
            quote = match.group(0)[0]
            content = match.group(1) or match.group(2)
            
            original_content = content
            for pattern in patterns:
                if pattern in content:
                    content = content.replace(pattern, '')
                    
            if content != original_content:
                removals += 1
            
            # Re-escape the cleaned content
            escaped = content.replace('\\', '\\\\').replace(quote, f'\\{quote}')
            return f'{quote}{escaped}{quote}'
        
        # Process all strings
        js = re.sub(
            r"'([^'\\]*(?:\\.[^'\\]*)*)'|\"([^\"\\]*(?:\\.[^\"\\]*)*)\"",
            clean_string,
            js
        )
        
        return js, removals


class EncodingDetector:
    """Detects and decodes various encoding schemes."""
    
    @staticmethod
    def detect_and_decode_base64(js: str) -> Tuple[str, int]:
        """
        Find and decode Base64 encoded strings.
        """
        decoded_count = 0
        
        def try_decode(match):
            nonlocal decoded_count
            quote = match.group(0)[0]
            content = match.group(1) or match.group(2)
            
            # Check if it looks like base64 (length divisible by 4, valid chars)
            if len(content) > 20 and len(content) % 4 == 0:
                if re.match(r'^[A-Za-z0-9+/=]+$', content):
                    try:
                        decoded_bytes = base64.b64decode(content)
                        decoded_str = decoded_bytes.decode('utf-8', errors='ignore')
                        
                        # Only replace if decoded string looks reasonable
                        if decoded_str.isprintable() and len(decoded_str) > 0:
                            decoded_count += 1
                            escaped = decoded_str.replace('\\', '\\\\').replace(quote, f'\\{quote}')
                            return f'{quote}{escaped}{quote}'
                    except:
                        pass
            
            return match.group(0)
        
        js = re.sub(
            r"'([^'\\]*(?:\\.[^'\\]*)*)'|\"([^\"\\]*(?:\\.[^\"\\]*)*)\"",
            try_decode,
            js
        )
        
        return js, decoded_count
    
    @staticmethod
    def decode_hex_sequences(js: str) -> Tuple[str, int]:
        """
        Decode hex escape sequences \\x## in strings.
        """
        decoded_count = 0
        
        def decode_hex(match):
            nonlocal decoded_count
            quote = match.group(0)[0]
            content = match.group(1) or match.group(2)
            
            if '\\x' in content:
                original = content
                content = re.sub(
                    r'\\x([0-9a-fA-F]{2})',
                    lambda m: chr(int(m.group(1), 16)),
                    content
                )
                if content != original:
                    decoded_count += 1
                
                escaped = content.replace('\\', '\\\\').replace(quote, f'\\{quote}')
                return f'{quote}{escaped}{quote}'
            
            return match.group(0)
        
        js = re.sub(
            r"'([^'\\]*(?:\\.[^'\\]*)*)'|\"([^\"\\]*(?:\\.[^\"\\]*)*)\"",
            decode_hex,
            js
        )
        
        return js, decoded_count
    
    @staticmethod
    def decode_unicode_sequences(js: str) -> Tuple[str, int]:
        """
        Decode unicode escape sequences \\u#### in strings.
        """
        decoded_count = 0
        
        def decode_unicode(match):
            nonlocal decoded_count
            quote = match.group(0)[0]
            content = match.group(1) or match.group(2)
            
            if '\\u' in content:
                original = content
                content = re.sub(
                    r'\\u([0-9a-fA-F]{4})',
                    lambda m: chr(int(m.group(1), 16)),
                    content
                )
                if content != original:
                    decoded_count += 1
                
                escaped = content.replace('\\', '\\\\').replace(quote, f'\\{quote}')
                return f'{quote}{escaped}{quote}'
            
            return match.group(0)
        
        js = re.sub(
            r"'([^'\\]*(?:\\.[^'\\]*)*)'|\"([^\"\\]*(?:\\.[^\"\\]*)*)\"",
            decode_unicode,
            js
        )
        
        return js, decoded_count


class StringTableExtractor:
    """Extracts string tables from obfuscated code."""
    
    @staticmethod
    def find_table(js: str) -> Tuple[Optional[str], Optional[List[str]]]:
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
        s = re.sub(r'\\x([0-9a-fA-F]{2})', 
                   lambda m: chr(int(m.group(1), 16)), s)
        s = re.sub(r'\\u([0-9a-fA-F]{4})', 
                   lambda m: chr(int(m.group(1), 16)), s)
        
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
        patterns = [
            r"_0x[0-9a-fA-F_]+\s*=\s*_0x[0-9a-fA-F_]+\s*-\s*\(\s*([^)]+)\)",
            r"\[_0x[0-9a-fA-F_]+\]\s*;\s*return[^;]*-\s*\(\s*([^)]+)\)",
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
        try:
            expr_py = re.sub(
                r'0x[0-9a-fA-F]+',
                lambda m: str(int(m.group(0), 16)),
                expr
            )
            return int(eval(expr_py, {"__builtins__": {}}))
        except:
            return None
    
    @staticmethod
    def extract_iife_info(js: str) -> Tuple[Optional[int], Optional[str]]:
        call_pattern = re.compile(
            r"\}\s*\(\s*_0x[a-fA-F0-9_]+\s*,\s*([^)]+)\)\s*\)",
            re.DOTALL
        )
        match = call_pattern.search(js)
        
        target = None
        if match:
            target = ResolverAnalyzer._eval_hex_expression(match.group(1))
        
        body_pattern = re.compile(
            r"\(function[^{]*\{([^}]+)\}\s*\(_0x",
            re.DOTALL
        )
        body_match = body_pattern.search(js)
        
        big_expr = None
        if body_match:
            body = body_match.group(1)
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
        n = len(table)
        
        if big_expr and target is not None:
            rotated = table.copy()
            
            for rotation in range(n):
                value = RotationSolver._evaluate_expression(big_expr, rotated, offset)
                
                if value is not None and abs(value - target) < 1e-6:
                    return rotated, rotation
                
                rotated.append(rotated.pop(0))
        
        return table.copy(), 0
    
    @staticmethod
    def _evaluate_expression(expr: str, table: List[str], offset: int) -> Optional[float]:
        if not expr:
            return None
        
        hex_pattern = r'0x[0-9a-fA-F]+'
        hex_tokens = list(set(re.findall(hex_pattern, expr)))
        
        substitutions = {}
        for hex_val in hex_tokens:
            string_val = RotationSolver._resolve_index(table, offset, hex_val)
            int_val = RotationSolver._parse_int_like(string_val)
            substitutions[hex_val] = int_val
        
        expr_sub = expr
        
        expr_sub = re.sub(
            r'parseInt\s*\(\s*_0x[a-fA-F0-9_]+\s*\(\s*(0x[0-9a-fA-F]+)\s*\)\s*\)',
            lambda m: str(substitutions.get(m.group(1), 0)),
            expr_sub
        )
        
        expr_sub = re.sub(
            r'_0x[a-fA-F0-9_]+\s*\(\s*(0x[0-9a-fA-F]+)\s*\)',
            lambda m: str(substitutions.get(m.group(1), 0)),
            expr_sub
        )
        
        expr_sub = re.sub(
            r'0x[0-9a-fA-F]+',
            lambda m: str(int(m.group(0), 16)),
            expr_sub
        )
        
        try:
            return float(eval(expr_sub, {"__builtins__": {}}))
        except:
            return None
    
    @staticmethod
    def _resolve_index(table: List[str], offset: int, hex_index: str) -> Optional[str]:
        dec = int(hex_index, 16)
        idx = dec - offset
        if 0 <= idx < len(table):
            return table[idx]
        return None
    
    @staticmethod
    def _parse_int_like(s: Optional[str]) -> int:
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
        
        pattern = re.compile(r'(?:_0x|[*]0x)[a-fA-F0-9_]+\s*\(\s*(0x[0-9a-fA-F]+)\s*\)')
        
        return pattern.sub(replacement, js), mapping
    
    @staticmethod
    def _create_js_string_literal(s: str) -> str:
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
        for _ in range(100):
            original = js
            
            js = re.sub(
                r"'([^'\\]*(?:\\.[^'\\]*)*)'\s*\+\s*'([^'\\]*(?:\\.[^'\\]*)*)'",
                lambda m: f"'{m.group(1)}{m.group(2)}'",
                js
            )
            
            js = re.sub(
                r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*\+\s*"([^"\\]*(?:\\.[^"\\]*)*)"',
                lambda m: f'"{m.group(1)}{m.group(2)}"',
                js
            )
            
            if js == original:
                break
        
        return js


class CodeBeautifier:
    """Beautifies and formats JavaScript code."""
    
    @staticmethod
    def beautify(js: str) -> str:
        """
        Apply basic beautification to JavaScript code.
        """
        # Add newlines after semicolons (if not in strings)
        js = CodeBeautifier._add_newlines(js)
        
        # Indent code blocks
        js = CodeBeautifier._indent_code(js)
        
        # Add spaces around operators
        js = CodeBeautifier._add_operator_spaces(js)
        
        return js
    
    @staticmethod
    def _add_newlines(js: str) -> str:
        """Add newlines after statements."""
        # Simple approach: add newline after ; if not followed by whitespace or }
        js = re.sub(r';\s*(?=[^\s\}])', ';\n', js)
        
        # Add newlines after { and before }
        js = re.sub(r'\{\s*', '{\n', js)
        js = re.sub(r'\s*\}', '\n}', js)
        
        return js
    
    @staticmethod
    def _indent_code(js: str) -> str:
        """Add basic indentation."""
        lines = js.split('\n')
        indented_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            if not stripped:
                continue
            
            # Decrease indent for closing braces
            if stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
            
            # Add indentation
            indented_lines.append('  ' * indent_level + stripped)
            
            # Increase indent for opening braces
            if stripped.endswith('{'):
                indent_level += 1
            
        return '\n'.join(indented_lines)
    
    @staticmethod
    def _add_operator_spaces(js: str) -> str:
        """Add spaces around operators."""
        # Add spaces around =, +, -, *, /, etc.
        js = re.sub(r'([^\s=!<>])=([^\s=])', r'\1 = \2', js)
        js = re.sub(r'([^\s+])\+([^\s+])', r'\1 + \2', js)
        js = re.sub(r'([^\s-])-([^\s-])', r'\1 - \2', js)
        
        return js


class EnhancedDeobfuscator:
    """Main enhanced deobfuscator with all features."""
    
    def __init__(self):
        pass
    
    def deobfuscate(
        self,
        js: str,
        repeat: bool = True,
        max_iterations: int = 6,
        remove_junk: bool = True,
        decode_encodings: bool = True,
        beautify: bool = True,
        collapse_concat: bool = True
    ) -> DeobfuscationResult:
        """Perform enhanced deobfuscation."""
        print("🔧 Phase 1: Syntax Fixing")
        js, syntax_fixes = SyntaxFixer.fix_syntax(js)
        print(f"   Fixed {syntax_fixes} syntax errors")
        
        initial_calls = self._count_resolver_calls(js)
        overall_mapping: Dict[str, Optional[str]] = {}
        current_js = js
        passes = 0
        
        print("\n🔓 Phase 2: String Table Deobfuscation")
        if repeat:
            for i in range(max(1, max_iterations)):
                passes += 1
                print(f"   Pass {passes}...")
                new_js, mapping, applied = self._single_pass(current_js)
                
                if not applied:
                    break
                
                overall_mapping.update(mapping)
                current_js = new_js
                
                if self._count_resolver_calls(current_js) == 0:
                    print(f"   ✓ All resolver calls removed!")
                    break
        else:
            passes = 1
            new_js, mapping, applied = self._single_pass(current_js)
            if applied:
                overall_mapping.update(mapping)
                current_js = new_js
        
        junk_removed = 0
        if remove_junk:
            print("\n🧹 Phase 3: Junk Pattern Removal")
            junk_patterns = JunkPatternDetector.find_junk_patterns(current_js)
            if junk_patterns:
                print(f"   Found {len(junk_patterns)} junk patterns: {junk_patterns[:5]}")
                current_js, junk_removed = JunkPatternDetector.remove_junk_patterns(
                    current_js, junk_patterns
                )
                print(f"   Removed junk from {junk_removed} strings")
            else:
                print("   No junk patterns detected")
        
        encodings_decoded = 0
        if decode_encodings:
            print("\n🔐 Phase 4: Encoding Detection & Decoding")
            
            current_js, base64_count = EncodingDetector.detect_and_decode_base64(current_js)
            if base64_count > 0:
                print(f"   Decoded {base64_count} Base64 strings")
                encodings_decoded += base64_count
            
            current_js, hex_count = EncodingDetector.decode_hex_sequences(current_js)
            if hex_count > 0:
                print(f"   Decoded {hex_count} hex sequences")
                encodings_decoded += hex_count
            
            current_js, unicode_count = EncodingDetector.decode_unicode_sequences(current_js)
            if unicode_count > 0:
                print(f"   Decoded {unicode_count} unicode sequences")
                encodings_decoded += unicode_count
            
            if encodings_decoded == 0:
                print("   No additional encodings detected")
        
        if collapse_concat:
            print("\n📝 Phase 5: String Concatenation Collapse")
            before_len = len(current_js)
            current_js = CodeTransformer.collapse_concatenations(current_js)
            after_len = len(current_js)
            reduction = before_len - after_len
            print(f"   Reduced code size by {reduction} characters")
        
        beautified = False
        if beautify:
            print("\n✨ Phase 6: Code Beautification")
            current_js = CodeBeautifier.beautify(current_js)
            beautified = True
            print("   Code beautified and formatted")
        
        final_calls = self._count_resolver_calls(current_js)
        
        return DeobfuscationResult(
            code=current_js,
            mapping=overall_mapping,
            resolver_calls_before=initial_calls,
            resolver_calls_after=final_calls,
            passes_completed=passes,
            syntax_fixes_applied=syntax_fixes,
            junk_patterns_removed=junk_removed,
            encodings_decoded=encodings_decoded,
            beautification_applied=beautified
        )
    
    def _single_pass(self, js: str) -> Tuple[str, Dict[str, Optional[str]], bool]:
        """Perform a single deobfuscation pass."""
        arr_name, table = StringTableExtractor.find_table(js)
        if table is None:
            return js, {}, False
        
        print(f"     Found string table with {len(table)} entries")
        
        offset = ResolverAnalyzer.find_offset(js)
        if offset is None:
            offset = 0
        else:
            print(f"     Offset: {offset} (0x{offset:x})")
        
        target, big_expr = ResolverAnalyzer.extract_iife_info(js)
        
        final_table, rotations = RotationSolver.find_rotation(
            table, offset, big_expr, target
        )
        if rotations > 0:
            print(f"     Applied {rotations} rotation(s)")
        
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
        description="Enhanced JavaScript Deobfuscator with advanced pattern removal",
        epilog="""
Examples:
  %(prog)s obfuscated.js
  %(prog)s input.js --no-beautify --out clean.js
  %(prog)s input.js --preview --mapping

Features:
  • String table deobfuscation (_0x pattern)
  • Junk pattern removal (duplicate strings)
  • Base64 and encoding detection
  • Code beautification
  • Multiple iteration passes
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input", help="Path to obfuscated JavaScript file")
    parser.add_argument("--out", "-o", metavar="FILE", 
                       help="Output file path (default: INPUT.deobf.js)")
    parser.add_argument("--no-repeat", action="store_true", 
                       help="Run only one pass (default: multiple passes)")
    parser.add_argument("--max-iter", type=int, default=6, metavar="N",
                       help="Maximum iterations (default: 6)")
    parser.add_argument("--no-junk-removal", action="store_true",
                       help="Skip junk pattern removal")
    parser.add_argument("--no-decode", action="store_true",
                       help="Skip encoding detection/decoding")
    parser.add_argument("--no-beautify", action="store_true",
                       help="Skip code beautification")
    parser.add_argument("--no-collapse", action="store_true",
                       help="Skip string concatenation collapse")
    parser.add_argument("--mapping", action="store_true", 
                       help="Print hex to string mapping table")
    parser.add_argument("--preview", action="store_true", 
                       help="Print preview without saving")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"❌ Error: File not found: {args.input}", file=sys.stderr)
        return 1
    
    with open(args.input, 'r', encoding='utf-8', errors='ignore') as f:
        js_code = f.read()
    
    print(f"📂 Processing: {args.input}")
    print(f"📊 Original size: {len(js_code)} characters\n")
    
    deobfuscator = EnhancedDeobfuscator()
    result = deobfuscator.deobfuscate(
        js_code,
        repeat=not args.no_repeat,
        max_iterations=args.max_iter,
        remove_junk=not args.no_junk_removal,
        decode_encodings=not args.no_decode,
        beautify=not args.no_beautify,
        collapse_concat=not args.no_collapse
    )
    
    print("\n" + "=" * 70)
    print("📈 DEOBFUSCATION SUMMARY")
    print("=" * 70)
    print(f"✓ Passes completed: {result.passes_completed}")
    print(f"✓ Syntax fixes: {result.syntax_fixes_applied}")
    print(f"✓ Resolver calls: {result.resolver_calls_before} → {result.resolver_calls_after}")
    print(f"✓ Unique mappings: {len(result.mapping)}")
    print(f"✓ Junk patterns removed: {result.junk_patterns_removed}")
    print(f"✓ Encodings decoded: {result.encodings_decoded}")
    print(f"✓ Beautification: {'Applied' if result.beautification_applied else 'Skipped'}")
    print(f"✓ Final size: {len(result.code)} characters")
    
    size_reduction = len(js_code) - len(result.code)
    if size_reduction > 0:
        percentage = (size_reduction / len(js_code)) * 100
        print(f"✓ Size reduction: {size_reduction} chars ({percentage:.1f}%)")
    
    print("=" * 70)
    
    if args.mapping and result.mapping:
        print("\n📋 MAPPING TABLE")
        print("-" * 70)
        for hex_val in sorted(result.mapping.keys(), key=lambda x: int(x, 16)):
            literal = result.mapping[hex_val]
            display = repr(literal)[:60] + "..." if literal and len(repr(literal)) > 60 else repr(literal)
            print(f"  {hex_val} → {display}")
        print("-" * 70)
    
    if args.preview:
        print("\n👁️  PREVIEW (first 3000 characters)")
        print("-" * 70)
        print(result.code[:3000])
        if len(result.code) > 3000:
            print("\n... (truncated) ...")
        print("-" * 70)
    else:
        output_path = args.out if args.out else args.input + ".deobf.js"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result.code)
        print(f"\n💾 Output written to: {output_path}")
    
    print("\n⚠️  WARNING: Always review deobfuscated code before execution!")
    print("    Deobfuscation may not be 100% perfect for all obfuscation types.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
