#!/usr/bin/env bash
# Syntax validation baseline: compiles every tracked Python file.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

py_files=$(git ls-files '*.py')

for f in $py_files; do
  echo "py_compile: $f"
  python3 -m py_compile "$f"
done

echo "OK: syntax check passed for $(echo "$py_files" | wc -w | tr -d ' ') file(s)"
