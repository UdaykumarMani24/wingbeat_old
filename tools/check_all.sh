#!/bin/sh
# Every check, in the order the audio flows. Any FAIL means do not ship.
set -e
d=$(dirname "$0")
fail=0
for f in check_resample check_features check_model check_chain check_classify; do
  printf '\n======== %s\n' "$f"
  if node "$d/$f.js" | tail -4; then :; else fail=1; fi
done
printf '\n'
[ $fail -eq 0 ] && echo "all checks ran" || echo "a check errored"
