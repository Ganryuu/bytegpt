#!/bin/bash
# Compile every fig*.tex in this folder to PDF, PNG and SVG.
# Uses the tikz-build Docker image (see ../Dockerfile.tikz):
#   docker build -t tikz-build -f ../Dockerfile.tikz ..
cd "$(dirname "$0")"
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD":/work tikz-build sh -c '
  for f in fig*.tex; do
    b="${f%.tex}"
    pdflatex -interaction=nonstopmode -halt-on-error "$f" > "$b.build.log" 2>&1 \
      || { echo "FAILED: $f"; grep -A6 "^!" "$b.build.log"; continue; }
    pdftocairo -png -r 240 -singlefile "$b.pdf" "$b"
    pdftocairo -svg "$b.pdf" "$b.svg"
    rm -f "$b.aux" "$b.log" "$b.build.log"
    echo "ok: $b"
  done'
