# Article figures

Article: https://claude.ai/artifact/AamTLQSJGp6uYueALTKjtm

`figures/fig*.tex` are standalone TikZ files. Each one compiles by itself, so it
can be pasted into a new Overleaf project as `main.tex` (compiler: pdfLaTeX).

To rebuild locally (PDF, PNG and SVG next to each `.tex`):

    docker build -t tikz-build -f Dockerfile.tikz .   # once, about 1 GB
    figures/build.sh

| File | Shows |
| --- | --- |
| `fig1_three_ways.tex` | One sentence cut into bytes, BPE tokens and H-Net chunks (real cuts) |
| `fig2_hnet_architecture.tex` | The 1-stage H-Net from `configs/hnet_small.json` |
| `fig3_other_modalities.tex` | The plan: the same H-Net on text, images and robot data |
