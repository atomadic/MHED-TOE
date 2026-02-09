# Makefile for compiling MHED-TOE documentation

.PHONY: all clean paper supplementary pdf figures

# Main targets
all: paper supplementary

# Compile main paper
paper: MHED_TOE_PAPER.pdf

MHED_TOE_PAPER.pdf: MHED_TOE_PAPER.tex MHED_TOE_PAPER.bbl
	pdflatex MHED_TOE_PAPER.tex
	bibtex MHED_TOE_PAPER
	pdflatex MHED_TOE_PAPER.tex
	pdflatex MHED_TOE_PAPER.tex

MHED_TOE_PAPER.bbl: MHED_TOE_PAPER.bib
	pdflatex MHED_TOE_PAPER.tex
	bibtex MHED_TOE_PAPER

# Compile supplementary material
supplementary: supplementary.pdf

supplementary.pdf: supplementary.tex
	pdflatex supplementary.tex
	pdflatex supplementary.tex

# Generate figures
figures:
	cd .. && python examples/generate_figures.py

# Create arXiv submission package
arxiv: paper supplementary
	mkdir -p arxiv_submission
	cp MHED_TOE_PAPER.tex arxiv_submission/
	cp MHED_TOE_PAPER.bbl arxiv_submission/
	cp supplementary.tex arxiv_submission/
	cp -r ../figures arxiv_submission/
	cp README.txt arxiv_submission/
	cd arxiv_submission && tar -czf mhed_toe_arxiv.tar.gz *
	@echo "arXiv package created: arxiv_submission/mhed_toe_arxiv.tar.gz"

# Clean up
clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc *.lof *.lot
	rm -f MHED_TOE_PAPER.pdf supplementary.pdf
	rm -rf arxiv_submission

# Quick compile for testing
quick:
	pdflatex MHED_TOE_PAPER.tex

# View paper
view: paper
	open MHED_TOE_PAPER.pdf

# View supplementary
view-supp: supplementary
	open supplementary.pdf

# Check for errors
check:
	chktex MHED_TOE_PAPER.tex
	chktex supplementary.tex

# Word count
wordcount:
	detex MHED_TOE_PAPER.tex | wc -w
	@echo "words in main paper"
	detex supplementary.tex | wc -w
	@echo "words in supplementary"

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Compile everything"
	@echo "  paper        - Compile main paper"
	@echo "  supplementary - Compile supplementary material"
	@echo "  figures      - Generate all figures"
	@echo "  arxiv        - Create arXiv submission package"
	@echo "  clean        - Remove generated files"
	@echo "  view         - View main paper"
	@echo "  view-supp    - View supplementary"
	@echo "  check        - Check for LaTeX errors"
	@echo "  wordcount    - Count words"
	@echo "  help         - Show this help"
