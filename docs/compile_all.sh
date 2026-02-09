#!/bin/bash
# Complete compilation script for MHED-TOE documentation

echo "========================================="
echo "MHED-TOE: COMPLETE DOCUMENTATION BUILD"
echo "========================================="

# Create necessary directories
mkdir -p ../figures
mkdir -p arxiv_submission

echo ""
echo "1. GENERATING FIGURES..."
echo "========================="
cd ..
if command -v python3 &> /dev/null; then
    python3 examples/generate_figures.py
else
    echo "Python3 not found, skipping figure generation"
fi
cd docs

echo ""
echo "2. COMPILING MAIN PAPER..."
echo "============================"
make paper

echo ""
echo "3. COMPILING SUPPLEMENTARY MATERIAL..."
echo "========================================"
make supplementary

echo ""
echo "4. CREATING ARXIV SUBMISSION PACKAGE..."
echo "========================================"
make arxiv

echo ""
echo "5. GENERATING DOCUMENTATION SUMMARY..."
echo "======================================="
echo "Paper Statistics:" > summary.txt
echo "================" >> summary.txt
echo "Main paper: $(detex MHED_TOE_PAPER.tex | wc -w) words" >> summary.txt
echo "Supplementary: $(detex supplementary.tex | wc -w) words" >> summary.txt
echo "" >> summary.txt
echo "Figures:" >> summary.txt
echo "========" >> summary.txt
ls -la ../figures/*.png | wc -l | xargs echo "Number of figures:" >> summary.txt
echo "" >> summary.txt
echo "File sizes:" >> summary.txt
echo "===========" >> summary.txt
ls -lh MHED_TOE_PAPER.pdf supplementary.pdf >> summary.txt

cat summary.txt

echo ""
echo "6. VALIDATING COMPILATION..."
echo "============================="
if [ -f "MHED_TOE_PAPER.pdf" ] && [ -f "supplementary.pdf" ]; then
    echo "✓ SUCCESS: All documents compiled"
    echo "✓ Main paper: MHED_TOE_PAPER.pdf"
    echo "✓ Supplementary: supplementary.pdf"
    echo "✓ arXiv package: arxiv_submission/mhed_toe_arxiv.tar.gz"
else
    echo "✗ ERROR: Compilation failed"
    exit 1
fi

echo ""
echo "========================================="
echo "BUILD COMPLETE - READY FOR SUBMISSION"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Upload to arXiv:"
echo "   https://arxiv.org/submit"
echo "2. Upload package: arxiv_submission/mhed_toe_arxiv.tar.gz"
echo "3. Categories: hep-th (primary), gr-qc, q-bio.NC, quant-ph"
echo "4. Announce on Twitter: @MHED_TOE"
echo ""
echo "Code repository: https://github.com/atomadic/MHED-TOE"
echo "Contact: atomadic@proton.me"
echo ""
