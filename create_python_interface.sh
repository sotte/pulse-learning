#!/bin/sh

set -e

echo "\n======================================================="
echo "Creating python interface\n"
python setup.py build_ext --inplace


echo "\n======================================================="
echo "Testing interface\n"
python test_pypulse.py
