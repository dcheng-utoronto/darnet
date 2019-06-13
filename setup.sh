#!/bin/bash

python setup_paths.py
python setup_vaihingen_bing_splits.py vaihingen
python setup_vaihingen_bing_splits.py bing

echo "Setup complete"