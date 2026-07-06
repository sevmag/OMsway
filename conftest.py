"""Put the project root on ``sys.path`` so ``import omsway`` works in tests and
scripts without an installed package (run from the project root)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
