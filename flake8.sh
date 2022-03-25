#!/bin/bash
# This command is actually defined in setup.cfg, but that's not
# obvious. It being defined there is how E266 is ignored. It could
# also be used to have rules ignored for specific files as well.
# Remember: no news is good news!
flake8 src
