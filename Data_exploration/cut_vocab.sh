#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat vocab_pos.txt | sed "s/^\s\+//g" | sort -rn | grep -v "^[1234]\s" > vocab_cut_pos.txt
