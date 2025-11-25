#!/bin/bash

for i in {1..8}
do
    uv run ./scripts/Culture_RS_multi_core.py & 
done

wait
echo "All instances have completed."