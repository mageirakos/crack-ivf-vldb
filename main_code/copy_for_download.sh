#!/bin/bash

SOURCE_DIR="./results"
DUMP_DIR="./rule_1only_revision_experiments"

# Ensure dump directory exists
mkdir -p "$DUMP_DIR"

# Find all files and directories containing "20250601" in their name
find "$SOURCE_DIR" \( -type f -o -type d \) -name '*20250601_REF_RULE1_ONLY_REVISION_final*' | while read -r path; do
    # Compute relative path from SOURCE_DIR
    rel_path="${path#$SOURCE_DIR/}"
    
    # Target path in dump directory
    target_path="$DUMP_DIR/$rel_path"
    
    # Ensure parent directory exists
    mkdir -p "$(dirname "$target_path")"
    
    # Copy file or directory
    if [ -f "$path" ]; then
        cp "$path" "$target_path"
        echo "Copied file: $path → $target_path"
    elif [ -d "$path" ]; then
        cp -r "$path" "$target_path"
        echo "Copied directory: $path → $target_path"
    fi
done
