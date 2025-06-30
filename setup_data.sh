#!/bin/bash

# Script to create directory, download, unzip and cleanup

set -e  # Exit on any error

# Configuration
DATA_DIR="desp/data"
DOWNLOAD_URL="https://figshare.com/ndownloader/files/46831312"  # Replace with actual URL
ZIP_FILENAME="downloaded-file.zip"

echo "Starting download and extraction process..."

# Create the data directory if it doesn't exist
echo "Creating directory: $DATA_DIR"
mkdir -p "$DATA_DIR"

# Change to the data directory
cd "$DATA_DIR"

# Download the file using wget
echo "Downloading file from: $DOWNLOAD_URL"
wget -O "$ZIP_FILENAME" "$DOWNLOAD_URL"

# Check if download was successful
if [ ! -f "$ZIP_FILENAME" ]; then
    echo "Error: Download failed. File $ZIP_FILENAME not found."
    exit 1
fi

echo "Download completed successfully."

# Unzip the file
echo "Extracting $ZIP_FILENAME..."
unzip "$ZIP_FILENAME"

# Check if unzip was successful
if [ $? -eq 0 ]; then
    echo "Extraction completed successfully."
    
    # Delete the zip file
    echo "Cleaning up: removing $ZIP_FILENAME"
    rm "$ZIP_FILENAME"
    echo "Cleanup completed."
else
    echo "Error: Extraction failed."
    exit 1
fi

echo "Process completed successfully!"
echo "Files extracted to: $(pwd)"
ls -la