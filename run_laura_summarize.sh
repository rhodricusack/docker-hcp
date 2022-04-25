#!/bin/bash
python laura_summarize.py
aws s3 cp hcp* s3://neurana-imaging/rhodricusack/laura_summarize/
