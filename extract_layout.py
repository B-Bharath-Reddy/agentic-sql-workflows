import os
import glob
import pymupdf4llm
from pathlib import Path

pdf_files = sorted(glob.glob("*.pdf"))

output_file = "pdf_layout_summary.md"

print(f"Starting advanced PDF extraction using pymupdf4llm. Found {len(pdf_files)} PDFs.")

with open(output_file, "w", encoding="utf-8") as out:
    for pdf_file in pdf_files:
        print(f"Reading {pdf_file}...")
        try:
            # We will read the whole document to ensure we capture all tables
            # pymupdf4llm extracts it natively into Markdown
            md_text = pymupdf4llm.to_markdown(pdf_file)
            
            out.write(f"\n# {'='*50}\n# {pdf_file}\n# {'='*50}\n\n")
            out.write(md_text)
            out.write("\n\n---\n")
            
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")
            out.write(f"\nError reading {pdf_file}: {e}\n")

print(f"Summary written to {output_file}")
