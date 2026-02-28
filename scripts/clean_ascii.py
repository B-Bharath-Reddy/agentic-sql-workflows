import os
import re

file_path = 'pdf_layout_summary.md'

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace smart quotes and typographic dashes
text = re.sub(r'[\u2018\u2019]', "'", text)
text = re.sub(r'[\u201c\u201d]', '"', text)
text = re.sub(r'[\u2013\u2014]', '-', text)
text = text.replace('\u2026', '...')
text = text.replace('\u2022', '-')
text = text.replace('\u2192', '->')

# Strip everything else that isn't ASCII
text = ''.join(char for char in text if ord(char) < 128)

with open(file_path, 'w', encoding='ascii') as f:
    f.write(text)

print("Markdown file successfully sanitized to US-ASCII.")
