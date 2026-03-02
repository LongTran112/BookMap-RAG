# EBooksSorter

Non-destructive technical PDF categorizer and index generator.

## What it does

- Scans a source folder recursively for `.pdf` files.
- Infers one category per file from:
  - filename,
  - PDF metadata (title/subject/keywords/author),
  - extracted text from the first N pages.
- Generates:
  - `output/books_by_category.md`
  - `output/books_by_category.csv`

No files are moved, renamed, or deleted.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 index_books.py \
  --source "/Users/longtran/Documents/E-Books" \
  --output-dir "./output" \
  --max-pages 8
```

## Output schema

CSV columns:

- `category`
- `confidence`
- `title`
- `filename`
- `absolute_path`
- `matched_keywords`

## Tuning categories

Edit `index_books.py`:

- `KEYWORDS`: update or add keywords under each category.
- `SOURCE_WEIGHTS`: adjust signal strength:
  - title/metadata > filename > body.
- `CATEGORY_ORDER`: tie-break priority for equal scores.

After any tuning, rerun the command to regenerate deterministic outputs.
