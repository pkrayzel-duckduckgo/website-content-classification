import pandas as pd
import csv
import sys

csv.field_size_limit(sys.maxsize)

def validate_csv(file_path):
    """Validate CSV structure and label correctness, printing errors for each problematic row."""
    
    expected_columns = ["title", "url", "sentence", "label"]
    problematic_rows = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")  # Ensure proper delimiter
        header = next(reader, None)  # Read the header
        
        # Check if the header is correct
        if header != expected_columns:
            print(f"❌ Header mismatch! Expected: {expected_columns}, Found: {header}")
            return
        
        for line_num, row in enumerate(reader, start=2):  # Start from line 2 (after header)
            if len(row) != 4:
                problematic_rows.append(f"⚠️ Line {line_num}: Incorrect column count ({len(row)} instead of 4) → {row}")
                continue  # Skip further validation for this row

            title, url, sentence, label = row

            # Check for empty fields
            if not title.strip() or not url.strip() or not sentence.strip():
                problematic_rows.append(f"⚠️ Line {line_num}: Empty field(s) detected → {row}")
            
            # Check if label is a valid integer (0 or 1)
            try:
                label = int(label.strip())
                if label not in [0, 1]:
                    problematic_rows.append(f"⚠️ Line {line_num}: Invalid label (should be 0 or 1) → {label}")
            except ValueError:
                problematic_rows.append(f"⚠️ Line {line_num}: Non-integer label → {label}")

    # Print summary
    if problematic_rows:
        print("\n🚨 Found issues in the following lines:")
        for issue in problematic_rows:
            print(issue)
    else:
        print("✅ CSV file is clean and correctly formatted!")

# Usage
csv_file = "manual_dataset.csv"  # Replace with your actual file path
validate_csv(csv_file)
