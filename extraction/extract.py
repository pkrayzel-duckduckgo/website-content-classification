import os
import sys
import trafilatura


def extract_paragraphs_from_html(html_content):
    try:
        paragraphs = trafilatura.extract(html_content, output_format='txt', favor_precision=True)
        if paragraphs:
            return paragraphs.split('\n\n')
        else:
            return []
    except Exception as e:
        print(f"Error extracting paragraphs: {e}")
        return []


def process_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        if os.path.isfile(input_path):
            with open(input_path, 'r', encoding='utf-8') as file:
                html_content = file.read()

            paragraphs = extract_paragraphs_from_html(html_content)
            output_path = os.path.join(output_dir, filename + "_paragraphs.txt")

            with open(output_path, 'w', encoding='utf-8') as outfile:
                outfile.write('\n\n'.join(paragraphs))

            print(f"Processed file: {input_path}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 script.py <input_directory> <output_directory>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    process_files(input_dir, output_dir)


if __name__ == "__main__":
    main()
