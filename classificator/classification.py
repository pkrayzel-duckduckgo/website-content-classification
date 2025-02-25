import torch
import trafilatura
from transformers import pipeline
import pandas as pd

def load_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Load the fine-tuned model
classifier = pipeline(
    "text-classification",
    model="../preparation/fine_tuned_distilbert_v2",
    tokenizer="../preparation/fine_tuned_distilbert_v2"  # Use local tokenizer
)

def classify_sentences(sentences, title, url):
    results = []
    device = torch.device("mps")  # Explicitly use MPS

    for sentence in sentences:
        if not sentence.strip():  # Skip empty sentences
            continue
        
        # Limit context size to avoid exceeding the 512-token max length
        context = f"Title: {title} URL: {url} Sentence: {sentence}"

        # Tokenize and truncate, moving input tensors to MPS
        inputs = classifier.tokenizer(context, truncation=True, max_length=512, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to MPS

        # Get classification result
        result = classifier.model(**inputs)
        predicted_label = result.logits.argmax().item()  # Get the predicted label

        results.append((title, url, sentence, predicted_label))
    return results

def main():
    source_files = [
        {
            "title": "Miss Scarlet and The Duke - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Miss_Scarlet_and_The_Duke",
            "output_csv": "wikipedia_miss_scarlet.csv",
            "file_path": "../extracted_paragraphs_to_review/https_www.homedepot.com_b_Altair_Frameless_N-5yc1vZm1jZ1z132br_js_enabled.html_paragraphs.txt",
        },
        {
            "title": "050644zf/ArknightsStoryTextReader",
            "url": "https://github.com/050644zf/ArknightsStoryTextReader",
            "output_csv": "github_arknight.csv",
            "file_path": "../extracted_paragraphs_to_review/https_github.com_050644zf_ArknightsStoryTextReader_js_enabled.html_paragraphs.txt",
        },
        {
            "title": "Suspect identified after 7 San Antonio Police officers injured in late-night shootout",
            "url": "https://news4sanantonio.com/news/local/four-san-antonio-police-officers-shot-in-stone-oak-shooting",
            "output_csv": "news4sanantonio.csv",
            "file_path": "../extracted_paragraphs_to_review/https_news4sanantonio.com_news_local_four-san-antonio-police-officers-shot-in-stone-oak-shooting_js_enabled.html_paragraphs.txt",
        },

        {
            "title": "Ex-teacher who fed students semen-laced cupcakes sentenced to 41 years",
            "url": "https://nypost.com/2022/02/20/ex-teacher-who-fed-students-semen-laced-cupcakes-sentenced-to-41-years/",
            "output_csv": "nypost_techer.csv",
            "file_path": "../extracted_paragraphs_to_review/https_nypost.com_2022_02_20_ex-teacher-who-fed-students-semen-laced-cupcakes-sentenced-to-41-years__js_enabled.html_paragraphs.txt",
        },
    ]
        
    for item in source_files:
        title = item["title"]
        url = item["url"]
        output_csv = item["output_csv"]
        file_with_sentences = load_html(item["file_path"])
        sentences = file_with_sentences.split("\n")

        classification_results = classify_sentences(sentences, title, url)
        
        df = pd.DataFrame(classification_results, columns=['title', 'url', 'sentence', 'label'])
        df.to_csv(output_csv, sep='|', index=False)

        print("\n==== RESULTS ====")
        print(f"Relevant content: {len(df[df['label'] == 1])}")
        print(f"Irrelevant content: {len(df[df['label'] == 0])}")
        
        print(f"\n✅ Classified sentences saved to {output_csv}")

if __name__ == "__main__":
    main()
