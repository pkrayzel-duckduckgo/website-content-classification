# Website content relevancy classification

The [ML model](https://github.com/pkrayzel/content-relevancy-csr-classification/blob/main/relevancy_classification_v2.tar.gz) from this project is meant to help with content relevancy classification for websites.

It is trained to classify sentences found on website as relevant/irrelevant to the website's title and URL.

Irrelevant are generic sentences like cookie banners, requires javascript, paywall messages etc. See [the list of these from the training dataset](preparation/generate_labeled_dataset.py#L124).

Relevant are sentences that are somehow linked to the title and URL of the webpage. See [examples from the training dataset](preparation/labeled_dataset.csv#L2).

## How was the model trained?

1. Pretrained text-classification model `distilbert-base-uncased` from hugging face was used as a base. 
2. The model was then trained on labeled dataset with 1895 rows generated with the help of LLM (see [the data generation script](preparation/generate_labeled_dataset.py) and [the dataset](preparation/labeled_dataset.csv)) using [this training script](preparation/train.py).
3. Trained model (add 2) was used to classify content from 15 websites. The results were stored in a file and manually verified and relabeled (see [the manually verified dataset](preparation/manual_dataset.csv)).
4. The trained model (add 2) was retrained with this new manually verified dataset with 6219 rows using [this script](preparation/retrain_model.py).

**Training / validating details**

The datasets were split into:
- Train: 0.9 * 0.8 = 0.72 => 72%
- Validation: 0.9 * 0.2 = 0.18 => 18%
- Test: 0.1 => 10%

**Evaluation**

**Note**: The intermediary model had almost perfect results, likely because the dataset was a bit artificial and predictable. Hence the need to retrain it on production like data.

```
eval_accuracy: 1.0
eval_precision: 1.0
eval_recall: 1.0
eval_f1: 1.0
```

The final model
```
eval_accuracy: 0.9756
eval_precision: 0.9771
eval_recall: 0.9697
eval_f1: 0.9734
```

## Training model - optimised versions

```
python train_model.py --model minilm
python retrain_model.py --model minilm

```

```
python train_model.py --model distilbert
python retrain_model.py --model distilbert
```