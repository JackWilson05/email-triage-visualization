# Email Triage Needs More Than Text

This project compares text-only, email-context-only, and combined spam-detection models using the SpamAssassin public email corpus. It is a visual case study for a university email-security team: model scores are translated into deliver, review, and quarantine actions, then checked against a held-out test set.

**Public site:** https://jackwilson05.github.io/email-triage-visualization/

## Repository contents

- `src/build_merged_dataset.py`: parses raw RFC-style emails, validates labels, and joins the parsed records to the downloaded label table.
- `src/train_and_evaluate_models.py`: trains the six shared-input model variants and exports safe website data.
- `src/make_static_figures.py`: exports the Python figures used in the report and site appendix.
- `src/build_offline_presentation.py`: creates an offline HTML backup of the presentation.
- `docs/`: public GitHub Pages website, including D3, safe derived JSON, and static assets.
- `data/submission/`: safe, tabular dataset exports used for the assignment submission. See `data_dictionary.md` in that folder.
- `figures/`: Python-generated analysis figures.
- `submission/presentation_offline.html`: network-free copy of the web presentation.

## Data sources

- [Spam and Ham Assassin Corpus Dataset on Kaggle](https://www.kaggle.com/datasets/haseebekhan/spam-and-ham-assassin-corpus-dataset)
- [SpamAssassin Public Mail Corpus](https://spamassassin.apache.org/old/publiccorpus/)

The public site and the tracked submission datasets contain anonymous IDs, labels, engineered structural/text-style features, model scores, and aggregate metrics. They do not include raw message bodies, email addresses, filenames, or original Message-ID values.

## Run the full pipeline locally

The raw archive is intentionally excluded from GitHub. Download/extract it under `data/raw/archive/` so these paths exist:

- `data/raw/archive/Spamdataset.csv`
- `data/raw/archive/spam_data/spam_data/`

Install requirements, then run:

```bash
python -m pip install -r requirements.txt
python src/build_merged_dataset.py
python src/train_and_evaluate_models.py
python src/make_static_figures.py
python src/build_offline_presentation.py
```

To preview the site locally:

```bash
python -m http.server 8000 --directory docs
```

Open `http://localhost:8000`.

## Deploy with GitHub Pages

Push this repository to the `main` branch. The included GitHub Actions workflow publishes `docs/` to GitHub Pages. In repository **Settings → Pages**, select **GitHub Actions** as the source if it is not already enabled.
