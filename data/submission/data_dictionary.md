# Submission datasets

These files are safe, tabular exports for the final-project submission. They contain no raw email bodies, sender addresses, filenames, or original Message-ID values. `record_id` is an anonymous hash-based join key.

- `01_kaggle_labels_cleaned.csv`: cleaned label table derived from the downloaded flattened Kaggle file; 5,796 verified records.
- `02_spamassassin_parsed_features.csv`: parsed raw-email corpus, saved as safe structural and text-style features before the label is attached; 5,796 records.
- `03_joined_modeling_features.csv`: final one-to-one joined modeling table; 5,796 records and 23 columns.
- `04_model_metrics.csv`: validation and held-out metrics for the six models.
- `05_test_predictions_safe.csv`: anonymous held-out predictions and engineered fields used by the final figures.

The original download contained 5,797 flattened records. 5,796 records had a verified one-to-one match to the parsed raw corpus; 1 unmatched record was excluded. Labels were verified as 0 = legitimate/ham and 1 = spam.

## Column groups in the modeling table

- Identifiers and outcome: `record_id`, `label`
- Text-style measures: subject/body length, word count, URL count, uppercase, exclamation, dollar, and digit rates
- Time and routing: sent hour, weekday, timezone offset, recipient count, received-header hops
- Structure: HTML, multipart, reply-to, unsubscribe, attachment, and content-type fields

The source code in `src/build_merged_dataset.py` documents how these values are extracted and saved.
