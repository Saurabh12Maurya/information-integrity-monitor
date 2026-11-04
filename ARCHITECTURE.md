# Phase 1 — Project architecture

## Input
Synthetic, non-personal examples with claim text, hourly mention counts, source status,
evidence status and a coarse potential-impact label.

## Processing
1. Validate and clean records.
2. Calculate change in mentions and percentage growth.
3. Apply transparent triage rules.
4. Produce a score and recommendation with a visible explanation.

## Output
A dashboard showing sample volume, evidence status, review queue, trends and recommended
human next steps.

## Human oversight
The system never declares a claim true solely from model output and never publishes a
response. An authorised reviewer checks reliable evidence and decides whether to communicate.

## Future ML extension
A text classifier may be evaluated only after obtaining an authorised, well-labelled dataset.
The model should be measured with precision, recall, F1 and a confusion matrix, with particular
attention to false positives and false negatives.
