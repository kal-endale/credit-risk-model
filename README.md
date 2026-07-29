# Fintech Review Analytics

An end-to-end customer-experience analytics project examining Google Play Store reviews for three Ethiopian mobile banking applications:

- Commercial Bank of Ethiopia
- Bank of Abyssinia
- Dashen Bank

The project transforms public app reviews into sentiment indicators, recurring customer-experience themes, structured database records, visual insights, and product recommendations.

## Business Objective

Omega Consultancy requires an evidence-based understanding of what mobile banking customers value, what frustrates them, and which product improvements Ethiopian banks should prioritize.

The project covers:

1. Google Play review collection and preprocessing
2. Sentiment and thematic analysis
3. PostgreSQL database engineering
4. Visualization and bank-specific recommendations

## Applications Analyzed

| Bank | Application | Google Play App ID |
|---|---|---|
| Commercial Bank of Ethiopia | CBE Mobile Banking | `com.combanketh.mobilebanking` |
| Bank of Abyssinia | BoA Mobile | `com.boa.boaMobileBanking` |
| Dashen Bank | Dashen Bank Super App | `com.dashen.dashensuperapp` |

## Project Structure

```text
fintech-review-analytics/
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .vscode/
│   └── settings.json
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── scripts/
│   └── scrape_reviews.py
├── src/
│   ├── data_collection.py
│   └── preprocessing.py
├── tests/
│   └── test_preprocessing.py
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt


# Task 2: Sentiment and Thematic Analysis

## Sentiment Methodology

Review sentiment was classified using the Hugging Face
`distilbert-base-uncased-finetuned-sst-2-english` model.

The model produces positive and negative predictions with a confidence score. Because the assignment requires positive, negative, and neutral labels, predictions with confidence below 0.70 were classified as neutral.

A signed polarity score was calculated as:

- Positive: positive confidence
- Negative: negative confidence multiplied by −1
- Neutral: 0

The model was executed in batches, with text truncated to the model's maximum supported sequence length.

## Overall Sentiment Results

| Sentiment | Reviews | Percentage |
|---|---:|---:|
| Positive | 771 | 57.11% |
| Negative | 530 | 39.26% |
| Neutral | 49 | 3.63% |
| **Total** | **1,350** | **100.00%** |

Sentiment was successfully assigned to 100% of reviews, exceeding the assignment KPI of 90%.

## Sentiment by Bank

| Bank | Positive | Neutral | Negative |
|---|---:|---:|---:|
| Commercial Bank of Ethiopia | 58.00% | 2.89% | 39.11% |
| Bank of Abyssinia | 52.44% | 5.33% | 42.22% |
| Dashen Bank | 60.89% | 2.67% | 36.44% |

Dashen recorded the strongest sentiment profile, with the highest positive-review share and lowest negative-review share. Bank of Abyssinia recorded the lowest positive sentiment and highest negative sentiment.

## Thematic Analysis

Reviews were grouped into five business-relevant themes:

1. Account Access & Authentication
2. Transaction Performance & Reliability
3. User Experience & General Satisfaction
4. Customer Support & Service
5. Features & Functionality

Theme assignment used weighted keyword matching. Multi-word phrases received greater weight than individual words. Reviews without a specific technical or service keyword were assigned to general user experience.

### Overall Theme Distribution

| Theme | Reviews | Percentage |
|---|---:|---:|
| User Experience & General Satisfaction | 1,176 | 87.11% |
| Transaction Performance & Reliability | 93 | 6.89% |
| Account Access & Authentication | 59 | 4.37% |
| Customer Support & Service | 12 | 0.89% |
| Features & Functionality | 10 | 0.74% |

The high general-experience share reflects the large number of short reviews such as “good app,” “nice,” or “best app.” Specific technical themes were analyzed separately to avoid allowing generic comments to hide actionable issues.

## Specific Issues by Bank

| Bank | Leading Specific Issue | Reviews | Share | Mean Polarity |
|---|---|---:|---:|---:|
| Bank of Abyssinia | Transaction Performance & Reliability | 44 | 9.78% | −0.9007 |
| Commercial Bank of Ethiopia | Account Access & Authentication | 34 | 7.56% | −0.9372 |
| Dashen Bank | Transaction Performance & Reliability | 32 | 7.11% | −0.8726 |

### Bank of Abyssinia

BOA's largest specific issue was transaction performance and reliability. The theme represented 9.78% of its reviews and had a strongly negative mean polarity of −0.9007.

Negative TF-IDF results highlighted:

- not working;
- update;
- slow;
- work/working;
- fix.

These findings suggest problems involving transaction speed, app availability, and reliability following updates.

### Commercial Bank of Ethiopia

CBE's largest specific issue was account access and authentication. The theme represented 7.56% of its reviews and had a mean polarity of −0.9372.

Negative TF-IDF results highlighted:

- update;
- login;
- PIN;
- work/working.

Login and PIN terms support prioritizing authentication, access recovery, and update-related regression investigation.

### Dashen Bank

Dashen recorded the strongest overall sentiment. However, transaction performance and reliability remained its leading specific issue, representing 7.11% of reviews.

Negative TF-IDF results highlighted:

- update;
- slow;
- working/not working;
- fix;
- service.

Customer-support complaints represented only 2.00% of Dashen reviews but had an extremely negative mean polarity of −0.9980. These complaints are less frequent but potentially high severity.

## Sentiment and Star-Rating Alignment

The transformer results generally aligned with star ratings:

| Bank | 1-Star Classified Negative | 5-Star Classified Positive |
|---|---:|---:|
| Bank of Abyssinia | 89.06% | 77.43% |
| Commercial Bank of Ethiopia | 84.85% | 83.65% |
| Dashen Bank | 87.25% | 85.41% |

The imperfect agreement indicates that star ratings and written sentiment are related but not interchangeable. Differences may result from short comments, ambiguous language, multilingual content, sarcasm, or users selecting a rating inconsistent with their written review.

## Task 2 Output Files

The Task 2 pipeline generates:

```text
data/processed/review_analysis.csv
data/processed/sentiment_by_bank.csv
data/processed/sentiment_by_bank_and_rating.csv
data/processed/theme_by_bank.csv
data/processed/tfidf_keywords_by_bank.csv
data/processed/negative_tfidf_keywords_by_bank.csv
data/processed/task2_analysis_metadata.json


# Task 3: PostgreSQL Database Engineering

## Objective

Task 3 stores the cleaned and NLP-processed Google Play reviews in a relational PostgreSQL database. The implementation includes schema constraints, foreign-key integrity, transactional loading, idempotent upserts, indexes, analytical views, and verification queries.

## Database Design

```mermaid
erDiagram
    BANKS ||--o{ REVIEWS : contains

    BANKS {
        smallint bank_id PK
        varchar bank_name UK
        varchar app_name
        varchar app_id UK
        timestamptz created_at
        timestamptz updated_at
    }

    REVIEWS {
        varchar review_id PK
        smallint bank_id FK
        text review_text
        smallint rating
        date review_date
        varchar sentiment_label
        numeric sentiment_score
        numeric sentiment_polarity
        varchar model_label
        varchar identified_theme
        text theme_keywords
        varchar source
        timestamptz created_at
        timestamptz updated_at
    }

    # Task 4: Insights and Recommendations

## Executive Comparison

| Bank | Reviews | Average Rating | Positive | Negative | Mean Polarity |
|---|---:|---:|---:|---:|---:|
| Bank of Abyssinia | 450 | 3.61 | 52.44% | 42.22% | 0.1056 |
| Commercial Bank of Ethiopia | 450 | 3.79 | 58.00% | 39.11% | 0.1952 |
| Dashen Bank | 450 | 3.81 | 60.89% | 36.44% | 0.2464 |

Dashen achieved the strongest customer-experience result, while Bank of Abyssinia recorded the highest negative sentiment and lowest average rating.

## Satisfaction Drivers

### Bank of Abyssinia

- Overall app satisfaction: 221 reviews, representing 93.64% of positive feedback.
- Ease of use and navigation: 7 reviews, representing 2.97% of positive feedback.
- Supporting keywords: good, best, nice, easy, simple, and easy to use.

### Commercial Bank of Ethiopia

- Overall app satisfaction: 239 reviews, representing 91.57% of positive feedback.
- Ease of use and navigation: 9 reviews, representing 3.45% of positive feedback.
- Supporting keywords: good, best, nice, easy, simple, and easy to use.

### Dashen Bank

- Overall app satisfaction: 232 reviews, representing 84.67% of positive feedback.
- Ease of use and navigation: 29 reviews, representing 10.58% of positive feedback.
- Supporting keywords: good, nice, best, easy to use, and user friendly.

Dashen's ease-of-use signal is considerably stronger than those of CBE and BOA and should be protected during future product releases.

## Pain Points

### Bank of Abyssinia

1. Transaction performance and reliability:
   - 42 negative reviews
   - 22.11% of BOA negative reviews
   - Mean polarity: −0.9909
   - Keywords: slow, balance, transfer

2. Account access and authentication:
   - 8 negative reviews
   - Mean polarity: −0.9926
   - Keywords: password, login, activation

### Commercial Bank of Ethiopia

1. Account access and authentication:
   - 33 negative reviews
   - 18.75% of CBE negative reviews
   - Mean polarity: −0.9958
   - Keywords: PIN, login, biometric

2. Transaction performance and reliability:
   - 14 negative reviews
   - Mean polarity: −0.9930
   - Keywords: transaction, slow, transfer

### Dashen Bank

1. Transaction performance and reliability:
   - 30 negative reviews
   - 18.29% of Dashen negative reviews
   - Mean polarity: −0.9974
   - Keywords: slow, transfer, transaction

2. Account access and authentication:
   - 14 negative reviews
   - Mean polarity: −0.9982
   - Keywords: login, password, PIN

3. Customer support:
   - 9 negative reviews
   - Mean polarity: −0.9980
   - Keywords: branch, customer service, response

## Prioritized Recommendations

### Bank of Abyssinia

1. Instrument all transfer stages and monitor transaction latency, failures, balance updates, and pending states.
2. Simplify login and account activation while adding clear resend, recovery, and support-escalation options.

### Commercial Bank of Ethiopia

1. Improve PIN recovery, login stability, biometric fallback, lockout guidance, and authentication monitoring.
2. Provide clear transaction status, reference numbers, failure reasons, and safe retry instructions.

### Dashen Bank

1. Preserve the app's usability advantage while reducing transfer latency and failed or unclear transaction outcomes.
2. Introduce in-app support case tracking, chatbot triage, service-level targets, and human escalation.

## AI Chatbot Opportunities

Recommended chatbot intents include:

- transfer pending;
- transfer failed;
- OTP not received;
- login locked;
- PIN or password recovery;
- balance discrepancy;
- complaint status.

Financial disputes, suspected fraud, and unresolved transactions should be escalated to human support rather than handled entirely by automation.

## Visualizations

Task 4 generates five visualizations:

1. Sentiment distribution by bank
2. Star-rating distribution by bank
3. Specific pain points by bank
4. Satisfaction drivers by bank
5. Monthly sentiment trend

Generate them with:

```powershell
python -m scripts.create_visualizations