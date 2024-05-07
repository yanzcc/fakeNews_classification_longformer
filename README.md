Preprocessing Pipeline:
* Data Wrangling: Utilizing AWS Canvas Data Wrangler, we efficiently handle simple transformations such as labeling, concatenating, managing missing values, and removing duplicates. This forms the initial stage of preparing our dataset for more complex operations.
* Advanced Cleaning and Tokenization: A dedicated notebook is used for deeper data cleaning, which involves removing special characters, URLs, and other non-essential elements. The text is then tokenized. Given the token limit of 4096 for the Longformer model, a decision was made to use truncation over a sliding window approach. This decision is justified as the model's inherent sliding window capability ensures that 4096 tokens typically capture sufficient context for accurate classification.
* Dataset Splitting: We structured the data into training, validation, and test sets to ensure the model is trained and evaluated under robust conditions, enhancing its ability to generalize to unseen data.

Storage Solutions
* AWS S3: Primary storage is on AWS S3, providing a scalable and secure environment for the data.
* Canvas Dataset: Leveraging AWS Canvas for dataset management allows for seamless integration and access during the model training phase.
* Local Storage: Additionally, datasets are maintained in local JSON files for quick access and backup, ensuring data integrity and flexibility in data handling.

Model Selection
* We choose to fine-tune longformer-4096 instead of more popular LLMs for text classification like BERT or DistilBERT due to its ability to handle extended lengths of text. Traditional models like BERT and its simpler variations are generally restricted to shorter text sequences (typically up to 512 tokens). This limitation can be a significant hurdle when dealing with detailed news stories that require context from a larger corpus to accurately assess their validity

Training Protocol
* Initial Setup: While AWS SageMaker was considered for its powerful GPU capabilities, the cost implications led us to opt for local training.
* Resource Optimization: Facing hardware limitations, we adjusted the training batch size from 4 to 2 and reduced the dataset size to 2,500 observations. This significantly cut down the training time from 850 hours to just 14 hours and reduced memory usage from 32 GB to 11 GB.
