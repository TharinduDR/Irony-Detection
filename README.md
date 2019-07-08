# Irony Detection in Arabic Tweets
This repository contains the code of RGCL's submission for IDAT 2019 Shared Task.
The task aims at detecting irony in Arabic tweets. Given a tweet, systems have to classify it as either ironic or not ironic. As far as we know, this is the first shared task on irony for the Arabic language.

| Model              | Precision | Recall| F1     |
| :-----------------:|---------: | -----:| -----: |
| Capsule            | 0.807     | 0.800 | 0.798  |
| CNN                | 0.806     | 0.801 | 0.800  |
| Pooled GRU         | 0.800     | 0.789 | 0.785  |
| Attention LSTM     | 0.788     | 0.766 | 0.760  |
| Attention LSTM GRU | 0.783     | 0.768 | 0.762  |
| Attention Capsule  | 0.776     | 0.768 | 0.764  |