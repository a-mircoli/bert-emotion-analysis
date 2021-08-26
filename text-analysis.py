#Codice eseguibile su Colab

# Authenticating in drive account and mounting drive filesystem
from google.colab import auth
from google.colab import drive

auth.authenticate_user()
drive.mount('/gdrive')


# library based on the Transformers library by HuggingFace
!pip install simpletransformers


import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from simpletransformers.classification import ClassificationModel
from simpletransformers.classification import ClassificationArgs
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

data = pd.read_excel('/gdrive/.../training_set.xlsx')
data.columns = ['text','labels']


model_args = ClassificationArgs(labels_list= ["joy", "anger", "sadness", "fear"],
                                learning_rate=2e-5,
                                train_batch_size=8,
                                eval_batch_size=8,
                                max_seq_length=95,
                                num_train_epochs=6,
                                adam_epsilon=1e-8,
                                overwrite_output_dir = True,
                                evaluate_during_training=True,

                                evaluate_during_training_verbose=True,
                                #wandb_project='test1',
                                output_dir = '/content')
                                

# Defining Model
model = ClassificationModel('bert', 'bert-base-uncased',use_cuda=True, args=model_args, num_labels=4)


train_df, valid_df = train_test_split(data, test_size=0.2, stratify=data['labels'], random_state=42)

# train the model
model.train_model(train_df, eval_df=valid_df, show_running_loss = True, acc = accuracy_score)


#testing
data_test = pd.read_excel('/gdrive/.../testing_set.xlsx')

test_predictions_split, raw_outputs_split = model.predict(data_test.text.to_list())
i = 0
true = 0
wrong = 0
for el in data_test.values.tolist():

  if (test_predictions_split[i] == el[1]):
    true = true + 1
  else:
    wrong = wrong + 1
  i = i + 1

print('true', true)
print('wrong', wrong)
print('acc', true/len(data_test.text)*100)