import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, TextClassificationPipeline
from datasets import Dataset
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch.nn as nn
import torch

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([5.5, 0.5])).cuda()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def tokenize_dataset(data):
    return tokenizer(data["text"], max_length=512, truncation=True, padding="max_length")
                     
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    # probabilities = tf.nn.softmax(logits)
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels)
    
def visualize_confusion_matrix(y_pred_argmax, y_true):
    """

    :param y_pred_arg: This is an array with values that are 0 or 1
    :param y_true: This is an array with values that are 0 or 1
    :return:
    """

    cm = tf.math.confusion_matrix(y_true, y_pred_argmax).numpy()
    con_mat_df = pd.DataFrame(cm)
    
    print(classification_report(y_pred_argmax, y_true))

    sns.heatmap(con_mat_df, annot=True, fmt='g', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#print(classification_report(test_labels, baseline_predicted))


if __name__ == "__main__":

    datapath = 'result7.csv'
    df = pd.read_csv(datapath)
    df.dropna()
    df.head()
    df1 = df.dropna(axis=0, how='any')
    df1['label'].astype('int')
    df1['text'].astype('str')
    df1 = df1.astype({'text':'string'})
    print(df.dtypes)

    train_data = df1.sample(frac=0.8, random_state=42)
    # Testing dataset
    test_data = df1.drop(train_data.index)
    #test_data.to_csv("test_data_ds.csv")
    # Check the number of records in training and testing dataset.
    print(f'The training dataset has {len(train_data)} records.')
    print(f'The testing dataset has {len(test_data)} records.')

    hg_train_data = Dataset.from_pandas(train_data)
    hg_test_data = Dataset.from_pandas(test_data)

    hg_train_data[0]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    hg_train_data[231]
    
    # Tokenize the dataset
    dataset_train = hg_train_data.map(tokenize_dataset)
    dataset_test = hg_test_data.map(tokenize_dataset)

    print(dataset_train)
    print(dataset_test)
    print(f'The unknown token is {tokenizer.unk_token} and the ID for the unkown token is {tokenizer.unk_token_id}.')

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    training_args = TrainingArguments(
        output_dir="./paratext_all_transfer_learning_transformer/",          
        logging_dir='./paratext_all_transfer_learning_transformer/logs',            
        logging_strategy='epoch',
        logging_steps=100,    
        num_train_epochs=5,              
        per_device_train_batch_size=4,  
        per_device_eval_batch_size=4,  
        learning_rate=5e-6,
        seed=42,
        save_strategy='epoch',
        save_steps=100,
        evaluation_strategy='epoch',
        eval_steps=100,
        load_best_model_at_end=True)
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])

    trainer.train()
    
    y_test_predict = trainer.predict(dataset_test)

    # Take a look at the predictions
    print(y_test_predict)
    
    y_test_logits = y_test_predict.predictions

    # First 5 predicted probabilities
    print(y_test_logits[:5])
    
    y_test_probabilities = tf.nn.softmax(y_test_logits)

    # First 5 predicted logits
    print(y_test_probabilities[:5])
    
    # Predicted labels
    y_test_pred_labels = np.argmax(y_test_probabilities, axis=1)
    
    y_pred = y_test_pred_labels.tolist()

    # First 5 predicted probabilities
    print(y_test_pred_labels[:5])
    
    # Actual labels
    y_test_actual_labels = y_test_predict.label_ids
    
    y_actual = y_test_actual_labels.tolist()
    
    res = list(zip(y_pred, y_actual))
    df = pd.DataFrame(res, columns=['predicted', 'true'])
    test_data1 = test_data.reset_index()
    df1 = pd.concat([df, test_data1], axis=1)
    #df = pd.DataFrame.from_dict(res)
    df1.to_csv("test_results_all.csv")

    # First 5 predicted probabilities
    print(y_test_actual_labels[:5])
    
    trainer.evaluate(dataset_test)
    
    # Load f1 metric
    metric_f1 = evaluate.load("f1")

    # Compute f1 metric
    f1 = metric_f1.compute(predictions=y_test_pred_labels, references=y_test_actual_labels)
    print("f1 metric is: ", f1)
    
    
    metric_recall = evaluate.load("recall")

    # Compute recall metric
    recall = metric_recall.compute(predictions=y_test_pred_labels, references=y_test_actual_labels)
    print("recall is : ", recall)
    
    
    precision_metric = evaluate.load("precision")
    precision = precision_metric.compute(predictions=y_test_pred_labels, references=y_test_actual_labels)
    
   
    print("precision is :", precision)
    
    # Save tokenizer
    tokenizer.save_pretrained('./paratext_all_transfer_learning_transformer/')

    # Save model
    trainer.save_model('./paratext_all_transfer_learning_transformer/')
    
    fig = plt.figure(figsize=(8, 6))
    visualize_confusion_matrix(y_test_pred_labels, y_test_actual_labels)
    fig.savefig("confusion_matrix_alldata.png")
    
    #loaded_model = AutoModelForSequenceClassification.from_pretrained('./paratext_transfer_learning_transformer/')