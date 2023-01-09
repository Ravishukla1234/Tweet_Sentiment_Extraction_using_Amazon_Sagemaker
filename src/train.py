# train.py

import pandas as pd
import numpy as np
import os
import json
import argparse
import random

os.system("pip install datasets")
os.system("pip install transformers==4.18")
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
)


def _train(args):
    """
    run training
    :param args  : argparse namespace
    """
    print(f"Model will be saved in - {args.model_dir}")
    print(f"Path to data folder - {args.data_dir}")
    print(f"Contents of folder {args.data_dir} - {os.listdir(args.data_dir)}")

    # read input
    training_dir = args.data_dir
    csv_file = [os.path.join(training_dir, file) for file in os.listdir(training_dir)][
        0
    ]
    print(csv_file)
    train_df = pd.read_csv(csv_file)
    train_df["text"] = train_df["text"].astype(str)
    train_df["selected_text"] = train_df["selected_text"].astype(str)
    answers_start = []
    for index, row in train_df.iterrows():
        answers_start.append(row["text"].find(row["selected_text"]))
    train_df["answer_start"] = answers_start
    print(f"Shape of training dataset - {train_df.shape}")
    train = Dataset.from_pandas(train_df)

    def preprocess_function(data):
        context = [q.strip() for q in data["text"]]
        questions = data["sentiment"]
        answers = data["selected_text"]

        #     print(type(answers))

        #     print(questions)
        inputs = tokenizer(
            questions,
            context,
            max_length=250,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            #         print(data["answer_start"][i])
            #         print(data["selected_text"][i])
            start_char = data["answer_start"][i]
            end_char = data["answer_start"][i] + len(data["selected_text"][i])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if (
                offset[context_start][0] > end_char
                or offset[context_end][1] < start_char
            ):
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_data = train.map(preprocess_function, batched=True)
    train_data = train_data.train_test_split(test_size=0.2)
    data_collator = DefaultDataCollator()
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=False,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data["train"],
        eval_dataset=train_data["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print(f"Saving model to -- {args.model_dir}")
    trainer.save_model(args.model_dir)
    print("Training Finished -- ")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="W",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        metavar="E",
        help="bert model to train (default: bert-base-uncased)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="E",
        help="number of total epochs to run (default: 2)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        metavar="BS",
        help="batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        metavar="LR",
        help="initial learning rate (default: 1e-4)",
    )

    parser.add_argument(
        "--dist_backend",
        type=str,
        default="gloo",
        help="distributed backend (default: gloo)",
    )

    parser.add_argument("--hosts", type=json.loads, default=os.environ["SM_HOSTS"])
    parser.add_argument(
        "--current-host", type=str, default=os.environ["SM_CURRENT_HOST"]
    )
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])

    parser.add_argument(
        "--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"]
    )
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    _train(parser.parse_args())


if __name__ == "__main__":
    main()
