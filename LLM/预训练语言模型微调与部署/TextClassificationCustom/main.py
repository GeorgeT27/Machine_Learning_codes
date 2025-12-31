import argparse
import os
import numpy as np
from email import parser
import json
from utils_data import load_data,TextDataset
from torch.utils.data import random_split
from peft import loraconfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    default_data_collator
)
def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    args=parser.parse_args()
    return args
def main():
    args=parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    set_seed(args.seed)
    
    # Load Model and Tokenizer
    config=AutoConfig.from_pretrained(args.model)
    tokeniser=AutoTokenizer.from_pretrained(args.model)
    model=BertForSequenceClassification.from_pretrained(args.model, config=config) 
    # Add LoRA
    peft_config=loraconfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=['query','value']
    )
    model =get_peft_model(model,peft_config)
    model.print_trainable_parameters()
    # Prepare Datasets
    train_val_data=load_data(args,'train')
    train_val_dataset=TextDataset(train_val_data,tokeniser,args.max_length,is_test=False)
    train_dataset, eval_dataset = random_split(train_val_dataset, [int(0.8*len(train_val_dataset)), len(train_val_dataset)-int(0.8*len(train_val_dataset))])
    test_data=load_data(args,'test')
    test_dataset=TextDataset(test_data,tokeniser,args.max_length,is_test=True)
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        correct = ((preds == p.label_ids).sum()).item()
        return {'accuracy': 1.0*correct/len(preds)}
    # Define Training Arguments and Trainer
    training_args = TrainingArguments(
            output_dir = args.output_dir,
            do_train=True,
            do_eval=True,
            do_predict=True,
            logging_strategy="steps",
            save_strategy="epoch",
            learning_rate= args.lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.epoch,
            report_to="none"
        )
    trainer= Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokeniser,
        data_collator=default_data_collator,
    )
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        predictions = trainer.predict(test_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    writer.write(f"{index}\t{item}\n")

if __name__ == "__main__":
    main()
        
    
    
    