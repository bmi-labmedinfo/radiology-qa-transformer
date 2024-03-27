# FINE-TUNING
import math
import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, TrainerCallback, EarlyStoppingCallback
from configuration import Config
from dataset_creation import *
from metrics import *
config = Config()

config.POST_PROCESSING = False

class EvaluationCallback(TrainerCallback):
    def __init__(self, multitask_dataset, tokenized_datasets):
        self.multitask_dataset = multitask_dataset
        self.tokenized_datasets = tokenized_datasets

    def on_step_end(self, args, state, control, model, tokenizer, logs=None, **kwargs):
        steps_per_epoch = len(self.tokenized_datasets["train"]) // (config.BATCH_SIZE * config.ACCUMULATION_STEP)
        current_epoch = math.floor(state.epoch)
        current_step = state.global_step
        if (current_epoch == 0):
            splits = [0.01, 0.1, 0.3, 0.5, 0.8, 1]
        elif (current_epoch == 1):
            splits = [0.25, 0.5, 0.75, 1]
        elif (current_epoch == 2):
            splits = [0.33, 0.66, 1]
        else:
            splits = [0.5, 1]
        steps_to_check = [round(step) for step in
                          np.add(steps_per_epoch * current_epoch, np.multiply(splits, steps_per_epoch))]
        if (current_step in steps_to_check):
            unconst_res = validate(model, tokenizer, self.multitask_dataset["dev"], constrained=False, verbose=False, opt=[])
            const_res = validate(model, tokenizer, self.multitask_dataset["dev"], constrained=True, verbose=False, opt=[])
            line = {
                "epoch": state.epoch,
                "step": current_step,
                "examples seen": round((state.epoch % 1) * len(self.multitask_dataset["train"])),
                "overall": {
                    "unconstrained": {
                        "F1": unconst_res["metric"]["overall"]["f1"],
                        "SA": unconst_res["metric"]["overall"]["strict_acc"],
                        "FA": unconst_res["metric"]["overall"]["format_acc"]
                    },
                    "constrained": {
                        "F1": const_res["metric"]["overall"]["f1"],
                        "SA": const_res["metric"]["overall"]["strict_acc"],
                        "FA": const_res["metric"]["overall"]["format_acc"],
                    }
                },
                "factual": {
                    "unconstrained": {
                        "F1": unconst_res["metric"]["factual"]["f1"],
                        "SA": unconst_res["metric"]["factual"]["strict_acc"],
                        "FA": unconst_res["metric"]["factual"]["format_acc"]
                    },
                    "constrained": {
                        "F1": const_res["metric"]["factual"]["f1"],
                        "SA": const_res["metric"]["factual"]["strict_acc"],
                        "FA": const_res["metric"]["factual"]["format_acc"]
                    }
                },
                "free-text": {
                    "unconstrained": {
                        "F1": unconst_res["metric"]["free-text"]["f1"],
                        "SA": unconst_res["metric"]["free-text"]["strict_acc"],
                        "FA": unconst_res["metric"]["free-text"]["format_acc"]
                    },
                    "constrained": {
                        "F1": const_res["metric"]["free-text"]["f1"],
                        "SA": const_res["metric"]["free-text"]["strict_acc"],
                        "FA": const_res["metric"]["free-text"]["format_acc"]
                    }
                },
                "multichoice": {
                    "unconstrained": {
                        "F1": unconst_res["metric"]["multichoice"]["f1"],
                        "SA": unconst_res["metric"]["multichoice"]["strict_acc"],
                        "FA": unconst_res["metric"]["multichoice"]["format_acc"]
                    },
                    "constrained": {
                        "F1": const_res["metric"]["multichoice"]["f1"],
                        "SA": const_res["metric"]["multichoice"]["strict_acc"],
                        "FA": const_res["metric"]["multichoice"]["format_acc"]
                    }
                }
            }
            wandb.log(line, step=current_step)

def batch_truncation_finetuning(k):
  output_dir = config.OUTPUT_DIR+"batch-truncation/"
  wandb_project_name = "IT5-RadiologyReports_Batch-Truncation"
  seed = config.SEEDS[0]
  multitask_dataset = bt_k_fold_splitting(k, seed)
  # DATA TOKENIZATION
  tokenizer = config.TOKENIZER
  tokenized_datasets = multitask_dataset.map(preprocess_function, batched=True)
  tokenized_datasets = tokenized_datasets.remove_columns(multitask_dataset["train"].column_names)

  model_name = config.MODEL_CHECKPOINT.split("/")[-1]

  # -----------------------------------------------------------------------
  # <<<<<< FINE-TUNING >>>>>>
  # WANDB MONITOR
  wandb.init(
      project=wandb_project_name + "_finetuning",
      name=model_name + "_finetuning_" + str(k) + "-fold-eval/loss",
      notes="radiology dataset multiple runs",
      tags=["t5", "italian", "radiology", "multitask", "question answering", "5-CV"],
      config={
          "epochs": config.EPOCHS,
          "learning_rate": config.LEARNING_RATE,
          "batch_size": config.BATCH_SIZE * config.ACCUMULATION_STEP,
          "model_checkpoint": config.MODEL_CHECKPOINT,
          "random_seed": seed
      },
      reinit=True
  )

  args = Seq2SeqTrainingArguments(
      save_strategy="epoch",
      load_best_model_at_end=True,
      metric_for_best_model="loss",
      greater_is_better=False,
      output_dir=output_dir + "output_" + str(k) + "_eval-loss",
      evaluation_strategy="epoch",
      learning_rate=config.LEARNING_RATE,
      per_device_train_batch_size=config.BATCH_SIZE,
      per_device_eval_batch_size=config.BATCH_SIZE,
      weight_decay=config.WEIGHT_DECAY,
      save_total_limit=config.SAVE_TOTAL_LIMIT,
      num_train_epochs=config.EPOCHS,
      predict_with_generate=True,
      logging_steps=len(tokenized_datasets["train"]) // config.BATCH_SIZE,
      gradient_accumulation_steps=config.ACCUMULATION_STEP,
      fp16=config.FP16 if config.DEVICE == 'cuda' else False,
      push_to_hub=False,
      report_to="wandb",
      run_name=f"{model_name}_fold{k}"
  )
  print("---->OUTPUT_DIR:", args.output_dir)
  model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_CHECKPOINT)
  data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
  trainer = Seq2SeqTrainer(
      model,
      args,
      train_dataset=tokenized_datasets["train"],
      eval_dataset=tokenized_datasets["dev"],
      data_collator=data_collator,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics,
      callbacks=[EvaluationCallback(multitask_dataset, tokenized_datasets),
                 EarlyStoppingCallback(early_stopping_patience=config.PATIENCE)]
  )

  trainer.train()
  trainer.save_model(args.output_dir)
  print(f"Model fold {k}: SAVED!")

  # CLEAR
  del model, trainer, model_name, data_collator, tokenized_datasets, args

  wandb.finish()
  torch.cuda.empty_cache()

def ex_post_combination_finetuning(k):
  output_dir = config.OUTPUT_DIR+"ex-post-comb/"
  wandb_project_name = "IT5-RadiologyReports_Ex-post-Combination"
  seed = config.SEEDS[0]

  #DATA TOKENIZATION and PREPARATION
  tokenizer = config.TOKENIZER
  multitask_dataset = epc_k_fold_splitting(k, seed, tokenizer)
  tokenized_datasets = multitask_dataset.map(preprocess_function, batched = True)
  tokenized_datasets = tokenized_datasets.remove_columns(multitask_dataset["train"].column_names)

  model_name = config.MODEL_CHECKPOINT.split("/")[-1]

  #-----------------------------------------------------------------------
  # <<<<<< FINE-TUNING >>>>>>
  #WANDB MONITOR
  wandb.init(
      project = wandb_project_name,
      name = model_name+"_finetuning_"+str(k)+"-fold-eval/loss",
      notes="radiology dataset multiple runs (context splitted in batches)",
      tags=["t5", "italian", "radiology", "multitask","question answering","5-CV"],
      config = {
        "epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE*config.ACCUMULATION_STEP,
        "model_checkpoint":config.MODEL_CHECKPOINT,
        "random_seed":seed
      },
      reinit = True
  )

  args = Seq2SeqTrainingArguments(
      save_strategy = "epoch",
      load_best_model_at_end = True,
      metric_for_best_model = "loss",
      greater_is_better = False,
      output_dir= output_dir + "output_" + str(k)+"_eval-loss",
      evaluation_strategy="epoch",
      learning_rate=config.LEARNING_RATE,
      per_device_train_batch_size=config.BATCH_SIZE,
      per_device_eval_batch_size=config.BATCH_SIZE,
      weight_decay=config.WEIGHT_DECAY,
      save_total_limit=config.SAVE_TOTAL_LIMIT,
      num_train_epochs=config.EPOCHS,
      predict_with_generate=True,
      logging_steps=len(tokenized_datasets["train"]) // config.BATCH_SIZE,
      gradient_accumulation_steps = config.ACCUMULATION_STEP,
      fp16=config.FP16 if config.DEVICE=='cuda' else False,
      push_to_hub=False,
      report_to="wandb",
      run_name=f"{model_name}_fold{k}"
  )
  print("---->OUTPUT_DIR:",args.output_dir)
  model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_CHECKPOINT)
  data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
  trainer = Seq2SeqTrainer(
      model,
      args,
      train_dataset=tokenized_datasets["train"],
      eval_dataset=tokenized_datasets["dev"],
      data_collator=data_collator,
      tokenizer=tokenizer,
      compute_metrics=compute_metrics,
      callbacks=[EvaluationCallback(multitask_dataset, tokenized_datasets),
                 EarlyStoppingCallback(early_stopping_patience=config.PATIENCE)]
  )

  trainer.train()
  trainer.save_model(args.output_dir)
  print(f"Model fold {k}: SAVED!")

  #CLEAR
  del model, trainer, model_name, data_collator, tokenized_datasets, args

  wandb.finish()
  torch.cuda.empty_cache()

def gs_validate(model, tokenizer, dataset, param_grid, verbose=True):
    res_list=[]
    results_table = pd.DataFrame([], columns=["repetition_penalty","num_beams","length_penalty","conf_threshold","f1","sa","tot_score"])
    rep_penalty,n_beams,length_pen = param_grid
    model_dev = model.device.type
    predictions = []; references = []; confidences = []

    # predict for each example
    for example in tqdm(dataset, leave = True, disable = False):
      generated_output = model.generate(tokenizer.encode(example["input_text"], return_tensors = "pt").to(model_dev),
                                        max_length=config.MAX_LEN, repetition_penalty=rep_penalty, num_beams=n_beams,
                                        length_penalty=length_pen)
      pred = tokenizer.decode(generated_output[0], skip_special_tokens = True)
      ref = example["output_text"]

      # confidence
      conf = compute_f1(pred,ref,tokenizer)
      confidences.append(conf)

      references.append(ref)
      predictions.append(pred)

    prediction_table = pd.DataFrame({"id":dataset["id"],"ref":references,"pred":predictions,"conf":confidences})
    unique_ids = list(prediction_table["id"].unique())
    conf_grid = [0.09,0.2,0.4]
    for conf_threshold in conf_grid:
      list_ref = []; list_input = []; list_pred = []
      for id in np.arange(len(unique_ids)):
        row_idx = list(prediction_table.index[prediction_table['id'] == unique_ids[id]])
        list_ref.append(prediction_table.iloc[row_idx[0],list(prediction_table.columns).index("ref")])

        conf = list(prediction_table.iloc[row_idx,list(prediction_table.columns).index("conf")]) # batches' confidence of one id
        if 1 in conf:
          max_conf_idx = conf.index(max(conf)) # index (in sample_table) of the batch with max confidence
          pred = prediction_table.iloc[row_idx[max_conf_idx],list(prediction_table.columns).index("pred")]
        else:
          c = np.array(conf.copy())
          res = np.where(c >= conf_threshold , True, False)
          if res.any():
            pred = ""
            for i in np.arange(len(row_idx)):
              if conf[i] >= conf_threshold : pred = pred + prediction_table.iloc[row_idx[i],list(prediction_table.columns).index("pred")]
          else:
            max_conf_idx = conf.index(max(conf)) # index (in sample_table) of the batch with max confidence
            pred = prediction_table.iloc[row_idx[max_conf_idx],list(prediction_table.columns).index("pred")]

        list_pred.append(pred)

      final_sample_table = pd.DataFrame(zip(unique_ids,list_ref,list_pred), columns = ["id","ref","best_pred"])

      exact_match = [i==j for i,j in zip(final_sample_table["best_pred"],final_sample_table["ref"])]
      strict_acc = round(100*len(np.where(exact_match)[0])/len(exact_match),2)
      f1 = round(100*np.mean([compute_f1(str(i),str(j),tokenizer) for i,j in zip(final_sample_table["best_pred"],final_sample_table["ref"])]),2)

      print({"repetition_penalty": rep_penalty,"num_beams": n_beams, "length_penalty":length_pen, "conf_threshold":conf_threshold,
             "strict_acc":strict_acc,"f1":f1, "total_score":strict_acc+f1})
      results_dic = [rep_penalty,n_beams,length_pen,conf_threshold,strict_acc,f1,strict_acc+f1]
      res_list.append(results_dic)
      del f1,strict_acc,list_ref, list_input, list_pred
    return res_list

# Hyperparameters tuning on dev set
def tuning_hyperparam(k):
    grid_vals = {"repetition_penalty": [1, 1.2, 1.6], "num_beams": [2, 3, 4], "length_penalty": [2, 3]}
    k_fold_scores = []
    grid_scores = []
    seed = config.SEEDS[0]

    output_dir = config.OUTPUT_DIR + "ex-post-comb/"
    ft_model = AutoModelForSeq2SeqLM.from_pretrained(output_dir + "output_" + str(k) + "_eval-loss")
    ft_tokenizer = AutoTokenizer.from_pretrained(output_dir + "output_" + str(k) + "_eval-loss")
    multitask_dataset = epc_k_fold_splitting(k, seed, ft_tokenizer)
    sample_test = multitask_dataset["dev"]
    ft_sample_test_index = []
    for t in np.arange(len(sample_test)):
        if sample_test["type"][t] == "free-text": ft_sample_test_index.append(t)
    ft_sample_test = sample_test.select(ft_sample_test_index)

    k_fold_scores.append(gs_validate(ft_model, ft_tokenizer, ft_sample_test, [1, 1, 1], verbose=True))

    for rp in grid_vals["repetition_penalty"]:
        for nb in grid_vals["num_beams"]:
            for lp in grid_vals["length_penalty"]:
                param_grid = [rp, nb, lp]
                print(rp, nb, lp)
                k_fold_scores.append(gs_validate(ft_model, ft_tokenizer, ft_sample_test, param_grid, verbose=True))

    k_fold_list = []
    for x in k_fold_scores:
        k_fold_list = k_fold_list + x
    param_table_k = pd.DataFrame(k_fold_list,
                                 columns=["repetition_penalty", "num_beams", "length_penalty", "conf_threshold",
                                          "strict_acc", "f1", "total_score"])
    param_table_k.to_csv(config.RESULTS_DIR + "/ex-post-comb/finetuned/" + "EPC_param_table_" + str(k) + ".csv")

    max_values = param_table_k.loc[param_table_k["total_score"] == max(param_table_k["total_score"])]
    print(max_values)

